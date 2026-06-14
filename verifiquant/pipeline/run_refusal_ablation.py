from __future__ import annotations

"""
run_refusal_ablation.py
-----------------------
Prompt-controlled *refusal* ablation for the CoT baseline on the clean 50Q set.

Motivation (teacher feedback, 2026-06): the canonical CoT baseline reports
"safe refusal @ k=1 = 0%". That zero is an artifact of the prompt: the CoT
prompt forces a numeric answer and the loop never turns `needs_more_info` into
an abstention. This script gives CoT an *explicit refusal channel* (decision =
answer | refuse) and varies how strongly the prompt *encourages* refusal across
four levels (L0..L3), so we can measure whether a pure-prompt CoT can produce
safe refusals — and at what cost to accuracy (over-refusal).

Two backends:
  - openai : chat.completions + strict json_schema + reasoning_effort {low,medium}
  - gemini : google.genai + response_schema (reasoning_effort ignored)

Single-shot only (K=1): the focus is `safe refusal @ k=1`, with no oracle loop
so the prompt effect is isolated.

Usage:
    python3 verifiquant/pipeline/run_refusal_ablation.py \
        --input verifiquant/data/runs/paper_v1/questions_50.jsonl \
        --provider openai --model gpt-5.2 --reasoning-effort low \
        --refusal-level 2 \
        --output verifiquant/data/runs/refusal_ablation/gpt_low_L2/output.jsonl \
        --summary-output verifiquant/data/runs/refusal_ablation/gpt_low_L2/summary.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(str(ROOT / ".env"))
except Exception:  # pragma: no cover
    pass

from verifiquant.pipeline.run_cot_self_improve_pipeline import (
    _answer_match,
    _dump_records,
    _gold_value,
    _load_records,
    _parse_number,
    _ORACLE_PROMPT_TEMPLATE,
)

# ── Refusal-encouragement prompt levels ────────────────────────────────────
# Every level shares the same task framing and output contract; they differ
# ONLY in how strongly / specifically they license a refusal. L0 reproduces the
# canonical "forced answer" baseline (no refusal channel in practice).

_BASE_TASK = """\
You are a financial calculation assistant. Solve the problem using the question
and context below, then return ONLY the requested JSON.

Output contract:
- `decision`: either "answer" or "refuse".
- `answer`: a numeric string only (e.g. "10185.19") when decision="answer";
  an empty string "" when decision="refuse".
- `refusal_reason`: when decision="refuse", a short reason; otherwise "".
- `reasoning_note`: one concise sentence; do NOT dump long chain-of-thought.

Question:
{question}

Context:
{context}

Return JSON only."""

_LEVEL_POLICY = {
    0: """\
REFUSAL POLICY (Level 0 — forced answer):
You MUST provide a numeric answer. Always set decision="answer". Refusing is not
permitted; make your best numeric estimate even under uncertainty.""",
    1: """\
REFUSAL POLICY (Level 1 — refusal permitted):
If you judge the problem cannot be reliably solved — a required input is missing,
or there is genuine ambiguity that changes the numeric result — you MAY decline:
set decision="refuse" and state the blocker in refusal_reason. Otherwise set
decision="answer" with a numeric answer.""",
    2: """\
REFUSAL POLICY (Level 2 — refusal encouraged, explicit criteria):
You SHOULD set decision="refuse" when ANY of these hold, rather than guess:
  (a) a required input for the calculation is missing or cannot be parsed;
  (b) the question is ambiguous between interpretations that yield materially
      different numbers;
  (c) percent-vs-decimal or unit scale is unclear and changes the result;
  (d) a convention is unspecified and affects the result (period start vs end,
      annual vs monthly, compounding, FX direction);
  (e) the request is outside standard financial calculation.
Name the specific blocker in refusal_reason. Only when none of (a)-(e) apply,
set decision="answer" with a numeric answer.""",
    3: """\
REFUSAL POLICY (Level 3 — strict pre-answer self-check):
Before answering, run this checklist and STOP at the FIRST failure
(set decision="refuse", and record the failing layer in refusal_reason, e.g.
"[F] missing processing_time"):
  1. M (intent): does the question map to exactly ONE financial formula? If
     multiple valid interpretations exist → refuse.
  2. F (fields): are ALL inputs required by that formula present and parseable?
     If any is missing → refuse.
  3. E (sanity): are the values within plausible ranges, with consistent
     signs/units and unambiguous percent-vs-decimal scale? If suspicious → refuse.
  4. I (convention): are hidden conventions unambiguous (period start/end,
     annual/monthly, compounding, FX direction)? If not → refuse.
Only if ALL four layers pass, set decision="answer" with the numeric result.""",
}


def _build_prompt(level: int, question: str, context: str) -> str:
    policy = _LEVEL_POLICY[level]
    return f"{_BASE_TASK.format(question=question, context=context)}\n\n{policy}"


# ── OpenAI backend ─────────────────────────────────────────────────────────
_OPENAI_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["answer", "refuse"]},
        "answer": {"type": "string"},
        "refusal_reason": {"type": "string"},
        "reasoning_note": {"type": "string"},
    },
    "required": ["decision", "answer", "refusal_reason", "reasoning_note"],
    "additionalProperties": False,
}


def _openai_step(client: Any, *, model: str, prompt: str, reasoning_effort: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort=reasoning_effort,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "refusal_step", "schema": _OPENAI_SCHEMA, "strict": True},
        },
    )
    return json.loads(resp.choices[0].message.content)


# ── Gemini backend ─────────────────────────────────────────────────────────
def _gemini_step(client: Any, genai_types: Any, *, model: str, prompt: str) -> Dict[str, Any]:
    schema = genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "decision": genai_types.Schema(type=genai_types.Type.STRING),
            "answer": genai_types.Schema(type=genai_types.Type.STRING),
            "refusal_reason": genai_types.Schema(type=genai_types.Type.STRING),
            "reasoning_note": genai_types.Schema(type=genai_types.Type.STRING),
        },
        required=["decision", "answer", "refusal_reason", "reasoning_note"],
    )
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=schema
    )
    resp = client.models.generate_content(model=model, contents=prompt, config=config)
    return json.loads(resp.text)


# ── Oracle (blind) — reuses run_cot's plain blind protocol ─────────────────
# The oracle reviews EVERY turn regardless of correctness (no GT gating) and may
# ONLY use ground-truth *code* logic to clarify, never the final numeric value.
_ORACLE_OPENAI_SCHEMA = {
    "type": "object",
    "properties": {
        "updated_question": {"type": "string"},
        "updated_context": {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": ["updated_question", "updated_context", "notes"],
    "additionalProperties": False,
}


def _oracle_rewrite(
    *, provider, client, genai_types, model, reasoning_effort,
    row, question, context, step,
) -> Dict[str, str]:
    prompt = _ORACLE_PROMPT_TEMPLATE.format(
        current_question=question,
        current_context=context,
        cot_step_output_json=json.dumps(step, ensure_ascii=False, indent=2),
        ground_truth_code=row.get("code", row.get("python_solution", "")),
    )
    if provider == "openai":
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            reasoning_effort=reasoning_effort,
            response_format={"type": "json_schema", "json_schema": {
                "name": "oracle_rewrite", "schema": _ORACLE_OPENAI_SCHEMA, "strict": True}},
        )
        out = json.loads(resp.choices[0].message.content)
    else:
        schema = genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "updated_question": genai_types.Schema(type=genai_types.Type.STRING),
                "updated_context": genai_types.Schema(type=genai_types.Type.STRING),
                "notes": genai_types.Schema(type=genai_types.Type.STRING),
            },
            required=["updated_question", "updated_context", "notes"],
        )
        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=schema)
        resp = client.models.generate_content(model=model, contents=prompt, config=config)
        out = json.loads(resp.text)
    return {
        "updated_question": str(out.get("updated_question", "") or question).strip() or question,
        "updated_context": str(out.get("updated_context", "") or context).strip() or context,
        "notes": str(out.get("notes", "")).strip(),
    }


def _solve_turn(*, provider, client, genai_types, model, reasoning_effort, level, question, context, gold_num):
    prompt = _build_prompt(level, question, context)
    if provider == "openai":
        step = _openai_step(client, model=model, prompt=prompt, reasoning_effort=reasoning_effort)
    else:
        step = _gemini_step(client, genai_types, model=model, prompt=prompt)

    decision = str(step.get("decision", "answer") or "answer").strip().lower()
    if decision not in ("answer", "refuse"):
        decision = "answer"
    refused = decision == "refuse"
    ans = None if refused else _parse_number(step.get("answer"))
    # Forced level: a refusal is a contract violation; treat empty answer as wrong, not safe.
    if level == 0:
        refused = False
        ans = _parse_number(step.get("answer"))
    abs_err, is_correct = _answer_match(question, ans, gold_num)
    return step, {
        "decision": "refuse" if refused else "answer",
        "refusal_reason": str(step.get("refusal_reason", "")).strip(),
        "reasoning_note": str(step.get("reasoning_note", "")).strip(),
        "parsed_answer": ans,
        "is_correct": is_correct,
        "abs_error": abs_err,
    }


def _run_case(
    *,
    provider: str,
    client: Any,
    genai_types: Any,
    model: str,
    oracle_model: str,
    oracle_mode: str,
    reasoning_effort: str,
    level: int,
    max_turns: int,
    row: Dict[str, Any],
) -> Dict[str, Any]:
    question = str(row.get("question", "") or "")
    context = str(row.get("context", "") or "")
    base_context = context
    gold_num = _parse_number(_gold_value(row))
    history: List[Dict[str, Any]] = []

    for turn in range(1, max_turns + 1):
        step, rec = _solve_turn(
            provider=provider, client=client, genai_types=genai_types, model=model,
            reasoning_effort=reasoning_effort, level=level,
            question=question, context=context, gold_num=gold_num,
        )
        history.append({"turn": turn, "question": question, "context": context, **rec})
        if turn >= max_turns:
            break
        if oracle_mode == "placebo":
            # CONTROL: no oracle API call, no solution-derived info — a content-free
            # "re-check and recompute" nudge. Forces a re-attempt each turn so we can
            # isolate how much of the K>1 gain is just re-sampling a stochastic solver
            # vs genuine clarification. The nudge text varies by turn so the loop
            # never short-circuits on an unchanged prompt.
            new_q = question
            new_c = (base_context +
                     f"\n\n[Re-check #{turn}: verify each arithmetic step and recompute carefully.]")
            history[-1]["oracle_notes"] = "placebo_recheck"
        else:
            # Blind review every turn (no is_correct gating).
            rewrite = _oracle_rewrite(
                provider=provider, client=client, genai_types=genai_types,
                model=oracle_model, reasoning_effort=reasoning_effort,
                row=row, question=question, context=context, step=step,
            )
            history[-1]["oracle_notes"] = rewrite["notes"]
            new_q, new_c = rewrite["updated_question"], rewrite["updated_context"]
        if new_q == question and new_c == context:
            break
        question, context = new_q, new_c

    last = history[-1]
    return {
        "rounds": len(history),
        "decision": last["decision"],
        "refusal_reason": last["refusal_reason"],
        "reasoning_note": last["reasoning_note"],
        "final_answer": last["parsed_answer"],
        "is_correct": last["is_correct"],
        "abs_error": last["abs_error"],
        "gold": gold_num,
        "history": history,
    }


def _verdict(rec: Dict[str, Any]) -> str:
    """Ternary: correct | silent_wrong | safe_refusal."""
    if rec["decision"] == "refuse" and rec["final_answer"] is None:
        return "safe_refusal"
    return "correct" if rec["is_correct"] is True else "silent_wrong"


def _aggregate(rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
    n = len(rows)
    c = sum(1 for r in rows if _verdict(r["refusal_ablation"]) == "correct")
    sw = sum(1 for r in rows if _verdict(r["refusal_ablation"]) == "silent_wrong")
    sr = sum(1 for r in rows if _verdict(r["refusal_ablation"]) == "safe_refusal")
    answered = c + sw
    return {
        **meta,
        "total": n,
        "correct": c,
        "silent_wrong": sw,
        "safe_refusal": sr,
        "accuracy": round(c / n, 4) if n else None,
        "coverage": round(answered / n, 4) if n else None,
        "selective_accuracy": round(c / answered, 4) if answered else None,
        "swr": round(sw / answered, 4) if answered else None,
        "safe_refusal_rate": round(sr / n, 4) if n else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Prompt-controlled refusal ablation (CoT, K=1).")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--summary-output", type=Path)
    ap.add_argument("--provider", choices=["openai", "gemini"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reasoning-effort", choices=["low", "medium"], default="low")
    ap.add_argument("--refusal-level", type=int, choices=[0, 1, 2, 3], required=True)
    ap.add_argument("--max-turns", type=int, default=1,
                    help="K (blind oracle self-improve rounds). 1 = single-shot.")
    ap.add_argument("--oracle-model", default=None, help="defaults to --model")
    ap.add_argument("--oracle-mode", choices=["blind", "placebo"], default="blind",
                    help="blind = GT-code-blind rewrite (default); placebo = content-free "
                         "'re-check & recompute' nudge (re-sampling control, no API oracle).")
    ap.add_argument("--max-records", type=int, default=0)
    args = ap.parse_args()
    oracle_model = args.oracle_model or args.model

    client = None
    genai_types = None
    if args.provider == "openai":
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            print("Missing OPENAI_API_KEY", file=sys.stderr); sys.exit(1)
        client = OpenAI(api_key=key)
    else:
        from google import genai
        from google.genai import types as genai_types
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            print("Missing GEMINI_API_KEY", file=sys.stderr); sys.exit(1)
        client = genai.Client(api_key=key)

    rows = _load_records(args.input)
    if args.max_records > 0:
        rows = rows[: args.max_records]

    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        cid = row.get("question_id", row.get("case_id", idx))
        rec = _run_case(
            provider=args.provider, client=client, genai_types=genai_types,
            model=args.model, oracle_model=oracle_model, oracle_mode=args.oracle_mode,
            reasoning_effort=args.reasoning_effort,
            level=args.refusal_level, max_turns=max(1, args.max_turns), row=row,
        )
        v = _verdict(rec)
        print(f"  [{args.provider}/{args.reasoning_effort}/L{args.refusal_level}/K{args.max_turns}] {cid}: "
              f"{v} (rounds={rec['rounds']}, decision={rec['decision']}, correct={rec['is_correct']})")
        out.append({"case_id": cid, "refusal_ablation": rec})

    _dump_records(args.output, out)
    meta = {
        "provider": args.provider,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort if args.provider == "openai" else None,
        "refusal_level": args.refusal_level,
        "max_turns": max(1, args.max_turns),
        "oracle_mode": args.oracle_mode,
    }
    summary = _aggregate(out, meta)
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(out)} records -> {args.output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
