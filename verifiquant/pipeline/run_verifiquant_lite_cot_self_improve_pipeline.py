from __future__ import annotations

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
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None
    genai_types = None

from verifiquant.pipeline.run_cot_self_improve_pipeline import (
    _answer_match,
    _dump_records,
    _gold_value,
    _llm_json,
    _load_records,
    _parse_number,
    _schema_oracle_rewrite,
)


def _schema_verifiquant_lite_cot_step() -> Any:
    return genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "answer": genai_types.Schema(type=genai_types.Type.STRING),
            "needs_more_info": genai_types.Schema(type=genai_types.Type.BOOLEAN),
            "missing_information": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
            "confidence": genai_types.Schema(type=genai_types.Type.NUMBER),
            "revised_question": genai_types.Schema(type=genai_types.Type.STRING),
            "revised_context": genai_types.Schema(type=genai_types.Type.STRING),
            "intended_formula": genai_types.Schema(type=genai_types.Type.STRING),
            "diagnostic_type": genai_types.Schema(type=genai_types.Type.STRING),
            "formula_check": genai_types.Schema(type=genai_types.Type.STRING),
            "value_binding_check": genai_types.Schema(type=genai_types.Type.STRING),
            "numeric_sanity_check": genai_types.Schema(type=genai_types.Type.STRING),
            "ambiguity_check": genai_types.Schema(type=genai_types.Type.STRING),
            "reasoning_note": genai_types.Schema(type=genai_types.Type.STRING),
        },
        required=[
            "answer",
            "needs_more_info",
            "missing_information",
            "confidence",
            "revised_question",
            "revised_context",
            "intended_formula",
            "diagnostic_type",
            "formula_check",
            "value_binding_check",
            "numeric_sanity_check",
            "ambiguity_check",
            "reasoning_note",
        ],
    )


def _verifiquant_lite_cot_step(
    *,
    client: Any,
    model: str,
    question: str,
    context: str,
) -> Dict[str, Any]:
    prompt = f"""
You are a financial CoT solver using a lightweight VerifiQuant self-check framework.
There is no card retrieval, no FIC database, and no external VerifiQuant diagnostic call.
Think step by step internally, then return only the requested JSON fields.

Use this VerifiQuant-lite order:
1. M / formula-intent check: infer the correct financial formula or calculation logic from the task.
   - This replaces "find a matching card"; do NOT mention cards.
   - If the task could require materially different formulas, set needs_more_info=true.
2. F / value-binding check: identify the required variables for that formula and bind values from context.
   - If required values are missing, unclear, unparsable, or unit scale is unclear, set needs_more_info=true.
3. E / numeric-sanity check: check whether bound values look strange for the chosen formula.
   - Examples: impossible signs, percent-vs-decimal scale risk, rates outside plausible ranges, incompatible units.
4. I / semantic ambiguity check: check hidden ambiguities that may change the result.
   - Examples: beginning vs ending period, nominal vs effective rate, annual vs monthly basis, FX direction, compounding convention.

Diagnostic type policy:
- Use "M" when the main problem is formula/intent ambiguity.
- Use "F" when required values are missing or cannot be bound.
- Use "E" when values are present but numerically suspicious or inconsistent.
- Use "I" when hidden semantics/conventions need clarification.
- Use "N" only when the requested logic is outside ordinary financial calculation scope.
- Use "none" when no blocking issue remains.

Solve using the current question/context and return a tentative answer.
IMPORTANT: `answer` must be a numeric string only (for example: "10185.19"), or empty string if you cannot provide one.
Keep reasoning_note concise; do not reveal long private chain-of-thought.

Question:
{question}

Context:
{context}

Return JSON only.
"""
    return _llm_json(client, model=model, prompt=prompt, schema=_schema_verifiquant_lite_cot_step())


def _oracle_rewrite_for_verifiquant_lite_cot(
    *,
    client: Any,
    model: str,
    row: Dict[str, Any],
    current_question: str,
    current_context: str,
    cot_step_output: Dict[str, Any],
    is_correct: Optional[bool],
) -> Dict[str, str]:
    prompt = f"""
You are Oracle-in-the-loop support for a VerifiQuant-lite CoT baseline.
There is no card retrieval, no FIC database, and no external VerifiQuant diagnostic call.
You can ONLY use the logic within the ground-truth code to clarify assumptions and revise user question/context.
You must NOT rely on the final numeric ground truth result.

Goal:
- Read the CoT solver's VerifiQuant-lite self-check.
- If it found M, clarify the intended financial formula/calculation logic, not a card identity.
- If it found F, add or clarify the corresponding values/units from the ground-truth code.
- If it found E, clarify suspicious scale, sign, percentage/decimal, or unit issues.
- If it found I, disambiguate the convention/basis/direction needed for the calculation.
- Keep the revised question/context faithful to the original task and code logic.
- Do NOT output the final numeric answer directly in question/context.

Current question:
{current_question}

Current context:
{current_context}

Current CoT step output:
{json.dumps(cot_step_output, ensure_ascii=False, indent=2)}

Current correctness against gold (if available): {is_correct}

Ground-truth code:
{row.get("code", row.get("python_solution", ""))}

Return JSON only.
"""
    out = _llm_json(client, model=model, prompt=prompt, schema=_schema_oracle_rewrite())
    return {
        "updated_question": str(out.get("updated_question", "") or current_question).strip() or current_question,
        "updated_context": str(out.get("updated_context", "") or current_context).strip() or current_context,
        "notes": str(out.get("notes", "")).strip(),
    }


def _verifiquant_lite_cot_oracle_loop(
    *,
    client: Any,
    cot_model: str,
    oracle_model: str,
    row: Dict[str, Any],
    max_turns: int,
) -> Dict[str, Any]:
    question = str(row.get("question", "") or "")
    context = str(row.get("context", "") or "")
    gold = _gold_value(row)
    gold_num = _parse_number(gold)
    history: List[Dict[str, Any]] = []
    final_answer: Optional[float] = None
    final_correct: Optional[bool] = None
    final_abs_error: Optional[float] = None
    final_diagnostic_type: Optional[str] = None

    for turn in range(1, max_turns + 1):
        step = _verifiquant_lite_cot_step(
            client=client,
            model=cot_model,
            question=question,
            context=context,
        )
        ans = _parse_number(step.get("answer"))
        abs_err, is_correct = _answer_match(question, ans, gold_num)
        diagnostic_type = str(step.get("diagnostic_type", "") or "none").strip().upper()
        if diagnostic_type == "NONE":
            diagnostic_type = "none"

        final_answer = ans
        final_correct = is_correct
        final_abs_error = abs_err
        final_diagnostic_type = diagnostic_type

        history.append(
            {
                "turn": turn,
                "question": question,
                "context": context,
                "cot_step": step,
                "parsed_answer": ans,
                "diagnostic_type": diagnostic_type,
                "is_correct": is_correct,
                "abs_error": abs_err,
            }
        )

        print(f"  -> VerifiQuant-lite CoT Turn {turn}: diagnostic={diagnostic_type} is_correct={is_correct}")

        needs_more = bool(step.get("needs_more_info"))
        should_iterate = needs_more or (is_correct is False)
        if not should_iterate or turn >= max_turns:
            break

        rewrite = _oracle_rewrite_for_verifiquant_lite_cot(
            client=client,
            model=oracle_model,
            row=row,
            current_question=question,
            current_context=context,
            cot_step_output=step,
            is_correct=is_correct,
        )
        new_q = rewrite["updated_question"] or str(step.get("revised_question", "")).strip() or question
        new_c = rewrite["updated_context"] or str(step.get("revised_context", "")).strip() or context
        if new_q == question and new_c == context:
            break
        question, context = new_q, new_c

    return {
        "rounds": len(history),
        "final_answer": final_answer,
        "final_diagnostic_type": final_diagnostic_type,
        "final_is_correct": final_correct,
        "final_abs_error": final_abs_error,
        "history": history,
    }


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    correct = sum(1 for r in rows if r["verifiquant_lite_cot_self_improve"].get("final_is_correct") is True)
    improved = 0
    diagnostic_counts: Dict[str, int] = {}
    for r in rows:
        result = r["verifiquant_lite_cot_self_improve"]
        diag = str(result.get("final_diagnostic_type") or "none")
        diagnostic_counts[diag] = diagnostic_counts.get(diag, 0) + 1
        h = result.get("history", [])
        if len(h) >= 2 and h[0].get("is_correct") is False and result.get("final_is_correct") is True:
            improved += 1
    return {
        "total_cases": total,
        "correct_count": correct,
        "accuracy": (correct / total) if total else 0.0,
        "improved_count": improved,
        "improved_rate": (improved / total) if total else 0.0,
        "final_diagnostic_counts": diagnostic_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "VerifiQuant-lite CoT self-improvement pipeline "
            "(CoT + Oracle rewrite with M/F/E/I prompt self-checks, no card retrieval)."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSON/JSONL with question/context/code/result")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON/JSONL records")
    parser.add_argument("--summary-output", type=Path, help="Optional summary JSON output")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--cot-model", default="gemini-2.5-flash")
    parser.add_argument("--oracle-model", default="gemini-2.5-flash")
    args = parser.parse_args()

    if genai is None or genai_types is None:
        print("Missing dependency: google.genai import failed.", file=sys.stderr)
        sys.exit(1)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Missing GEMINI_API_KEY in environment.", file=sys.stderr)
        sys.exit(1)

    rows = _load_records(args.input)
    if args.max_records > 0:
        rows = rows[: args.max_records]
    client = genai.Client(api_key=api_key)

    out_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        cid = row.get("case_id", row.get("question_id", idx))
        print(f"[verifiquant-lite-cot-self-improve] processing {cid}")
        cot = _verifiquant_lite_cot_oracle_loop(
            client=client,
            cot_model=args.cot_model,
            oracle_model=args.oracle_model,
            row=row,
            max_turns=max(1, args.max_turns),
        )
        out_rows.append(
            {
                "case_id": cid,
                "source_sample_id": row.get("source_sample_id", row.get("question_id")),
                "verifiquant_lite_cot_self_improve": cot,
            }
        )

    _dump_records(args.output, out_rows)
    summary = _aggregate(out_rows)
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(out_rows)} VerifiQuant-lite CoT self-improve records to {args.output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
