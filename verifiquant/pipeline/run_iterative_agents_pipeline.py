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

from verifiquant.pipeline.run_error_classification_pipeline import (
    ErrorClassificationAPI,
    _answer_match,
    _dump_records,
    _load_records,
    _parse_number,
)


def _schema_oracle_rewrite() -> Any:
    return genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "updated_question": genai_types.Schema(type=genai_types.Type.STRING),
            "updated_context": genai_types.Schema(type=genai_types.Type.STRING),
            "notes": genai_types.Schema(type=genai_types.Type.STRING),
        },
        required=["updated_question", "updated_context", "notes"],
    )


def _schema_cot_step() -> Any:
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
            "reasoning_note": genai_types.Schema(type=genai_types.Type.STRING),
        },
        required=[
            "answer",
            "needs_more_info",
            "missing_information",
            "confidence",
            "revised_question",
            "revised_context",
            "reasoning_note",
        ],
    )


def _llm_json(client: Any, *, model: str, prompt: str, schema: Any) -> Dict[str, Any]:
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
    )
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return json.loads(resp.text)


def _gold_value(row: Dict[str, Any]) -> Any:
    return row.get("gold_answer", row.get("answer", row.get("ground_truth")))


def _oracle_rewrite_for_framework(
    *,
    client: Any,
    model: str,
    row: Dict[str, Any],
    current_question: str,
    current_context: str,
    diagnostic: Dict[str, Any],
) -> Dict[str, str]:
    prompt = f"""
You are an Oracle-in-the-loop rewriter for VerifiQuant.
You can only use ground-truth code + result to clarify assumptions and revise user question/context.

Goal:
- Given framework diagnostic output, rewrite question/context so the next run can pass gates.
- Keep semantics faithful to original task and provided code/result.
- Do NOT output the final numeric answer directly in question/context.

Current question:
{current_question}

Current context:
{current_context}

Framework diagnostic:
{json.dumps(diagnostic, ensure_ascii=False, indent=2)}

Ground-truth code:
{row.get("code", row.get("python_solution", ""))}

Ground-truth result:
{_gold_value(row)}

Return JSON only.
"""
    out = _llm_json(client, model=model, prompt=prompt, schema=_schema_oracle_rewrite())
    return {
        "updated_question": str(out.get("updated_question", "") or current_question).strip() or current_question,
        "updated_context": str(out.get("updated_context", "") or current_context).strip() or current_context,
        "notes": str(out.get("notes", "")).strip(),
    }


def _cot_step(
    *,
    client: Any,
    model: str,
    question: str,
    context: str,
) -> Dict[str, Any]:
    prompt = f"""
You are a financial CoT solver with self-improvement hints.
Solve using the current question/context and return a tentative answer.
If information is missing/ambiguous, set needs_more_info=true and list missing items.
IMPORTANT: `answer` must be a numeric string only (for example: "10185.19"), or empty string if you cannot provide one.

Question:
{question}

Context:
{context}

Return JSON only.
"""
    return _llm_json(client, model=model, prompt=prompt, schema=_schema_cot_step())


def _oracle_rewrite_for_cot(
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
You are Oracle-in-the-loop support for a pure CoT baseline (no VerifiQuant framework).
Use only ground-truth code + result to rewrite question/context to reduce ambiguity and missing fields.

Current question:
{current_question}

Current context:
{current_context}

Current CoT step output:
{json.dumps(cot_step_output, ensure_ascii=False, indent=2)}

Current correctness against gold (if available): {is_correct}

Ground-truth code:
{row.get("code", row.get("python_solution", ""))}

Ground-truth result:
{_gold_value(row)}

Return JSON only.
"""
    out = _llm_json(client, model=model, prompt=prompt, schema=_schema_oracle_rewrite())
    return {
        "updated_question": str(out.get("updated_question", "") or current_question).strip() or current_question,
        "updated_context": str(out.get("updated_context", "") or current_context).strip() or current_context,
        "notes": str(out.get("notes", "")).strip(),
    }


def _framework_oracle_loop(
    *,
    api: ErrorClassificationAPI,
    client: Any,
    oracle_model: str,
    row: Dict[str, Any],
    max_turns: int,
) -> Dict[str, Any]:
    question = str(row.get("question", "") or "")
    context = str(row.get("context", "") or "")
    history: List[Dict[str, Any]] = []
    final_diag: Dict[str, Any] = {}

    for turn in range(1, max_turns + 1):
        diag = api.diagnose_row(
            {
                **row,
                "question": question,
                "context": context,
                "case_id": row.get("case_id", row.get("question_id", f"case_{turn}")),
            }
        )
        final_diag = diag
        history.append(
            {
                "turn": turn,
                "question": question,
                "context": context,
                "diagnostic": diag,
            }
        )
        if diag.get("status") == "success":
            break
        if turn >= max_turns:
            break
        if str(diag.get("diagnostic_type", "")).strip().upper() not in {"M", "N", "F", "E", "I"}:
            break

        rewrite = _oracle_rewrite_for_framework(
            client=client,
            model=oracle_model,
            row=row,
            current_question=question,
            current_context=context,
            diagnostic=diag,
        )
        new_q = rewrite["updated_question"]
        new_c = rewrite["updated_context"]
        if new_q == question and new_c == context:
            break
        question, context = new_q, new_c

    return {
        "rounds": len(history),
        "final_status": final_diag.get("status"),
        "final_diagnostic_type": final_diag.get("diagnostic_type"),
        "final_output_value": final_diag.get("output_value"),
        "final_is_correct": final_diag.get("is_correct"),
        "final_abs_error": final_diag.get("abs_error"),
        "history": history,
    }


def _cot_oracle_loop(
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

    for turn in range(1, max_turns + 1):
        step = _cot_step(
            client=client,
            model=cot_model,
            question=question,
            context=context,
        )
        ans = _parse_number(step.get("answer"))
        abs_err, is_correct = _answer_match(question, ans, gold_num)
        final_answer = ans
        final_correct = is_correct
        final_abs_error = abs_err

        history.append(
            {
                "turn": turn,
                "question": question,
                "context": context,
                "cot_step": step,
                "parsed_answer": ans,
                "is_correct": is_correct,
                "abs_error": abs_err,
            }
        )

        needs_more = bool(step.get("needs_more_info"))
        should_iterate = needs_more or (is_correct is False)
        if not should_iterate or turn >= max_turns:
            break

        rewrite = _oracle_rewrite_for_cot(
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
        "final_is_correct": final_correct,
        "final_abs_error": final_abs_error,
        "history": history,
    }


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    fw_success = sum(1 for r in rows if r["framework_oracle"].get("final_status") == "success")
    fw_correct = sum(1 for r in rows if r["framework_oracle"].get("final_is_correct") is True)
    cot_correct = sum(1 for r in rows if r["cot_oracle"].get("final_is_correct") is True)
    recovered = 0
    for r in rows:
        h = r["framework_oracle"].get("history", [])
        if len(h) >= 2 and h[0].get("diagnostic", {}).get("status") != "success" and r["framework_oracle"].get("final_status") == "success":
            recovered += 1
    return {
        "total_cases": total,
        "framework_success_count": fw_success,
        "framework_success_rate": (fw_success / total) if total else 0.0,
        "framework_correct_count": fw_correct,
        "framework_accuracy": (fw_correct / total) if total else 0.0,
        "framework_recovery_count": recovered,
        "framework_recovery_rate": (recovered / total) if total else 0.0,
        "cot_correct_count": cot_correct,
        "cot_accuracy": (cot_correct / total) if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run two iterative comparison pipelines: "
            "(A) VerifiQuant framework + Oracle rewriter, "
            "(B) pure CoT + Oracle rewriter."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSON/JSONL with question/context/code/result")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON/JSONL records")
    parser.add_argument("--summary-output", type=Path, help="Optional summary JSON output")
    parser.add_argument("--db-url", help="If provided, load cards from SQLAlchemy store")
    parser.add_argument("--core", type=Path, help="fic_core JSON/JSONL when --db-url is not set")
    parser.add_argument("--retrieval", type=Path, help="fic_retrieval JSON/JSONL when --db-url is not set")
    parser.add_argument("--repair", type=Path, help="repair_rule JSON/JSONL when --db-url is not set")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--m-min-top-score", type=float, default=0.05)
    parser.add_argument("--selector-model", default="gemini-2.5-flash")
    parser.add_argument("--extractor-model", default="gemini-2.5-flash")
    parser.add_argument("--judge-model", default="gemini-2.5-flash")
    parser.add_argument("--oracle-model", default="gemini-2.5-flash")
    parser.add_argument("--cot-model", default="gemini-2.5-flash")
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

    if args.db_url:
        api = ErrorClassificationAPI.from_db(
            db_url=args.db_url,
            client=client,
            selector_model=args.selector_model,
            extractor_model=args.extractor_model,
            judge_model=args.judge_model,
            top_k=args.top_k,
            m_min_top_score=args.m_min_top_score,
        )
    else:
        if not (args.core and args.retrieval and args.repair):
            raise ValueError("When --db-url is not provided, --core --retrieval --repair are required.")
        api = ErrorClassificationAPI.from_files(
            core_path=args.core,
            retrieval_path=args.retrieval,
            repair_path=args.repair,
            client=client,
            selector_model=args.selector_model,
            extractor_model=args.extractor_model,
            judge_model=args.judge_model,
            top_k=args.top_k,
            m_min_top_score=args.m_min_top_score,
        )

    out_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        cid = row.get("case_id", row.get("question_id", idx))
        print(f"[iterative] processing {cid}")
        fw = _framework_oracle_loop(
            api=api,
            client=client,
            oracle_model=args.oracle_model,
            row=row,
            max_turns=max(1, args.max_turns),
        )
        cot = _cot_oracle_loop(
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
                "framework_oracle": fw,
                "cot_oracle": cot,
            }
        )

    _dump_records(args.output, out_rows)
    summary = _aggregate(out_rows)
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(out_rows)} iterative records to {args.output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
