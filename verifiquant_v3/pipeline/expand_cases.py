from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - runtime dependency guard
    genai = None
    genai_types = None


if genai_types is not None:
    VARIANT_SCHEMA = genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "m_variant": genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "question": genai_types.Schema(type=genai_types.Type.STRING),
                    "context": genai_types.Schema(type=genai_types.Type.STRING),
                    "reason": genai_types.Schema(type=genai_types.Type.STRING),
                },
                required=["question", "context", "reason"],
            ),
            "f_variant": genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "question": genai_types.Schema(type=genai_types.Type.STRING),
                    "context": genai_types.Schema(type=genai_types.Type.STRING),
                    "reason": genai_types.Schema(type=genai_types.Type.STRING),
                },
                required=["question", "context", "reason"],
            ),
            "e_variant": genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "question": genai_types.Schema(type=genai_types.Type.STRING),
                    "context": genai_types.Schema(type=genai_types.Type.STRING),
                    "reason": genai_types.Schema(type=genai_types.Type.STRING),
                },
                required=["question", "context", "reason"],
            ),
        },
        required=["m_variant", "f_variant", "e_variant"],
    )
else:
    VARIANT_SCHEMA = None


PROMPT_TEMPLATE = """You are generating controlled benchmark variants for financial reasoning error analysis.

You are given ONE clean case with:
- question
- context
- code
- answer

Generate exactly THREE modified variants:
1) m_variant (M type: misunderstanding/semantic ambiguity)
2) f_variant (F type: formula/spec mismatch, mainly due to missing required inputs/spec)
3) e_variant (E type: extraction/binding/scale inconsistency, numbers exist but are implausible/inconsistent)

Important M/F/E boundary:
- M: make intent ambiguous (e.g., user asks if project is profitable but does not specify NPV vs IRR vs Payback), OR make wording likely to retrieve wrong conceptual card.
- F: keep intent mostly clear, but remove key required information so the formula cannot be completed (e.g., missing discount rate, period count, required variable).
- E: keep intent and required fields present, but alter numeric expression/context so values are likely mis-bound or scale-wrong (e.g., 8 vs 0.08, swapped fields, inconsistent units/frequency).

Hard constraints:
- Keep scenario realistic and close to original.
- Keep topic family similar to the original case (do not switch to unrelated finance domain).
- Do NOT reveal or compute the final answer.
- Each variant must be meaningfully different from clean and from each other.
- Each variant must include a concise `reason`:
  - what was changed
  - why it should trigger that specific error type

Output requirements:
- Return JSON only (no markdown, no extra text).
- Follow exactly this shape:
  - m_variant: question, context, reason
  - f_variant: question, context, reason
  - e_variant: question, context, reason

Base case:
<BASE_CASE_JSON>
{base_case_json}
</BASE_CASE_JSON>
"""


@dataclass
class CaseRecord:
    sample_id: str
    question: str
    context: str
    code: str
    answer: Any


def _load_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported JSON payload type in {path}")


def _dump_records(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def _to_case(record: Dict[str, Any], idx: int) -> CaseRecord:
    sample_id = (
        str(record.get("sample_id") or "").strip()
        or str(record.get("question_id") or "").strip()
        or f"sample_{idx}"
    )
    question = str(record.get("question", "") or "").strip()
    context = str(record.get("context", "") or "").strip()
    code = str(record.get("code", "") or record.get("python_solution", "") or "").strip()
    answer = record.get("answer", record.get("ground_truth"))
    if not question:
        raise ValueError(f"{sample_id}: missing question")
    if answer is None:
        raise ValueError(f"{sample_id}: missing answer/ground_truth")
    return CaseRecord(sample_id=sample_id, question=question, context=context, code=code, answer=answer)


def _call_llm_variants(*, client: Any, model: str, case: CaseRecord) -> Dict[str, Any]:
    if genai is None or genai_types is None or VARIANT_SCHEMA is None:
        raise RuntimeError(
            "google.genai is not available. Please install compatible google-genai package."
        )
    prompt = PROMPT_TEMPLATE.format(
        base_case_json=json.dumps(
            {
                "sample_id": case.sample_id,
                "question": case.question,
                "context": case.context,
                "code": case.code,
                "answer": case.answer,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=VARIANT_SCHEMA,
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    try:
        return json.loads(response.text)
    except json.JSONDecodeError as err:
        raise ValueError(f"Variant generator returned invalid JSON: {response.text}") from err


def _mk_clean(case: CaseRecord) -> Dict[str, Any]:
    return {
        "case_id": f"{case.sample_id}__clean",
        "source_sample_id": case.sample_id,
        "variant_type": "clean",
        "question": case.question,
        "context": case.context,
        "code": case.code,
        "gold_answer": case.answer,
        "expected_status": "success",
        "expected_diagnostic_type": None,
        "reason": "Original clean question/context without injected trap.",
    }


def _mk_variant(case: CaseRecord, vtype: str, payload: Dict[str, Any], expected_status: str, expected_diag: str) -> Dict[str, Any]:
    return {
        "case_id": f"{case.sample_id}__{vtype}",
        "source_sample_id": case.sample_id,
        "variant_type": vtype,
        "question": payload["question"],
        "context": payload["context"],
        "code": case.code,
        "gold_answer": case.answer,
        "expected_status": expected_status,
        "expected_diagnostic_type": expected_diag,
        "reason": payload["reason"],
    }


def expand_case(*, case: CaseRecord, client: Any, model: str) -> List[Dict[str, Any]]:
    payload = _call_llm_variants(client=client, model=model, case=case)
    return [
        _mk_clean(case),
        _mk_variant(case, "M_trap", payload["m_variant"], "refusal", "M"),
        _mk_variant(case, "F_trap", payload["f_variant"], "error", "F"),
        _mk_variant(case, "E_trap", payload["e_variant"], "alert", "E"),
    ]


def run_expansion(
    *,
    input_records: Iterable[Dict[str, Any]],
    client: Any,
    model: str,
    max_records: int,
) -> List[Dict[str, Any]]:
    records = list(input_records)
    if max_records > 0:
        records = records[:max_records]
    out: List[Dict[str, Any]] = []
    for idx, raw in enumerate(records, start=1):
        case = _to_case(raw, idx)
        print(f"[expand_cases_v2] processing {case.sample_id}")
        out.extend(expand_case(case=case, client=client, model=model))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "LLM-based expansion: from clean JSON/JSONL cases to clean+M/F/E variants "
            "using only question/context/code/answer."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSON or JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON or JSONL")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()

    if genai is None:
        print(
            "Missing dependency: google.genai import failed. Please install compatible google-genai package.",
            file=sys.stderr,
        )
        sys.exit(1)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Missing GEMINI_API_KEY in environment.", file=sys.stderr)
        sys.exit(1)

    raw_records = _load_records(args.input)
    client = genai.Client(api_key=api_key)
    expanded = run_expansion(
        input_records=raw_records,
        client=client,
        model=args.model,
        max_records=args.max_records,
    )
    _dump_records(args.output, expanded)
    print(f"Wrote {len(expanded)} expanded cases to {args.output}")


if __name__ == "__main__":
    main()
