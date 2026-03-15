from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - runtime dependency guard
    genai = None
    genai_types = None

from verifiquant_v2.taxonomy import TAXONOMY


INVESTMENT_ANALYSIS_TOPICS = set(TAXONOMY["investment_analysis"])


if genai_types is not None:
    VARIANT_SCHEMA = genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "m_variant": genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "question": genai_types.Schema(type=genai_types.Type.STRING),
                    "context": genai_types.Schema(type=genai_types.Type.STRING),
                    "mutation_note": genai_types.Schema(type=genai_types.Type.STRING),
                },
                required=["question", "context", "mutation_note"],
            ),
            "f_variant": genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "question": genai_types.Schema(type=genai_types.Type.STRING),
                    "context": genai_types.Schema(type=genai_types.Type.STRING),
                    "missing_fields": genai_types.Schema(
                        type=genai_types.Type.ARRAY,
                        items=genai_types.Schema(type=genai_types.Type.STRING),
                    ),
                    "mutation_note": genai_types.Schema(type=genai_types.Type.STRING),
                },
                required=["question", "context", "missing_fields", "mutation_note"],
            ),
            "e_variant": genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "question": genai_types.Schema(type=genai_types.Type.STRING),
                    "context": genai_types.Schema(type=genai_types.Type.STRING),
                    "trigger_rule_ids": genai_types.Schema(
                        type=genai_types.Type.ARRAY,
                        items=genai_types.Schema(type=genai_types.Type.STRING),
                    ),
                    "mutation_note": genai_types.Schema(type=genai_types.Type.STRING),
                },
                required=["question", "context", "trigger_rule_ids", "mutation_note"],
            ),
        },
        required=["m_variant", "f_variant", "e_variant"],
    )
else:
    VARIANT_SCHEMA = None


PROMPT_TEMPLATE = """You are generating controlled test variants for a financial QA benchmark.

Goal:
Given one clean case + one linked FIC card summary, generate exactly 3 modified variants:
1) M_variant: semantic ambiguity / intent mismatch (should trigger diagnostic type M refusal)
2) F_variant: missing required spec/input (should trigger diagnostic type F)
3) E_variant: wrong numeric binding/scale/logical inconsistency (should trigger diagnostic type E)

Constraints:
- Keep the scenario realistic and close to the original question/context.
- Do NOT change the base task domain.
- F variant should mainly remove or obscure required fields.
- E variant should keep numbers present but make them implausible/inconsistent.
- Do not provide the final numeric answer in the generated question/context.

Input case:
<BASE_CASE>
{base_case_json}
</BASE_CASE>

Linked FIC summary:
<FIC_SUMMARY>
{fic_summary_json}
</FIC_SUMMARY>

Output JSON format:
- m_variant: question/context/mutation_note
- f_variant: question/context/missing_fields/mutation_note
- e_variant: question/context/trigger_rule_ids/mutation_note

Return JSON only.
"""


@dataclass
class CaseRecord:
    sample_id: str
    question: str
    context: str
    code: str
    answer: Any
    raw: Dict[str, Any]


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
    return CaseRecord(
        sample_id=sample_id,
        question=question,
        context=context,
        code=code,
        answer=answer,
        raw=record,
    )


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).lower().strip()


def _infer_topic(text: str) -> str:
    topic_keywords = {
        "npv": ["npv", "net present value", "淨現值"],
        "irr": ["irr", "internal rate of return", "內部報酬率"],
        "roi": ["roi", "return on investment", "投資報酬率"],
        "rar": ["risk-adjusted return", "rar", "sharpe", "sortino", "風險調整"],
        "ci": ["capital investment", "ci", "資本投入"],
    }
    t = _normalize_text(text)
    best_topic = "npv"
    best_score = -1
    for topic, words in topic_keywords.items():
        score = sum(1 for w in words if w in t)
        if score > best_score:
            best_score = score
            best_topic = topic
    return best_topic


def _load_fic_cards(path: Path) -> List[Dict[str, Any]]:
    cards = _load_records(path)
    out = [
        c
        for c in cards
        if str(c.get("domain", "")).strip().lower() == "investment_analysis"
        and str(c.get("topic", "")).strip().lower() in INVESTMENT_ANALYSIS_TOPICS
    ]
    if not out:
        raise ValueError("No investment_analysis cards found in --fic-cards")
    return out


def _select_fic(case: CaseRecord, cards: List[Dict[str, Any]]) -> Dict[str, Any]:
    target_id = str(case.raw.get("target_fic_id", "")).strip()
    if target_id:
        for c in cards:
            if str(c.get("id")) == target_id:
                return c
        raise ValueError(f"{case.sample_id}: target_fic_id '{target_id}' not found")

    topic_hint = str(case.raw.get("topic", "")).strip().lower()
    if topic_hint not in INVESTMENT_ANALYSIS_TOPICS:
        topic_hint = _infer_topic(f"{case.question}\n{case.context}")

    topic_cards = [c for c in cards if str(c.get("topic", "")).strip().lower() == topic_hint]
    return topic_cards[0] if topic_cards else cards[0]


def _fic_summary(fic: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": fic.get("id"),
        "name": fic.get("name"),
        "domain": fic.get("domain"),
        "topic": fic.get("topic"),
        "inputs": fic.get("inputs", []),
        "output_var": fic.get("output_var"),
        "diagnostics": fic.get("diagnostics", {}),
        "selection_hints": fic.get("selection_hints", {}),
    }


def _call_variant_generator(
    *,
    client: Any,
    model: str,
    case: CaseRecord,
    fic: Dict[str, Any],
) -> Dict[str, Any]:
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
        ),
        fic_summary_json=json.dumps(_fic_summary(fic), ensure_ascii=False, indent=2),
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


def _make_clean(case: CaseRecord, fic: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "case_id": f"{case.sample_id}__clean",
        "source_sample_id": case.sample_id,
        "variant_type": "clean",
        "question": case.question,
        "context": case.context,
        "code": case.code,
        "gold_answer": case.answer,
        "target_fic_id": fic.get("id"),
        "target_domain": fic.get("domain"),
        "target_topic": fic.get("topic"),
        "expected_status": "success",
        "expected_diagnostic_type": None,
        "expected_requested_fields": [],
        "mutation_meta": {"mutation_rule": "none"},
    }


def _make_m(case: CaseRecord, fic: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    m = payload["m_variant"]
    return {
        "case_id": f"{case.sample_id}__M_trap",
        "source_sample_id": case.sample_id,
        "variant_type": "M_trap",
        "question": m["question"],
        "context": m["context"],
        "code": case.code,
        "gold_answer": case.answer,
        "target_fic_id": fic.get("id"),
        "target_domain": fic.get("domain"),
        "target_topic": fic.get("topic"),
        "expected_status": "refusal",
        "expected_diagnostic_type": "M",
        "expected_requested_fields": [],
        "mutation_meta": {
            "mutation_rule": "llm_semantic_ambiguity",
            "mutation_note": m.get("mutation_note", ""),
        },
    }


def _make_f(case: CaseRecord, fic: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    f = payload["f_variant"]
    missing_fields = f.get("missing_fields", [])
    if not isinstance(missing_fields, list):
        missing_fields = []
    return {
        "case_id": f"{case.sample_id}__F_trap",
        "source_sample_id": case.sample_id,
        "variant_type": "F_trap",
        "question": f["question"],
        "context": f["context"],
        "code": case.code,
        "gold_answer": case.answer,
        "target_fic_id": fic.get("id"),
        "target_domain": fic.get("domain"),
        "target_topic": fic.get("topic"),
        "expected_status": "error",
        "expected_diagnostic_type": "F",
        "expected_requested_fields": missing_fields,
        "mutation_meta": {
            "mutation_rule": "llm_missing_spec",
            "mutation_note": f.get("mutation_note", ""),
        },
    }


def _make_e(case: CaseRecord, fic: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    e = payload["e_variant"]
    trigger_rules = e.get("trigger_rule_ids", [])
    if not isinstance(trigger_rules, list):
        trigger_rules = []
    return {
        "case_id": f"{case.sample_id}__E_trap",
        "source_sample_id": case.sample_id,
        "variant_type": "E_trap",
        "question": e["question"],
        "context": e["context"],
        "code": case.code,
        "gold_answer": case.answer,
        "target_fic_id": fic.get("id"),
        "target_domain": fic.get("domain"),
        "target_topic": fic.get("topic"),
        "expected_status": "alert",
        "expected_diagnostic_type": "E",
        "expected_requested_fields": [],
        "expected_trigger_rule_ids": trigger_rules,
        "mutation_meta": {
            "mutation_rule": "llm_numeric_inconsistency",
            "mutation_note": e.get("mutation_note", ""),
        },
    }


def expand_case_with_llm(
    *,
    case: CaseRecord,
    fic: Dict[str, Any],
    client: Any,
    model: str,
) -> List[Dict[str, Any]]:
    payload = _call_variant_generator(client=client, model=model, case=case, fic=fic)
    return [
        _make_clean(case, fic),
        _make_m(case, fic, payload),
        _make_f(case, fic, payload),
        _make_e(case, fic, payload),
    ]


def run_expansion(
    *,
    input_records: Iterable[Dict[str, Any]],
    fic_cards: List[Dict[str, Any]],
    client: Any,
    model: str,
    max_records: int = 0,
) -> List[Dict[str, Any]]:
    records = list(input_records)
    if max_records > 0:
        records = records[:max_records]

    expanded: List[Dict[str, Any]] = []
    for idx, raw in enumerate(records, start=1):
        case = _to_case(raw, idx)
        fic = _select_fic(case, fic_cards)
        print(f"[expand_cases_v2] processing {case.sample_id} with fic={fic.get('id')}")
        expanded.extend(expand_case_with_llm(case=case, fic=fic, client=client, model=model))
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "LLM-based case expansion for investment_analysis topics "
            "(ci/irr/npv/rar/roi): clean + M/F/E variants."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSON or JSONL")
    parser.add_argument(
        "--fic-cards",
        required=True,
        type=Path,
        help="FIC cards JSON/JSONL (investment_analysis cards).",
    )
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
    fic_cards = _load_fic_cards(args.fic_cards)
    client = genai.Client(api_key=api_key)
    expanded = run_expansion(
        input_records=raw_records,
        fic_cards=fic_cards,
        client=client,
        model=args.model,
        max_records=args.max_records,
    )
    _dump_records(args.output, expanded)
    print(f"Wrote {len(expanded)} expanded cases to {args.output}")


if __name__ == "__main__":
    main()
