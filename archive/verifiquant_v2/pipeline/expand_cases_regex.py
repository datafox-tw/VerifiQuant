from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from verifiquant_v2.taxonomy import TAXONOMY


INVESTMENT_ANALYSIS_TOPICS = set(TAXONOMY["investment_analysis"])

TOPIC_KEYWORDS = {
    "npv": ["npv", "net present value", "淨現值", "present value"],
    "irr": ["irr", "internal rate of return", "內部報酬率"],
    "roi": ["roi", "return on investment", "投資報酬率"],
    "rar": ["risk-adjusted return", "rar", "風險調整", "sharpe", "sortino"],
    "ci": ["capital investment", "capital index", "ci", "資本投入"],
}


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


def _infer_topic(text: str) -> Optional[str]:
    t = _normalize_text(text)
    best_topic = None
    best_score = 0
    for topic, words in TOPIC_KEYWORDS.items():
        score = sum(1 for w in words if w in t)
        if score > best_score:
            best_score = score
            best_topic = topic
    return best_topic


def _keyword_overlap_score(question: str, fic: Dict[str, Any]) -> int:
    hay = _normalize_text(question)
    score = 0
    for inp in fic.get("inputs", []):
        name = _normalize_text(str(inp.get("name", "")))
        desc = _normalize_text(str(inp.get("description", "")))
        if name and name in hay:
            score += 2
        if desc:
            for token in re.findall(r"[a-z0-9_]+", desc):
                if len(token) > 3 and token in hay:
                    score += 1
    for tag in fic.get("tags", []):
        tag_s = _normalize_text(str(tag))
        if tag_s and tag_s in hay:
            score += 2
    return score


def _select_fic(case: CaseRecord, fic_cards: List[Dict[str, Any]]) -> Dict[str, Any]:
    if "target_fic_id" in case.raw and case.raw["target_fic_id"]:
        target = str(case.raw["target_fic_id"])
        for card in fic_cards:
            if str(card.get("id")) == target:
                return card
        raise ValueError(f"{case.sample_id}: target_fic_id '{target}' not found")

    topic_hint = str(case.raw.get("topic", "")).strip().lower()
    if topic_hint not in INVESTMENT_ANALYSIS_TOPICS:
        topic_hint = _infer_topic(f"{case.question}\n{case.context}") or ""

    candidates = [
        c
        for c in fic_cards
        if str(c.get("domain", "")).strip().lower() == "investment_analysis"
        and str(c.get("topic", "")).strip().lower() in INVESTMENT_ANALYSIS_TOPICS
    ]
    if topic_hint:
        topic_candidates = [
            c for c in candidates if str(c.get("topic", "")).strip().lower() == topic_hint
        ]
        if topic_candidates:
            candidates = topic_candidates

    if not candidates:
        raise ValueError(f"{case.sample_id}: no investment_analysis FIC candidates found")

    return max(candidates, key=lambda c: _keyword_overlap_score(case.question, c))


def _remove_rate_mentions(text: str) -> str:
    s = re.sub(r"\b\d+(\.\d+)?\s*%\b", "[MISSING_RATE]", text, flags=re.IGNORECASE)
    s = re.sub(r"\bdiscount rate\b[^,.。\n]*", "discount rate [MISSING_RATE]", s, flags=re.IGNORECASE)
    return s


def _remove_year_mentions(text: str) -> str:
    s = re.sub(r"\b\d+\s*(year|years|yr|yrs)\b", "[MISSING_PERIOD]", text, flags=re.IGNORECASE)
    s = re.sub(r"\bfor\s+\d+\s+periods?\b", "for [MISSING_PERIOD]", s, flags=re.IGNORECASE)
    return s


def _remove_currency_mentions(text: str) -> str:
    s = re.sub(r"\$\s*\d[\d,]*(\.\d+)?", "$[MISSING_AMOUNT]", text)
    s = re.sub(r"\b\d[\d,]*(\.\d+)?\s*(USD|dollars?)\b", "[MISSING_AMOUNT]", s, flags=re.IGNORECASE)
    return s


def _pick_missing_field(fic: Dict[str, Any]) -> str:
    required = [inp for inp in fic.get("inputs", []) if bool(inp.get("required", True))]
    if not required:
        return "required_input"
    name_priority = ["discount_rate", "r", "n", "periods", "cash_flows", "initial_investment"]
    names = [str(inp.get("name", "")) for inp in required]
    for key in name_priority:
        for n in names:
            if key in n:
                return n
    return names[0]


def _f_trap_transform(question: str, context: str, missing_field: str) -> Tuple[str, str]:
    q, c = question, context
    f = missing_field.lower()
    if "rate" in f or f == "r":
        q, c = _remove_rate_mentions(q), _remove_rate_mentions(c)
    elif "n" == f or "period" in f or "year" in f:
        q, c = _remove_year_mentions(q), _remove_year_mentions(c)
    elif "cash" in f or "flow" in f or "invest" in f or "x" == f:
        q, c = _remove_currency_mentions(q), _remove_currency_mentions(c)
    else:
        q = q + f" [MISSING_FIELD:{missing_field}]"
    return q, c


def _e_trap_transform(question: str, context: str, fic: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    diagnostics = fic.get("diagnostics", {}) or {}
    scale_checks = diagnostics.get("scale_checks", []) or []
    chosen = scale_checks[0] if scale_checks else None
    trigger_rule_id = str(chosen.get("id")) if chosen else "synthetic_e_rule"

    extra_note = ""
    if chosen and "frequency" in str(chosen.get("id", "")).lower():
        extra_note = (
            " Note: cash flows are monthly, but the discount rate provided is annual and not converted."
        )
    elif chosen and "rate" in str(chosen.get("id", "")).lower():
        extra_note = " Note: use discount rate as 8 (not 0.08)."
    else:
        extra_note = (
            " Note: one extracted numeric field may be in wrong scale; verify units before execution."
        )
    return question + extra_note, context, {"trigger_rule_id": trigger_rule_id}


def _build_clean_variant(case: CaseRecord, fic: Dict[str, Any]) -> Dict[str, Any]:
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


def _build_m_variant(case: CaseRecord, fic: Dict[str, Any]) -> Dict[str, Any]:
    ambiguous_q = "這個專案會不會賺錢？請幫我評估。"
    if re.search(r"\b(project|investment|return|profit)\b", case.question, flags=re.IGNORECASE):
        ambiguous_q = "Can you evaluate whether this project is worth doing?"
    return {
        "case_id": f"{case.sample_id}__M_trap",
        "source_sample_id": case.sample_id,
        "variant_type": "M_trap",
        "question": ambiguous_q,
        "context": case.context,
        "code": case.code,
        "gold_answer": case.answer,
        "target_fic_id": fic.get("id"),
        "target_domain": fic.get("domain"),
        "target_topic": fic.get("topic"),
        "expected_status": "refusal",
        "expected_diagnostic_type": "M",
        "expected_requested_fields": [],
        "mutation_meta": {
            "mutation_rule": "semantic_ambiguity",
            "note": "Removed explicit metric signal (e.g., NPV/IRR/ROI).",
        },
    }


def _build_f_variant(case: CaseRecord, fic: Dict[str, Any]) -> Dict[str, Any]:
    missing_field = _pick_missing_field(fic)
    q, c = _f_trap_transform(case.question, case.context, missing_field)
    return {
        "case_id": f"{case.sample_id}__F_trap",
        "source_sample_id": case.sample_id,
        "variant_type": "F_trap",
        "question": q,
        "context": c,
        "code": case.code,
        "gold_answer": case.answer,
        "target_fic_id": fic.get("id"),
        "target_domain": fic.get("domain"),
        "target_topic": fic.get("topic"),
        "expected_status": "error",
        "expected_diagnostic_type": "F",
        "expected_requested_fields": [missing_field],
        "mutation_meta": {
            "mutation_rule": "drop_required_input_signal",
            "dropped_field": missing_field,
        },
    }


def _build_e_variant(case: CaseRecord, fic: Dict[str, Any]) -> Dict[str, Any]:
    q, c, extra = _e_trap_transform(case.question, case.context, fic)
    return {
        "case_id": f"{case.sample_id}__E_trap",
        "source_sample_id": case.sample_id,
        "variant_type": "E_trap",
        "question": q,
        "context": c,
        "code": case.code,
        "gold_answer": case.answer,
        "target_fic_id": fic.get("id"),
        "target_domain": fic.get("domain"),
        "target_topic": fic.get("topic"),
        "expected_status": "alert",
        "expected_diagnostic_type": "E",
        "expected_requested_fields": [],
        "expected_trigger_rule_id": extra["trigger_rule_id"],
        "mutation_meta": {
            "mutation_rule": "scale_or_binding_anomaly",
            **extra,
        },
    }


def expand_case(case: CaseRecord, fic: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        _build_clean_variant(case, fic),
        _build_m_variant(case, fic),
        _build_f_variant(case, fic),
        _build_e_variant(case, fic),
    ]


def _load_fic_cards(path: Path) -> List[Dict[str, Any]]:
    cards = _load_records(path)
    filtered = []
    for card in cards:
        if (
            str(card.get("domain", "")).strip().lower() == "investment_analysis"
            and str(card.get("topic", "")).strip().lower() in INVESTMENT_ANALYSIS_TOPICS
        ):
            filtered.append(card)
    if not filtered:
        raise ValueError("No investment_analysis FIC cards found in given fic file.")
    return filtered


def run_expansion(
    *,
    input_records: Iterable[Dict[str, Any]],
    fic_cards: List[Dict[str, Any]],
    max_records: int = 0,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    random.seed(seed)
    expanded: List[Dict[str, Any]] = []
    records = list(input_records)
    if max_records > 0:
        records = records[:max_records]
    for idx, raw in enumerate(records, start=1):
        case = _to_case(raw, idx)
        fic = _select_fic(case, fic_cards)
        expanded.extend(expand_case(case, fic))
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Expand raw financial QA records into clean/M/F/E variants for "
            "investment_analysis topics (ci/irr/npv/rar/roi)."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSON or JSONL")
    parser.add_argument(
        "--fic-cards",
        required=True,
        type=Path,
        help="FIC cards JSON/JSONL file (must contain investment_analysis cards).",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output JSON or JSONL")
    parser.add_argument("--max-records", type=int, default=0, help="0 means all records")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_records = _load_records(args.input)
    fic_cards = _load_fic_cards(args.fic_cards)
    expanded = run_expansion(
        input_records=raw_records,
        fic_cards=fic_cards,
        max_records=args.max_records,
        seed=args.seed,
    )
    _dump_records(args.output, expanded)
    print(f"Wrote {len(expanded)} expanded cases to {args.output}")


if __name__ == "__main__":
    main()

