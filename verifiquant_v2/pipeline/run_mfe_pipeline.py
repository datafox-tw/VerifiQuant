from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None
    genai_types = None

from verifiquant_v2.card_store import FICStore


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _parse_number(value: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace(",", "").replace("_", "")
    if not s:
        return None
    s_lower = s.lower()

    # Unit multipliers (handles patterns like "150 million", "1.5b", "200k")
    multiplier = 1.0
    if re.search(r"\btrillion\b|\bt\b", s_lower):
        multiplier = 1e12
    elif re.search(r"\bbillion\b|\bbn\b|\bb\b", s_lower):
        multiplier = 1e9
    elif re.search(r"\bmillion\b|\bmm\b|\bmn\b|\bm\b", s_lower):
        multiplier = 1e6
    elif re.search(r"\bthousand\b|\bk\b", s_lower):
        multiplier = 1e3
    elif "百萬" in s_lower:
        multiplier = 1e6
    elif "千萬" in s_lower:
        multiplier = 1e7
    elif "億" in s_lower:
        multiplier = 1e8
    elif "萬" in s_lower:
        multiplier = 1e4

    is_pct = "%" in s
    s = s.replace("%", "")
    m = re.search(r"-?\d+(\.\d+)?", s)
    if not m:
        return None
    n = float(m.group())
    if is_pct:
        n = n / 100.0
    else:
        n = n * multiplier
    return n


def _extract_numeric_list(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for v in value:
            n = _parse_number(v)
            if n is not None:
                out.append(n)
        return out

    s = str(value).strip()
    if not s:
        return []

    # Try JSON list first
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            out = []
            for v in parsed:
                n = _parse_number(v)
                if n is not None:
                    out.append(n)
            if out:
                return out
    except Exception:
        pass

    # Fallback: extract repeated number tokens, preserving list-like structure.
    tokens = re.findall(r"-?\d[\d,]*(?:\.\d+)?%?", s)
    out = []
    for tok in tokens:
        n = _parse_number(tok)
        if n is not None:
            out.append(n)
    return out


def _parse_typed_value(raw_value: Any, declared_type: str) -> Optional[Any]:
    dt = (declared_type or "").lower()
    if "list" in dt or "array" in dt:
        vals = _extract_numeric_list(raw_value)
        return vals if vals else None
    if "int" in dt:
        n = _parse_number(raw_value)
        if n is None:
            return None
        return int(round(n))
    # default numeric scalar
    return _parse_number(raw_value)


def _answer_match(question: str, output_value: Optional[float], gold_num: Optional[float]) -> Tuple[Optional[float], Optional[bool]]:
    if output_value is None or gold_num is None:
        return None, None
    q = question.lower()
    round_mode = any(x in q for x in ["nearest integer", "nearest whole", "round to nearest", "四捨五入", "整數"])
    if round_mode:
        lhs = float(round(output_value))
        rhs = float(round(gold_num))
        return abs(lhs - rhs), lhs == rhs
    abs_err = abs(output_value - gold_num)
    return abs_err, math.isclose(output_value, gold_num, rel_tol=1e-6, abs_tol=1e-2)


def _safe_eval_rule(rule: str, env: Dict[str, Any]) -> Optional[bool]:
    if not rule:
        return None
    local_env = dict(env)
    local_env.update({"abs": abs, "min": min, "max": max, "len": len, "sum": sum, "math": math})
    try:
        return bool(eval(rule, {"__builtins__": {}}, local_env))
    except Exception:
        return None


@dataclass
class RetrievalCandidate:
    card: Dict[str, Any]
    score: float


def _card_search_text(card: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k in ("id", "name", "short_description", "domain", "topic"):
        v = card.get(k)
        if isinstance(v, str):
            parts.append(v)
    for inp in card.get("inputs", []):
        parts.append(str(inp.get("name", "")))
        parts.append(str(inp.get("description", "")))
    hints = card.get("selection_hints", {}) or {}
    for k in ("self_description",):
        v = hints.get(k)
        if isinstance(v, str):
            parts.append(v)
    for k in ("applicable_when", "common_confusions", "required_input_summary"):
        v = hints.get(k, [])
        if isinstance(v, list):
            for item in v:
                parts.append(json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item))
    return " ".join(parts)


def retrieve_candidates(cards: List[Dict[str, Any]], query: str, top_k: int) -> List[RetrievalCandidate]:
    q_tokens = set(_tokenize(query))
    scored: List[RetrievalCandidate] = []
    for card in cards:
        text = _card_search_text(card)
        c_tokens = set(_tokenize(text))
        overlap = len(q_tokens & c_tokens)
        denom = max(len(q_tokens), 1)
        score = overlap / denom
        scored.append(RetrievalCandidate(card=card, score=score))
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]


def retrieve_candidates_from_store(store: FICStore, query: str, top_k: int, alpha: float = 0.4) -> List[RetrievalCandidate]:
    results = store.retrieve_top_k(query=query, top_k=top_k, alpha=alpha)
    return [RetrievalCandidate(card=record.data, score=score) for record, score in results]


def _load_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported input format: {path}")


def _dump_records(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_cards(path: Path) -> List[Dict[str, Any]]:
    cards = _load_records(path)
    out = []
    for c in cards:
        if isinstance(c, dict) and "id" in c and "inputs" in c and "execution" in c:
            out.append(c)
    if not out:
        raise ValueError("No valid FIC cards found.")
    return out


def _schema_selection() -> Any:
    return genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "chosen_id": genai_types.Schema(type=genai_types.Type.STRING),
            "reason": genai_types.Schema(type=genai_types.Type.STRING),
        },
        required=["chosen_id", "reason"],
    )


def _schema_extraction() -> Any:
    return genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "inputs": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        "name": genai_types.Schema(type=genai_types.Type.STRING),
                        "status": genai_types.Schema(type=genai_types.Type.STRING, enum=["provided", "missing"]),
                        "value": genai_types.Schema(type=genai_types.Type.STRING),
                    },
                    required=["name", "status", "value"],
                ),
            ),
            "normalization_note": genai_types.Schema(type=genai_types.Type.STRING),
        },
        required=["inputs", "normalization_note"],
    )


def _schema_echeck() -> Any:
    return genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "is_e_alert": genai_types.Schema(type=genai_types.Type.BOOLEAN),
            "reason": genai_types.Schema(type=genai_types.Type.STRING),
            "triggered_rule_ids": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
        },
        required=["is_e_alert", "reason", "triggered_rule_ids"],
    )


def _llm_json(client: Any, *, model: str, prompt: str, schema: Any) -> Dict[str, Any]:
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
    )
    response = client.models.generate_content(model=model, contents=prompt, config=config)
    return json.loads(response.text)


def _select_card_with_llm(client: Any, model: str, question: str, context: str, cands: List[RetrievalCandidate]) -> Dict[str, Any]:
    block = []
    for i, c in enumerate(cands, 1):
        block.append(
            f"Candidate {i} | id={c.card.get('id')} | score={c.score:.3f}\n"
            f"name={c.card.get('name')}\n"
            f"topic={c.card.get('topic')}\n"
            f"summary={c.card.get('short_description','')}\n"
            f"required_inputs={[inp.get('name') for inp in c.card.get('inputs', [])]}"
        )
    prompt = f"""
You are selecting a financial formula card.
If the query is ambiguous or none match, choose empty chosen_id "" (M-type refusal).

Question:
{question}

Context:
{context}

Candidates:
{chr(10).join(block)}

Return JSON only.
"""
    return _llm_json(client, model=model, prompt=prompt, schema=_schema_selection())


def _extract_inputs_with_llm(client: Any, model: str, question: str, context: str, card: Dict[str, Any]) -> Dict[str, Any]:
    required = card.get("inputs", [])
    prompt = f"""
Extract required input values from question/context.
If missing, set status="missing" and value="".
Keep list-like values as JSON array strings when input type is list/array.
Percent values should preserve unit signal (e.g. "8%" or "0.08"), do not concatenate numbers.
If text includes magnitude words (million/billion/thousand), keep that unit in the value text
or convert explicitly to absolute number (e.g., "150 million" -> "150000000") unless input description says values are in millions.
For share counts and currency amounts, prefer absolute numeric values in `value` (no words), e.g.:
- "1.5 million shares" -> "1500000"
- "$2.3 billion" -> "2300000000"
- "150 million" -> "150000000"
For rates/probabilities, keep clear rate format (e.g., "8%" or "0.08").
Also return `normalization_note` describing any unit normalization you applied.

Required inputs:
{json.dumps(required, ensure_ascii=False, indent=2)}

Question:
{question}

Context:
{context}

Return JSON only.
"""
    return _llm_json(client, model=model, prompt=prompt, schema=_schema_extraction())


def _echeck_with_llm(
    client: Any,
    model: str,
    question: str,
    context: str,
    card: Dict[str, Any],
    provided_inputs: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = f"""
Determine whether this case should trigger E-type alert (input binding/scale/logical inconsistency).
Use the card diagnostics and provided inputs.
Return is_e_alert=true only when likely inconsistent.

Question:
{question}

Context:
{context}

Card diagnostics:
{json.dumps(card.get('diagnostics', {}), ensure_ascii=False, indent=2)}

Provided inputs:
{json.dumps(provided_inputs, ensure_ascii=False, indent=2)}

Return JSON only.
"""
    return _llm_json(client, model=model, prompt=prompt, schema=_schema_echeck())


def _evaluate_execution(card: Dict[str, Any], provided_inputs: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    execution = card.get("execution", {})
    code = str(execution.get("code", ""))
    output_var = str(card.get("output_var", ""))
    if "def compute(inputs)" not in code:
        return None, "execution code missing compute(inputs)"

    safe_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "float": float,
        "int": int,
        "pow": pow,
        "round": round,
        "range": range,
        "list": list,
        "dict": dict,
        "enumerate": enumerate,
    }
    local_env: Dict[str, Any] = {}
    try:
        exec(code, {"__builtins__": safe_builtins}, local_env)
        fn = local_env.get("compute")
        if not callable(fn):
            return None, "compute function missing after exec"
        result = fn(provided_inputs)
        if not isinstance(result, dict):
            return None, "compute did not return dict"
        if output_var not in result:
            return None, f"output_var '{output_var}' missing from compute result"
        val = _parse_number(result.get(output_var))
        if val is None:
            return None, "output value not numeric"
        return val, None
    except Exception as exc:
        return None, str(exc)


def _auto_rule_check(card: Dict[str, Any], provided_inputs: Dict[str, Any]) -> List[str]:
    triggered: List[str] = []
    for inv in (card.get("diagnostics", {}) or {}).get("invariants", []):
        rule = str(inv.get("rule", ""))
        rid = str(inv.get("id", ""))
        ok = _safe_eval_rule(rule, provided_inputs)
        if ok is False:
            triggered.append(rid or "invariant")
    return triggered


def run_case(
    *,
    row: Dict[str, Any],
    cards: List[Dict[str, Any]],
    store: Optional[FICStore],
    client: Any,
    selector_model: str,
    extractor_model: str,
    judge_model: str,
    top_k: int,
) -> Dict[str, Any]:
    q = str(row.get("question", "") or "")
    c = str(row.get("context", "") or "")
    case_id = str(row.get("case_id") or row.get("question_id") or "")
    if store is not None:
        candidates = retrieve_candidates_from_store(store, f"{q}\n{c}", top_k=top_k)
    else:
        candidates = retrieve_candidates(cards, f"{q}\n{c}", top_k=top_k)
    if not candidates:
        return {
            "case_id": case_id,
            "status": "refusal",
            "diagnostic_type": "M",
            "reason": "No candidate cards retrieved",
            "candidate_ids": [],
        }

    selection = _select_card_with_llm(client, selector_model, q, c, candidates)
    chosen_id = str(selection.get("chosen_id", "") or "").strip()
    if not chosen_id:
        return {
            "case_id": case_id,
            "status": "refusal",
            "diagnostic_type": "M",
            "reason": selection.get("reason", "Ambiguous task or no suitable card"),
            "candidate_ids": [cand.card.get("id") for cand in candidates],
        }

    card = next((cand.card for cand in candidates if str(cand.card.get("id")) == chosen_id), None)
    if card is None:
        return {
            "case_id": case_id,
            "status": "refusal",
            "diagnostic_type": "M",
            "reason": f"Selected card '{chosen_id}' not in retrieved top-k",
            "candidate_ids": [cand.card.get("id") for cand in candidates],
        }

    extraction = _extract_inputs_with_llm(client, extractor_model, q, c, card)
    normalization_note = str(extraction.get("normalization_note", "") or "")
    card_inputs = card.get("inputs", [])
    type_map = {str(inp.get("name")): str(inp.get("type", "float")) for inp in card_inputs}
    required_inputs = [inp for inp in card_inputs if bool(inp.get("required", True))]
    required_names = [str(inp.get("name")) for inp in required_inputs]
    extracted = {str(x.get("name")): x for x in extraction.get("inputs", []) if x.get("name")}

    missing = []
    provided_inputs: Dict[str, Any] = {}
    for name in required_names:
        item = extracted.get(name)
        if not item or item.get("status") != "provided":
            missing.append(name)
            continue
        parsed = _parse_typed_value(item.get("value", ""), type_map.get(name, "float"))
        if parsed is None:
            missing.append(name)
        else:
            provided_inputs[name] = parsed

    # keep optional values if parseable
    for name, item in extracted.items():
        if name in provided_inputs:
            continue
        parsed = _parse_typed_value(item.get("value", ""), type_map.get(name, "float"))
        if parsed is not None:
            provided_inputs[name] = parsed

    if missing:
        return {
            "case_id": case_id,
            "status": "error",
            "diagnostic_type": "F",
            "reason": card.get("refusal_hints", {}).get("f_error_message", "Missing required input"),
            "card_id": card.get("id"),
            "candidate_ids": [cand.card.get("id") for cand in candidates],
            "requested_fields": missing,
            "provided_inputs": provided_inputs,
            "normalization_note": normalization_note,
        }

    # E-check: first deterministic rule eval, then LLM judge as supplement.
    auto_triggered = _auto_rule_check(card, provided_inputs)
    e_judge = _echeck_with_llm(client, judge_model, q, c, card, provided_inputs)
    e_flag = bool(e_judge.get("is_e_alert")) or bool(auto_triggered)
    triggered = list(dict.fromkeys(auto_triggered + list(e_judge.get("triggered_rule_ids", []))))
    if e_flag:
        return {
            "case_id": case_id,
            "status": "alert",
            "diagnostic_type": "E",
            "reason": e_judge.get("reason")
            or card.get("refusal_hints", {}).get("e_alert_message", "Potential inconsistency detected"),
            "card_id": card.get("id"),
            "candidate_ids": [cand.card.get("id") for cand in candidates],
            "provided_inputs": provided_inputs,
            "triggered_rule_ids": triggered,
            "normalization_note": normalization_note,
        }

    output_value, exec_err = _evaluate_execution(card, provided_inputs)
    if exec_err:
        return {
            "case_id": case_id,
            "status": "error",
            "diagnostic_type": "C",
            "reason": exec_err,
            "card_id": card.get("id"),
            "candidate_ids": [cand.card.get("id") for cand in candidates],
            "provided_inputs": provided_inputs,
            "normalization_note": normalization_note,
        }

    gold = row.get("gold_answer", row.get("answer", row.get("ground_truth")))
    gold_num = _parse_number(gold)
    abs_error, is_correct = _answer_match(q, output_value, gold_num)

    return {
        "case_id": case_id,
        "status": "success",
        "diagnostic_type": None,
        "reason": "Execution completed",
        "card_id": card.get("id"),
        "candidate_ids": [cand.card.get("id") for cand in candidates],
        "provided_inputs": provided_inputs,
        "output_var": card.get("output_var"),
        "output_value": output_value,
        "gold_answer": gold_num if gold_num is not None else gold,
        "abs_error": abs_error,
        "is_correct": is_correct,
        "normalization_note": normalization_note,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run v2 M/F/E pipeline on question JSONL using FIC cards."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSON/JSONL questions.")
    parser.add_argument("--fic-cards", required=True, type=Path, help="FIC v2 JSON/JSONL.")
    parser.add_argument("--store", type=Path, help="Optional FICStore .pkl for hybrid retrieval.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON/JSONL results.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--selector-model", default="gemini-2.5-flash")
    parser.add_argument("--extractor-model", default="gemini-2.5-flash")
    parser.add_argument("--judge-model", default="gemini-2.5-flash")
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
    cards = _load_cards(args.fic_cards)
    store: Optional[FICStore] = None
    if args.store:
        if not args.store.exists():
            raise FileNotFoundError(args.store)
        store = FICStore.load(args.store)
    client = genai.Client(api_key=api_key)

    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        cid = row.get("case_id", row.get("question_id", idx))
        print(f"[run_mfe_pipeline] processing {cid}")
        result = run_case(
            row=row,
            cards=cards,
            store=store,
            client=client,
            selector_model=args.selector_model,
            extractor_model=args.extractor_model,
            judge_model=args.judge_model,
            top_k=args.top_k,
        )
        # keep expected fields for easy eval if present in input
        for k in ("variant_type", "expected_status", "expected_diagnostic_type", "source_sample_id", "reason"):
            if k in row and k not in result:
                result[f"expected_{k}" if k == "reason" else k] = row[k]
        out.append(result)

    _dump_records(args.output, out)
    print(f"Wrote {len(out)} results to {args.output}")


if __name__ == "__main__":
    main()
