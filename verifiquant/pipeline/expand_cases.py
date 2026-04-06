from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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
            "n_variant": genai_types.Schema(
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
            "i_variant": genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "question": genai_types.Schema(type=genai_types.Type.STRING),
                    "context": genai_types.Schema(type=genai_types.Type.STRING),
                    "reason": genai_types.Schema(type=genai_types.Type.STRING),
                },
                required=["question", "context", "reason"],
            ),
        },
        required=["m_variant", "n_variant", "f_variant", "e_variant", "i_variant"],
    )
    SINGLE_VARIANT_SCHEMA = genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "question": genai_types.Schema(type=genai_types.Type.STRING),
            "context": genai_types.Schema(type=genai_types.Type.STRING),
            "reason": genai_types.Schema(type=genai_types.Type.STRING),
        },
        required=["question", "context", "reason"],
    )
else:
    VARIANT_SCHEMA = None
    SINGLE_VARIANT_SCHEMA = None


PROMPT_TEMPLATE = """You are generating controlled benchmark variants for financial reasoning error analysis.

You are given ONE clean case with:
- question
- context
- code
- answer

Generate exactly THREE modified variants:
1) m_variant (M type: misunderstanding/semantic ambiguity)
2) n_variant (N type: intent is clear but formula family is out-of-scope / not supported)
3) f_variant (F type: formula/spec mismatch, mainly due to missing required inputs/spec)
4) e_variant (E type: extraction/binding/scale inconsistency, numbers exist but are implausible/inconsistent)
5) i_variant (I type: hidden semantic ambiguity such as FX direction/time basis)

Important M/N/F/E/I boundary:
- M: make intent ambiguous (e.g., user asks if project is profitable but does not specify NPV vs IRR vs Payback), OR make wording likely to retrieve wrong conceptual card.
- N: keep intent explicit, but require a formula family likely outside current supported library.
- F: keep intent mostly clear, but remove key required information so the formula cannot be completed (e.g., missing discount rate, period count, required variable).
- E: keep intent and required fields present, but alter numeric expression/context so values are likely mis-bound or scale-wrong (e.g., 8 vs 0.08, swapped fields, inconsistent units/frequency).
- I: keep numbers present, but introduce hidden interpretation ambiguity (e.g., FX quote direction or begin/end-of-period timing).

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
  - n_variant: question, context, reason
  - f_variant: question, context, reason
  - e_variant: question, context, reason
  - i_variant: question, context, reason

Base case:
<BASE_CASE_JSON>
{base_case_json}
</BASE_CASE_JSON>
"""

SEMI_LLM_REFINE_PROMPT_TEMPLATE = """You are refining synthetic benchmark variants for financial error diagnostics.

You are given:
1) A clean base case.
2) A seeded synthetic M/N/F/E/I variant set.

Task:
- Keep the seeded defect type intention unchanged.
- Rewrite question/context to be realistic and naturally phrased.
- Stay close to the base topic.
- Do NOT compute the final answer.
- Keep each variant clearly belonging to M/N/F/E/I category.

Return JSON only with the same schema: m_variant/n_variant/f_variant/e_variant/i_variant, each containing question/context/reason.

<BASE_CASE_JSON>
{base_case_json}
</BASE_CASE_JSON>

<SEEDED_VARIANTS_JSON>
{seeded_variants_json}
</SEEDED_VARIANTS_JSON>
"""

SEMI_LLM_REGEN_ONE_TEMPLATE = """You are regenerating ONE benchmark variant for financial diagnostic testing.

Required target type: {target_type}

Rules:
- Keep topic close to base case.
- Keep defect aligned to target type only.
- Ensure variant is materially different from base question/context.
- Do NOT compute the final answer.

Target type definitions:
- M: ambiguous intent (e.g., worth doing? without explicit NPV/IRR/payback metric).
- N: explicit intent but requires unsupported formula family / out-of-scope reasoning.
- F: required spec/input missing.
- E: intent/spec present but numeric scale/binding/frequency inconsistency.
- I: hidden semantic ambiguity (FX direction, time basis, period convention).

Return JSON only with:
- question
- context
- reason

<BASE_CASE_JSON>
{base_case_json}
</BASE_CASE_JSON>

<CURRENT_VARIANT_JSON>
{current_variant_json}
</CURRENT_VARIANT_JSON>
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


def _llm_json(client: Any, *, model: str, prompt: str) -> Dict[str, Any]:
    if genai is None or genai_types is None or VARIANT_SCHEMA is None:
        raise RuntimeError("google.genai is not available. Please install compatible google-genai package.")
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


def _llm_json_single_variant(client: Any, *, model: str, prompt: str) -> Dict[str, Any]:
    if genai is None or genai_types is None or SINGLE_VARIANT_SCHEMA is None:
        raise RuntimeError("google.genai is not available. Please install compatible google-genai package.")
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=SINGLE_VARIANT_SCHEMA,
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


def _call_llm_variants(*, client: Any, model: str, case: CaseRecord) -> Dict[str, Any]:
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
    return _llm_json(client, model=model, prompt=prompt)


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9FFF]", text or ""))


def _remove_rate_mentions(text: str) -> str:
    out = str(text or "")
    out = re.sub(r"\b\d+(?:\.\d+)?\s*%\b", "[MISSING_RATE]", out, flags=re.IGNORECASE)
    out = re.sub(r"\b(?:discount|required|hurdle|cost of capital)\s+rate\b[^,.。\n]*", "discount rate [MISSING_RATE]", out, flags=re.IGNORECASE)
    return out


def _remove_period_mentions(text: str) -> str:
    out = str(text or "")
    out = re.sub(r"\b\d+\s*(?:year|years|yr|yrs|period|periods)\b", "[MISSING_PERIOD]", out, flags=re.IGNORECASE)
    return out


def _remove_amount_mentions(text: str) -> str:
    out = str(text or "")
    out = re.sub(r"\$\s*\d[\d,]*(?:\.\d+)?", "$[MISSING_AMOUNT]", out)
    out = re.sub(r"\b\d[\d,]*(?:\.\d+)?\s*(?:usd|dollars?)\b", "[MISSING_AMOUNT]", out, flags=re.IGNORECASE)
    return out


def _pick_f_missing_field(case: CaseRecord) -> str:
    hay = f"{case.question}\n{case.context}\n{case.code}".lower()
    if any(w in hay for w in ["discount rate", "cost of capital", "required rate", "hurdle rate", "rate_of_return"]):
        return "discount_rate"
    if any(w in hay for w in ["year", "period", "n="]):
        return "period_count"
    if any(w in hay for w in ["cash flow", "cash_flows"]):
        return "cash_flows"
    if any(w in hay for w in ["initial investment", "initial_investment", "initial outlay"]):
        return "initial_investment"
    return "required_input"


def _f_drop(question: str, context: str, field: str) -> tuple[str, str]:
    if field == "discount_rate":
        return _remove_rate_mentions(question), _remove_rate_mentions(context)
    if field == "period_count":
        return _remove_period_mentions(question), _remove_period_mentions(context)
    if field in {"cash_flows", "initial_investment"}:
        return _remove_amount_mentions(question), _remove_amount_mentions(context)
    return question + f" [MISSING_FIELD:{field}]", context


def _flip_percent_scale_once(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return match.group(1)

    return re.sub(r"\b(\d+(?:\.\d+)?)\s*%", repl, str(text or ""), count=1)


def _build_seeded_variants(case: CaseRecord) -> Dict[str, Dict[str, str]]:
    is_cjk = _contains_cjk(case.question) or _contains_cjk(case.context)

    if is_cjk:
        m_question = "這個專案整體來看值不值得做？請幫我評估。"
    else:
        m_question = "Can you evaluate whether this project is worth doing overall?"

    f_field = _pick_f_missing_field(case)
    f_q, f_c = _f_drop(case.question, case.context, f_field)

    if is_cjk:
        n_question = "請用 Heston 隨機波動率模型計算這個專案的風險中性價格並比較 NPV。"
    else:
        n_question = "Please price this project with a Heston stochastic-volatility framework and compare with NPV."

    e_q = _flip_percent_scale_once(case.question)
    e_c = _flip_percent_scale_once(case.context)
    if e_q == case.question and e_c == case.context:
        if is_cjk:
            e_c = (e_c + " 現金流以月為單位，但折現率沿用年化數值且未做轉換。").strip()
        else:
            e_c = (e_c + " Cash flows are monthly, but the discount rate is annual and left unconverted.").strip()

    if is_cjk:
        i_context = (case.context + " 題目未說明匯率是 USD/TWD 還是 TWD/USD，且未說明現金流是期初還是期末。").strip()
    else:
        i_context = (
            case.context
            + " The prompt does not specify FX quote direction (USD/TWD vs TWD/USD) nor begin/end-of-period cash-flow timing."
        ).strip()

    return {
        "m_variant": {
            "question": m_question,
            "context": case.context,
            "reason": "Made target metric intent ambiguous while keeping same finance scenario.",
        },
        "n_variant": {
            "question": n_question,
            "context": case.context,
            "reason": "Kept clear intent but requested a likely out-of-scope formula family to test N-class graceful exit.",
        },
        "f_variant": {
            "question": f_q,
            "context": f_c,
            "reason": f"Removed required specification signal ({f_field}) to force missing-input diagnostic.",
        },
        "e_variant": {
            "question": e_q,
            "context": e_c,
            "reason": "Kept task intent but injected plausible scale/frequency inconsistency for binding checks.",
        },
        "i_variant": {
            "question": case.question,
            "context": i_context,
            "reason": "Injected hidden semantic ambiguity (FX/time-basis) that should trigger I-gate clarification.",
        },
    }


def _refine_seeded_with_llm(*, client: Any, model: str, case: CaseRecord, seeded: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    prompt = SEMI_LLM_REFINE_PROMPT_TEMPLATE.format(
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
        seeded_variants_json=json.dumps(seeded, ensure_ascii=False, indent=2),
    )
    return _llm_json(client, model=model, prompt=prompt)


def _regen_single_variant_with_llm(
    *,
    client: Any,
    model: str,
    case: CaseRecord,
    vtype: str,
    current_variant: Dict[str, str],
) -> Dict[str, Any]:
    prompt = SEMI_LLM_REGEN_ONE_TEMPLATE.format(
        target_type=vtype,
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
        current_variant_json=json.dumps(current_variant, ensure_ascii=False, indent=2),
    )
    return _llm_json_single_variant(client, model=model, prompt=prompt)


def _normalize_cmp(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _is_materially_changed(case: CaseRecord, payload: Dict[str, str]) -> bool:
    q_same = _normalize_cmp(payload.get("question", "")) == _normalize_cmp(case.question)
    c_same = _normalize_cmp(payload.get("context", "")) == _normalize_cmp(case.context)
    return not (q_same and c_same)


def _has_expected_signal(vtype: str, case: CaseRecord, payload: Dict[str, str]) -> bool:
    q = str(payload.get("question", "") or "")
    c = str(payload.get("context", "") or "")
    comb = f"{q}\n{c}".lower()
    base = f"{case.question}\n{case.context}".lower()

    if vtype == "M":
        has_metric = any(tok in comb for tok in ["npv", "irr", "payback", "wacc", "roi"])
        return not has_metric
    if vtype == "N":
        return any(
            tok in comb
            for tok in ["heston", "stochastic", "copula", "kalman", "monte carlo", "garch"]
        )
    if vtype == "F":
        if "[missing_" in comb:
            return True
        return comb.count("%") < base.count("%")
    if vtype == "E":
        if "monthly" in comb and "annual" in comb:
            return True
        if "月" in comb and "年" in comb:
            return True
        return comb.count("%") < base.count("%")
    if vtype == "I":
        return any(
            tok in comb
            for tok in [
                "usd/twd",
                "twd/usd",
                "fx",
                "匯率",
                "期初",
                "期末",
                "begin",
                "end-of-period",
            ]
        )
    return True


def _needs_single_variant_regen(vtype: str, case: CaseRecord, payload: Dict[str, str]) -> bool:
    if not _is_materially_changed(case, payload):
        return True
    if not _has_expected_signal(vtype, case, payload):
        return True
    return False


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
        "update_method": "original_clean",
    }


def _mk_variant(
    case: CaseRecord,
    vtype: str,
    payload: Dict[str, Any],
    expected_status: str,
    expected_diag: str,
    update_method: str,
) -> Dict[str, Any]:
    return {
        "case_id": f"{case.sample_id}__{vtype}",
        "source_sample_id": case.sample_id,
        "variant_type": vtype,
        "question": str(payload.get("question", "") or ""),
        "context": str(payload.get("context", "") or ""),
        "code": case.code,
        "gold_answer": case.answer,
        "expected_status": expected_status,
        "expected_diagnostic_type": expected_diag,
        "reason": str(payload.get("reason", "") or ""),
        "update_method": update_method,
    }


def expand_case(*, case: CaseRecord, client: Optional[Any], model: str, mode: str) -> List[Dict[str, Any]]:
    methods = {
        "m_variant": "unknown",
        "n_variant": "unknown",
        "f_variant": "unknown",
        "e_variant": "unknown",
        "i_variant": "unknown",
    }

    if mode == "llm":
        if client is None:
            raise RuntimeError("mode=llm requires GEMINI_API_KEY and google-genai dependency.")
        payload = _call_llm_variants(client=client, model=model, case=case)
        methods = {
            "m_variant": "llm_full_generation",
            "n_variant": "llm_full_generation",
            "f_variant": "llm_full_generation",
            "e_variant": "llm_full_generation",
            "i_variant": "llm_full_generation",
        }
    elif mode == "semi-llm":
        seeded = _build_seeded_variants(case)
        payload = seeded
        methods = {
            "m_variant": "seeded_regex",
            "n_variant": "seeded_regex",
            "f_variant": "seeded_regex",
            "e_variant": "seeded_regex",
            "i_variant": "seeded_regex",
        }
        if client is not None:
            try:
                payload = _refine_seeded_with_llm(client=client, model=model, case=case, seeded=seeded)
                methods = {
                    "m_variant": "seeded_regex+llm_refine",
                    "n_variant": "seeded_regex+llm_refine",
                    "f_variant": "seeded_regex+llm_refine",
                    "e_variant": "seeded_regex+llm_refine",
                    "i_variant": "seeded_regex+llm_refine",
                }
            except Exception as exc:
                print(f"[expand_cases] semi-llm LLM refine failed for {case.sample_id}: {exc}; fallback to seeded.")
                payload = seeded

        key_to_type = {
            "m_variant": "M",
            "n_variant": "N",
            "f_variant": "F",
            "e_variant": "E",
            "i_variant": "I",
        }
        for key, vtype in key_to_type.items():
            cur = payload.get(key, {})
            if not isinstance(cur, dict):
                cur = {}
            if _needs_single_variant_regen(vtype, case, cur):
                if client is None:
                    methods[key] = methods[key] + "+regen_unavailable"
                    continue
                try:
                    regen = _regen_single_variant_with_llm(
                        client=client,
                        model=model,
                        case=case,
                        vtype=vtype,
                        current_variant=cur,
                    )
                    payload[key] = regen
                    methods[key] = "llm_regen_fallback"
                except Exception as exc:
                    print(
                        f"[expand_cases] fallback regen failed for {case.sample_id}/{vtype}: {exc}; keep current."
                    )
                    methods[key] = methods[key] + "+regen_failed"
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return [
        _mk_clean(case),
        _mk_variant(
            case,
            "M_trap",
            payload["m_variant"],
            "refusal",
            "M",
            methods["m_variant"],
        ),
        _mk_variant(
            case,
            "N_trap",
            payload["n_variant"],
            "refusal",
            "N",
            methods["n_variant"],
        ),
        _mk_variant(
            case,
            "F_trap",
            payload["f_variant"],
            "error",
            "F",
            methods["f_variant"],
        ),
        _mk_variant(
            case,
            "E_trap",
            payload["e_variant"],
            "alert",
            "E",
            methods["e_variant"],
        ),
        _mk_variant(
            case,
            "I_trap",
            payload["i_variant"],
            "needs_clarification",
            "I",
            methods["i_variant"],
        ),
    ]


def run_expansion(
    *,
    input_records: Iterable[Dict[str, Any]],
    client: Optional[Any],
    model: str,
    max_records: int,
    mode: str,
) -> List[Dict[str, Any]]:
    records = list(input_records)
    if max_records > 0:
        records = records[:max_records]
    out: List[Dict[str, Any]] = []
    for idx, raw in enumerate(records, start=1):
        case = _to_case(raw, idx)
        print(f"[expand_cases] processing {case.sample_id} (mode={mode})")
        out.extend(expand_case(case=case, client=client, model=model, mode=mode))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Expand clean JSON/JSONL cases to clean+M/N/F/E/I variants. "
            "mode=llm uses pure LLM generation; mode=semi-llm uses controlled traps with optional LLM refinement."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSON or JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON or JSONL")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--mode", choices=["llm", "semi-llm"], default="llm")
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()

    client: Optional[Any] = None
    api_key = os.environ.get("GEMINI_API_KEY")
    if args.mode == "llm":
        if genai is None:
            print("Missing dependency: google.genai import failed. Please install compatible google-genai package.", file=sys.stderr)
            sys.exit(1)
        if not api_key:
            print("Missing GEMINI_API_KEY in environment.", file=sys.stderr)
            sys.exit(1)
        client = genai.Client(api_key=api_key)
    else:
        if genai is not None and api_key:
            client = genai.Client(api_key=api_key)
        else:
            print("[expand_cases] mode=semi-llm running without LLM refinement (seeded deterministic only).")

    raw_records = _load_records(args.input)
    expanded = run_expansion(
        input_records=raw_records,
        client=client,
        model=args.model,
        max_records=args.max_records,
        mode=args.mode,
    )
    _dump_records(args.output, expanded)
    print(f"Wrote {len(expanded)} expanded cases to {args.output}")


if __name__ == "__main__":
    main()
