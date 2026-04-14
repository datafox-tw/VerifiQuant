from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None
    genai_types = None

from verifiquant.preprocessing.validate_relations import validate_artifact_relations
from verifiquant.preprocessing.stage_repair import GLOBAL_N_NOT_SUPPORTED_RULE, GLOBAL_SCOPE_FIC_ID
from verifiquant.card_store import SQLAlchemyArtifactStore


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", str(text or "").lower())


def _parse_number(value: Any, *, percent_as_decimal: bool = True) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace(",", "").replace("_", "")
    if not s:
        return None

    multiplier = 1.0
    s_lower = s.lower()
    if re.search(r"\btrillion\b|\bt\b", s_lower):
        multiplier = 1e12
    elif re.search(r"\bbillion\b|\bbn\b|\bb\b", s_lower):
        multiplier = 1e9
    elif re.search(r"\bmillion\b|\bmm\b|\bmn\b|\bm\b", s_lower):
        multiplier = 1e6
    elif re.search(r"\bthousand\b|\bk\b", s_lower):
        multiplier = 1e3

    is_pct = "%" in s
    s = s.replace("%", "")
    m = re.search(r"-?\d+(\.\d+)?", s)
    if not m:
        return None
    n = float(m.group())
    if is_pct:
        return n / 100.0 if percent_as_decimal else n
    return n * multiplier


def _extract_numeric_list(value: Any, *, percent_as_decimal: bool = True) -> List[float]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for v in value:
            n = _parse_number(v, percent_as_decimal=percent_as_decimal)
            if n is not None:
                out.append(n)
        return out

    s = str(value).strip()
    if not s:
        return []

    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            out = []
            for v in parsed:
                n = _parse_number(v, percent_as_decimal=percent_as_decimal)
                if n is not None:
                    out.append(n)
            if out:
                return out
    except Exception:
        pass

    tokens = re.findall(r"-?\d[\d,]*(?:\.\d+)?%?", s)
    out = []
    for tok in tokens:
        n = _parse_number(tok, percent_as_decimal=percent_as_decimal)
        if n is not None:
            out.append(n)
    return out


def _infer_percent_as_decimal(unit: Any, description: Any) -> bool:
    u = str(unit or "").strip().lower()
    d = str(description or "").strip().lower()
    if "decimal" in u:
        return True
    if "as a decimal" in d or "expressed as a decimal" in d:
        return True
    if u in {"%", "percent", "percentage", "percentage_point", "percentage_points"}:
        return False
    if "percentage" in u and "decimal" not in u:
        return False
    return True


def _is_percent_point_unit(unit: Any, description: Any) -> bool:
    u = str(unit or "").strip().lower()
    d = str(description or "").strip().lower()
    if "decimal" in u or "as a decimal" in d or "expressed as a decimal" in d:
        return False
    return u in {"%", "percent", "percentage", "percentage_point", "percentage_points"} or (
        "percentage" in u and "decimal" not in u
    )


def _parse_typed_value(raw_value: Any, declared_type: str, *, unit: Any = None, description: Any = None) -> Optional[Any]:
    dt = str(declared_type or "").lower()
    percent_as_decimal = _infer_percent_as_decimal(unit, description)
    if "array" in dt or "list" in dt:
        vals = _extract_numeric_list(raw_value, percent_as_decimal=percent_as_decimal)
        return vals if vals else None
    if "int" in dt:
        n = _parse_number(raw_value, percent_as_decimal=percent_as_decimal)
        if n is None:
            return None
        return int(round(n))
    if "bool" in dt:
        s = str(raw_value or "").strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no"}:
            return False
        return None
    return _parse_number(raw_value, percent_as_decimal=percent_as_decimal)


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


class _AttrDict(dict):
    """Dict wrapper that also supports attribute access (inputs.foo)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _safe_eval_rule(
    rule: str,
    inputs: Dict[str, Any],
    *,
    compute_fn: Optional[Any] = None,
) -> Tuple[Optional[bool], Optional[str]]:
    """Evaluate a diagnostic rule expression against provided inputs.

    Parameters
    ----------
    rule:
        The expression string to evaluate (e.g. ``inputs['x'] < 0``).
    inputs:
        The bound input values for the current FIC.
    compute_fn:
        Optional: the compiled ``compute`` function from the FIC's execution
        block.  Some FIC E-check expressions call ``compute(...)`` to inspect
        the output range, so we expose it in the eval environment when provided.
    """
    if not rule:
        return None, "empty rule expression"
    inputs_obj = _AttrDict(inputs)
    env: Dict[str, Any] = {
        # NOTE: __builtins__ must live inside env (passed as globals to eval).
        # Generator / comprehension expressions look up free variables in the
        # *globals* dict, NOT in the calling frame's locals.  Placing all
        # helpers here ensures they are visible inside generator bodies too.
        "__builtins__": {},
        "inputs": inputs_obj,
        "abs": abs,
        "min": min,
        "max": max,
        "len": len,
        "sum": sum,
        "all": all,
        "any": any,
        "isinstance": isinstance,
        "float": float,
        "int": int,
        "bool": bool,
        "str": str,
        "list": list,
        "tuple": tuple,
        "dict": dict,
        "set": set,
        "range": range,
        "zip": zip,
        "enumerate": enumerate,
        "math": math,
    }
    # Flatten input values into the env so expressions can reference them
    # directly (e.g. ``discount_rate < 0`` instead of ``inputs['discount_rate'] < 0``).
    env.update(inputs_obj)
    # Expose the FIC's compiled compute function when available so that E-check
    # expressions like ``compute({...}) > 10`` work correctly.
    if compute_fn is not None and callable(compute_fn):
        env["compute"] = compute_fn
    try:
        return bool(eval(rule, env)), None
    except Exception as exc:
        return None, str(exc)


def _infer_predicate_mode(chk: Dict[str, Any]) -> str:
    """
    Infer whether expression means:
    - violation: True => trigger
    - validity:  True => pass
    """
    explicit = str(chk.get("predicate_mode", chk.get("expression_mode", ""))).strip().lower()
    if explicit in {"violation", "validity"}:
        return explicit

    expr = str(chk.get("expression", "")).strip()
    e = re.sub(r"\s+", "", expr.lower())

    # Strong violation-pattern signals.
    if e.startswith("not(") or e.startswith("not"):
        return "violation"
    if re.search(r"(<0|<=0|<=-1|==0|!=|isnone)", e):
        return "violation"

    # Strong validity-pattern signals.
    if re.search(r"(len\(.+\)>0|all\(|any\(|isinstance\()", e):
        return "validity"
    if re.search(r"(>=0|>0|>-1|<=1|<=10)", e):
        return "validity"
    if re.search(r"<(0?\.\d+|[1-9]\d*(?:\.\d+)?)", e) and "<0" not in e:
        return "validity"
    if ("<=" in e or ">=" in e) and "not" not in e:
        return "validity"

    # Fallback to violation semantics.
    return "violation"


def _is_check_triggered(
    chk: Dict[str, Any],
    inputs: Dict[str, Any],
    *,
    compute_fn: Optional[Any] = None,
) -> Tuple[bool, Optional[str]]:
    result, err = _safe_eval_rule(str(chk.get("expression", "")), inputs, compute_fn=compute_fn)
    if result is None:
        return False, err or "evaluation returned no result"
    mode = _infer_predicate_mode(chk)
    if mode == "validity":
        return result is False, None
    return result is True, None


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


def _load_core(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = _load_records(path)
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        fic_id = str(r.get("fic_id", "")).strip()
        if fic_id:
            out[fic_id] = r
    if not out:
        raise ValueError("No valid core cards loaded")
    return out


def _load_retrieval(path: Path) -> List[Dict[str, Any]]:
    rows = _load_records(path)
    out = [r for r in rows if isinstance(r, dict) and str(r.get("fic_id", "")).strip()]
    if not out:
        raise ValueError("No valid retrieval cards loaded")
    return out


def _load_repair(path: Path) -> List[Dict[str, Any]]:
    rows = _load_records(path)
    out = [r for r in rows if isinstance(r, dict) and str(r.get("rule_id", "")).strip()]
    if not out:
        raise ValueError("No valid repair rules loaded")
    return out


@dataclass
class RetrievalCandidate:
    retrieval: Dict[str, Any]
    score: float


def _retrieval_text(card: Dict[str, Any]) -> str:
    pieces: List[str] = []
    for k in ("title", "summary", "domain", "topic", "embedding_text"):
        v = card.get(k)
        if isinstance(v, str):
            pieces.append(v)
    for k in ("selection_hints", "applicable_when", "not_applicable_when", "scope_boundaries", "keywords"):
        vals = card.get(k, [])
        if isinstance(vals, list):
            pieces.extend(str(x) for x in vals)
    for c in card.get("common_confusions", []):
        if isinstance(c, dict):
            pieces.append(str(c.get("label", "")))
            pieces.append(str(c.get("difference", "")))
    return " ".join(pieces)


def retrieve_candidates(cards: List[Dict[str, Any]], query: str, top_k: int) -> List[RetrievalCandidate]:
    q_tokens = set(_tokenize(query))
    scored: List[RetrievalCandidate] = []
    for card in cards:
        c_tokens = set(_tokenize(_retrieval_text(card)))
        overlap = len(q_tokens & c_tokens)
        denom = max(len(q_tokens), 1)
        score = overlap / denom
        scored.append(RetrievalCandidate(retrieval=card, score=score))
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]


def retrieve_candidates_from_store(
    store: SQLAlchemyArtifactStore,
    *,
    query: str,
    top_k: int,
    domain: Optional[str] = None,
    topic: Optional[str] = None,
) -> List[RetrievalCandidate]:
    rows = store.retrieve_candidates(query=query, top_k=top_k, domain=domain, topic=topic)
    return [RetrievalCandidate(retrieval=row, score=float(row.get("score", 0.0))) for row in rows]


def _schema_selection() -> Any:
    return genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "decision": genai_types.Schema(type=genai_types.Type.STRING, enum=["select_card", "abstain_m", "abstain_n"]),
            "chosen_fic_id": genai_types.Schema(type=genai_types.Type.STRING),
            "reason": genai_types.Schema(type=genai_types.Type.STRING),
            "support_gap_reason": genai_types.Schema(type=genai_types.Type.STRING),
            "ambiguity_tags": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
            "clarification_questions": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
        },
        required=[
            "decision",
            "chosen_fic_id",
            "reason",
            "support_gap_reason",
            "ambiguity_tags",
            "clarification_questions",
        ],
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


def _schema_critic_check() -> Any:
    return genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "needs_clarification": genai_types.Schema(type=genai_types.Type.BOOLEAN),
            "triggered_hint_ids": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
            "ambiguity_tags": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
            "clarification_questions": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
            "clarification_options": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
            "reason": genai_types.Schema(type=genai_types.Type.STRING),
        },
        required=[
            "needs_clarification",
            "triggered_hint_ids",
            "ambiguity_tags",
            "clarification_questions",
            "clarification_options",
            "reason",
        ],
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
        r = c.retrieval
        block.append(
            f"Candidate {i} | fic_id={r.get('fic_id')} | score={c.score:.3f}\n"
            f"title={r.get('title')}\n"
            f"topic={r.get('topic')}\n"
            f"summary={r.get('summary','')}\n"
            f"selection_hints={r.get('selection_hints', [])}\n"
            f"applicable_when={r.get('applicable_when', [])}\n"
            f"scope_boundaries={r.get('scope_boundaries', [])}"
        )
    prompt = f"""
You are selecting a financial formula card from candidates.

Decision policy:
1) If one candidate clearly matches user intent, return decision="select_card" and chosen_fic_id.
2) If intent is ambiguous (e.g., could be NPV vs IRR vs payback) or no candidate fits, return decision="abstain_m".
3) If intent is clear but current candidate library does not support the requested logic/formula, return decision="abstain_n".
4) For abstain_m, provide 1-3 clarification_questions and ambiguity_tags.
5) For abstain_n, provide support_gap_reason; clarification_questions can be empty.

Question:
{question}

Context:
{context}

Candidates:
{chr(10).join(block)}

Return JSON only.
"""
    return _llm_json(client, model=model, prompt=prompt, schema=_schema_selection())


def _extract_inputs_with_llm(client: Any, model: str, question: str, context: str, core: Dict[str, Any]) -> Dict[str, Any]:
    required = core.get("inputs", [])
    prompt = f"""
Extract input values from question/context for this core card.
If missing, set status="missing" and value="".
Keep list-like values as JSON array strings when input type is array.
Keep rates with explicit signal (e.g., 8% or 0.08) and summarize normalization decisions in normalization_note.

Required inputs:
{json.dumps(required, ensure_ascii=False, indent=2)}

Question:
{question}

Context:
{context}

Return JSON only.
"""
    return _llm_json(client, model=model, prompt=prompt, schema=_schema_extraction())


def _semantic_echeck_with_llm(
    client: Any,
    model: str,
    question: str,
    context: str,
    semantic_checks: List[Dict[str, Any]],
    provided_inputs: Dict[str, Any],
) -> Dict[str, Any]:
    # Backward compatibility shim for older callers.
    return _critic_check_with_llm(
        client=client,
        model=model,
        question=question,
        context=context,
        semantic_hints=semantic_checks,
        provided_inputs=provided_inputs,
    )


def _critic_check_with_llm(
    client: Any,
    model: str,
    question: str,
    context: str,
    semantic_hints: List[Dict[str, Any]],
    provided_inputs: Dict[str, Any],
) -> Dict[str, Any]:
    has_hints = bool(semantic_hints)
    mode_desc = "hint_oriented" if has_hints else "open"
    mode_rules = (
        "- semantic_hints are available: use them as primary reference and fill triggered_hint_ids when relevant.\n"
        "- If a hint is not relevant, do not trigger it.\n"
    ) if has_hints else (
        "- semantic_hints are empty: perform OPEN semantic ambiguity scan from question/context.\n"
        "- In open mode, triggered_hint_ids should be empty.\n"
        "- If clarification is needed, propose concise clarification_options when possible.\n"
    )
    prompt = f"""
You are the VerifiQuant I-gate Critic Agent.
Goal: detect hidden semantic ambiguity before deterministic execution.

Ambiguity examples:
- FX direction (USD/TWD vs TWD/USD)
- time basis (begin-year vs end-year cash flow)
- compounding convention (monthly vs annual)

Rules:
- If ambiguity materially changes calculation interpretation, set needs_clarification=true.
- Provide concise clarification questions.
- Mode: {mode_desc}
{mode_rules}

Question:
{question}

Context:
{context}

Semantic hints:
{json.dumps(semantic_hints, ensure_ascii=False, indent=2)}

Provided inputs:
{json.dumps(provided_inputs, ensure_ascii=False, indent=2)}

Return JSON only.
"""
    return _llm_json(client, model=model, prompt=prompt, schema=_schema_critic_check())


def _evaluate_execution(core: Dict[str, Any], provided_inputs: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    execution = core.get("execution", {})
    code = str(execution.get("code", ""))
    output_name = str((core.get("output") or {}).get("name", ""))
    if "def compute(inputs)" not in code:
        return None, "execution code missing compute(inputs)"
    if not output_name:
        return None, "output.name missing in core"

    safe_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "all": all,
        "any": any,
        "isinstance": isinstance,
        "bool": bool,
        "str": str,
        "float": float,
        "int": int,
        "pow": pow,
        "round": round,
        "range": range,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "zip": zip,
        "enumerate": enumerate,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "Exception": Exception,
    }
    local_env: Dict[str, Any] = {}
    try:
        exec(code, {"__builtins__": safe_builtins}, local_env)
        fn = local_env.get("compute")
        if not callable(fn):
            return None, "compute function missing after exec"
        result = fn(provided_inputs)
        if isinstance(result, dict):
            if output_name not in result:
                return None, f"output '{output_name}' missing from compute result"
            raw_output = result.get(output_name)
        else:
            # Backward compatibility: some generated cards return scalar directly.
            raw_output = result
        val = _parse_number(raw_output)
        if val is None:
            return None, "output value not numeric"
        return val, None
    except Exception as exc:
        return None, str(exc)


def _build_repair_index(repair_rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in repair_rows:
        fic_id = str(row.get("fic_id", "")).strip()
        rid = str(row.get("rule_id", "")).strip()
        if fic_id and rid:
            out[(fic_id, rid)] = row
    return out


def _summarize_repairs(fic_id: str, rule_ids: List[str], repair_index: Dict[Tuple[str, str], Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for rid in rule_ids:
        rule = repair_index.get((fic_id, rid))
        if not rule:
            rule = repair_index.get((GLOBAL_SCOPE_FIC_ID, rid))
        if not rule:
            continue
        out.append(
            {
                "rule_id": rid,
                "title": rule.get("title", ""),
                "user_message": rule.get("user_message", ""),
                "repair_action": rule.get("repair_action", {}),
                "allowed_next_steps": rule.get("allowed_next_steps", []),
                "ask_user_for": rule.get("ask_user_for", []),
            }
        )
    return out


def _default_scope_repair_hint() -> List[Dict[str, Any]]:
    return [
        {
            "rule_id": str(GLOBAL_N_NOT_SUPPORTED_RULE.get("rule_id", "global_n_not_supported")),
            "title": str(GLOBAL_N_NOT_SUPPORTED_RULE.get("title", "")),
            "user_message": str(GLOBAL_N_NOT_SUPPORTED_RULE.get("user_message", "")),
            "repair_action": GLOBAL_N_NOT_SUPPORTED_RULE.get("repair_action", {}),
            "allowed_next_steps": GLOBAL_N_NOT_SUPPORTED_RULE.get("allowed_next_steps", []),
            "ask_user_for": GLOBAL_N_NOT_SUPPORTED_RULE.get("ask_user_for", []),
        }
    ]


def _i_rule_id_from_hint_id(hint_id: str, i_level: str = "soft") -> str:
    token = re.sub(r"[^a-zA-Z0-9_]+", "_", str(hint_id or "").strip().lower()).strip("_")
    level = str(i_level or "soft").strip().lower()
    if level not in {"hard", "soft"}:
        level = "soft"
    return f"i_{token or 'semantic_ambiguity'}_{level}"


def _semantic_hint_level(hint: Dict[str, Any]) -> str:
    level = str(hint.get("i_level", "soft")).strip().lower()
    if level not in {"hard", "soft"}:
        return "soft"
    return level


def _infer_open_critic_i_level(critic: Dict[str, Any]) -> str:
    reason = str(critic.get("reason", "") or "").lower()
    tags = [str(x).strip().lower() for x in critic.get("ambiguity_tags", []) if str(x).strip()]
    hard_signals = [
        "formula",
        "definition",
        "direction",
        "basis",
        "unit",
        "scale",
        "numerator",
        "denominator",
        "fx",
        "currency",
    ]
    if any(sig in reason for sig in hard_signals):
        return "hard"
    if any(any(sig in tag for sig in hard_signals) for tag in tags):
        return "hard"
    if "materially" in reason or "significantly different" in reason:
        return "hard"
    return "soft"


def _build_soft_warnings(
    *,
    critic: Dict[str, Any],
    semantic_hints: List[Dict[str, Any]],
    triggered_hint_ids: List[str],
) -> List[Dict[str, Any]]:
    hint_by_id = {str(h.get("id", "")).strip(): h for h in semantic_hints if str(h.get("id", "")).strip()}
    warnings: List[Dict[str, Any]] = []

    if triggered_hint_ids:
        for hid in triggered_hint_ids:
            hint = hint_by_id.get(hid)
            if not hint:
                continue
            if _semantic_hint_level(hint) != "soft":
                continue
            warnings.append(
                {
                    "hint_id": hid,
                    "i_level": "soft",
                    "impact_scope": str(hint.get("impact_scope", "output_interpretation")),
                    "assumption_if_not_clarified": str(
                        hint.get("assumption_if_not_clarified", "Proceed with default interpretation.")
                    ),
                    "clarification_question": str(hint.get("clarification_question", "")),
                    "options": [str(x) for x in (hint.get("options") or []) if str(x).strip()],
                }
            )
        return warnings

    # open-mode soft fallback
    warnings.append(
        {
            "hint_id": "open_critic",
            "i_level": "soft",
            "impact_scope": "output_interpretation",
            "assumption_if_not_clarified": "Proceed with default interpretation inferred from context.",
            "clarification_question": "; ".join(
                [str(x).strip() for x in critic.get("clarification_questions", []) if str(x).strip()]
            ),
            "options": [str(x).strip() for x in critic.get("clarification_options", []) if str(x).strip()],
        }
    )
    return warnings


def _best_scope_repair(
    candidate_ids: List[Any],
    repair_index: Dict[Tuple[str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    for cid in candidate_ids:
        fic_id = str(cid or "").strip()
        if not fic_id:
            continue
        rows = _summarize_repairs(fic_id, ["global_n_not_supported"], repair_index)
        if rows:
            return rows
    rows = _summarize_repairs(GLOBAL_SCOPE_FIC_ID, ["global_n_not_supported"], repair_index)
    if rows:
        return rows
    return _default_scope_repair_hint()


def run_case(
    *,
    row: Dict[str, Any],
    core_by_id: Dict[str, Dict[str, Any]],
    retrieval_cards: List[Dict[str, Any]],
    store: Optional[SQLAlchemyArtifactStore],
    repair_index: Dict[Tuple[str, str], Dict[str, Any]],
    client: Any,
    selector_model: str,
    extractor_model: str,
    judge_model: str,
    top_k: int,
    m_min_top_score: float,
    debug_sanity: bool = False,
) -> Dict[str, Any]:
    q = str(row.get("question", "") or "")
    c = str(row.get("context", "") or "")
    domain = str(row.get("domain", "") or "").strip() or None
    topic = str(row.get("topic", "") or "").strip() or None
    case_id = str(row.get("case_id") or row.get("question_id") or "")
    base = {"case_id": case_id, "has_i_soft": False, "soft_warnings": []}
    debug_prefix = f"[sanity-debug][{case_id or 'unknown_case'}]"

    def _dbg(message: str) -> None:
        if debug_sanity:
            print(f"{debug_prefix} {message}")

    if store is not None:
        candidates = retrieve_candidates_from_store(
            store,
            query=f"{q}\n{c}",
            top_k=top_k,
            domain=domain,
            topic=topic,
        )
    else:
        candidates = retrieve_candidates(retrieval_cards, f"{q}\n{c}", top_k=top_k)
    _dbg(
        "retrieval: "
        + ", ".join(
            f"{str(cand.retrieval.get('fic_id', '')).strip() or '<missing_fic_id>'}@{cand.score:.3f}"
            for cand in candidates
        )
    )
    if not candidates:
        _dbg("early-exit: N (no candidate cards retrieved)")
        return {
            **base,
            "status": "refusal",
            "diagnostic_type": "N",
            "funnel_layer": "Scope",
            "gate_action": "graceful_exit",
            "reason": "No candidate cards retrieved",
            "support_gap_reason": "no_candidate_cards",
            "candidate_ids": [],
            "ambiguity_tags": [],
            "clarification_request": None,
            "repair_hints": [],
        }

    selection = _select_card_with_llm(client, selector_model, q, c, candidates)
    decision = str(selection.get("decision", "")).strip()
    chosen_fic_id = str(selection.get("chosen_fic_id", "")).strip()
    candidate_ids = [cand.retrieval.get("fic_id") for cand in candidates]
    support_gap_reason = str(selection.get("support_gap_reason", "")).strip()
    ambiguity_tags = [str(x).strip() for x in selection.get("ambiguity_tags", []) if str(x).strip()]
    clarification_questions = [str(x).strip() for x in selection.get("clarification_questions", []) if str(x).strip()]
    _dbg(
        "selector: "
        f"decision={decision or '<empty>'}, chosen_fic_id={chosen_fic_id or '<empty>'}, "
        f"reason={str(selection.get('reason', '')).strip() or '<empty>'}"
    )

    if decision == "abstain_m":
        _dbg("early-exit: M (selector abstain_m)")
        return {
            **base,
            "status": "refusal",
            "diagnostic_type": "M",
            "funnel_layer": "Intent",
            "gate_action": "refusal",
            "reason": selection.get("reason", "Ambiguous task intent before card commitment"),
            "candidate_ids": candidate_ids,
            "support_gap_reason": None,
            "ambiguity_tags": ambiguity_tags,
            "clarification_request": {
                "questions": clarification_questions
                or ["Please specify the target metric (for example NPV, IRR, or Payback)."],
                "options": [],
            },
        }

    if decision == "abstain_n":
        _dbg("early-exit: N (selector abstain_n)")
        return {
            **base,
            "status": "refusal",
            "diagnostic_type": "N",
            "funnel_layer": "Scope",
            "gate_action": "graceful_exit",
            "reason": selection.get("reason", "Requested intent is out of supported scope"),
            "candidate_ids": candidate_ids,
            "support_gap_reason": support_gap_reason or "unsupported_formula_family",
            "ambiguity_tags": ambiguity_tags,
            "clarification_request": None,
            "repair_hints": _best_scope_repair(candidate_ids, repair_index),
        }

    if decision != "select_card" or not chosen_fic_id:
        _dbg("early-exit: M (selector did not commit valid card)")
        return {
            **base,
            "status": "refusal",
            "diagnostic_type": "M",
            "funnel_layer": "Intent",
            "gate_action": "refusal",
            "reason": "Selector did not commit a valid card decision",
            "candidate_ids": candidate_ids,
            "support_gap_reason": None,
            "ambiguity_tags": ambiguity_tags,
            "clarification_request": {
                "questions": ["Please specify the target metric explicitly (for example NPV, IRR, Payback)."],
                "options": [],
            },
        }

    top_score = candidates[0].score if candidates else 0.0
    _dbg(f"retrieval-confidence: top_score={top_score:.3f}, threshold={m_min_top_score:.3f}")
    if top_score < m_min_top_score:
        _dbg("early-exit: N (low retrieval confidence)")
        return {
            **base,
            "status": "refusal",
            "diagnostic_type": "N",
            "funnel_layer": "Scope",
            "gate_action": "graceful_exit",
            "reason": f"Low retrieval confidence before card commitment (top score={top_score:.3f})",
            "candidate_ids": candidate_ids,
            "support_gap_reason": "low_retrieval_confidence",
            "ambiguity_tags": ambiguity_tags,
            "clarification_request": None,
            "repair_hints": _best_scope_repair(candidate_ids, repair_index),
        }

    core = core_by_id.get(chosen_fic_id)
    if core is None:
        _dbg(f"early-exit: N (missing core card for chosen_fic_id={chosen_fic_id})")
        return {
            **base,
            "status": "refusal",
            "diagnostic_type": "N",
            "funnel_layer": "Scope",
            "gate_action": "graceful_exit",
            "reason": f"Chosen fic_id '{chosen_fic_id}' has no matching core card",
            "candidate_ids": candidate_ids,
            "support_gap_reason": "missing_core_card",
            "ambiguity_tags": ambiguity_tags,
            "clarification_request": None,
            "repair_hints": _best_scope_repair(candidate_ids, repair_index),
        }

    # card_committed=True beyond this point; only F/E/I/C/success should appear.
    extraction = _extract_inputs_with_llm(client, extractor_model, q, c, core)
    normalization_note = str(extraction.get("normalization_note", "") or "")

    core_inputs = core.get("inputs", [])
    type_map = {str(inp.get("name")): str(inp.get("type", "number")) for inp in core_inputs}
    unit_map = {str(inp.get("name")): inp.get("unit") for inp in core_inputs}
    desc_map = {str(inp.get("name")): str(inp.get("description", "")) for inp in core_inputs}
    required_names = [str(inp.get("name")) for inp in core_inputs if bool(inp.get("required", True))]
    extracted = {str(x.get("name")): x for x in extraction.get("inputs", []) if x.get("name")}

    missing = []
    provided_inputs: Dict[str, Any] = {}
    for name in required_names:
        item = extracted.get(name)
        if not item or item.get("status") != "provided":
            missing.append(name)
            continue
        parsed = _parse_typed_value(
            item.get("value", ""),
            type_map.get(name, "number"),
            unit=unit_map.get(name),
            description=desc_map.get(name, ""),
        )
        if parsed is None:
            missing.append(name)
        else:
            provided_inputs[name] = parsed

    for name, item in extracted.items():
        if name in provided_inputs:
            continue
        parsed = _parse_typed_value(
            item.get("value", ""),
            type_map.get(name, "number"),
            unit=unit_map.get(name),
            description=desc_map.get(name, ""),
        )
        if parsed is not None:
            provided_inputs[name] = parsed
    _dbg(
        "input-binding: "
        f"required={required_names}, missing={missing}, provided_keys={sorted(list(provided_inputs.keys()))}, "
        f"provided_inputs={json.dumps(provided_inputs, ensure_ascii=False, sort_keys=True)}"
    )

    if missing:
        _dbg("early-exit: F (missing or unparsable required inputs)")
        return {
            **base,
            "status": "error",
            "diagnostic_type": "F",
            "funnel_layer": "Schema",
            "gate_action": "slot_filling",
            "reason": "Missing or unparsable required inputs",
            "fic_id": chosen_fic_id,
            "candidate_ids": candidate_ids,
            "support_gap_reason": None,
            "ambiguity_tags": [],
            "clarification_request": None,
            "requested_fields": missing,
            "provided_inputs": provided_inputs,
            "normalization_note": normalization_note,
            "repair_hints": [
                {
                    "repair_action": {"type": "request_missing_fields", "target": ",".join(missing)},
                    "ask_user_for": [{"slot": m, "label": m, "type": "text", "required": True, "options": []} for m in missing],
                    "allowed_next_steps": ["ask_followup", "rerun_same_fic"],
                }
            ],
        }

    # Generic scale sanity: percent-point semantic inputs should not silently run in decimal scale.
    percent_point_fields = [
        str(inp.get("name"))
        for inp in core_inputs
        if _is_percent_point_unit(inp.get("unit"), inp.get("description"))
    ]
    percent_point_values = [
        float(provided_inputs[name])
        for name in percent_point_fields
        if name in provided_inputs and isinstance(provided_inputs[name], (int, float))
    ]
    if len(percent_point_values) >= 2:
        max_abs = max(abs(v) for v in percent_point_values)
        if max_abs <= 1.0:
            triggered = ["auto_percent_scale_mismatch"]
            _dbg(f"early-exit: E (triggered_rule_ids={triggered}, max_abs_percent_point_input={max_abs})")
            return {
                **base,
                "status": "alert",
                "diagnostic_type": "E",
                "funnel_layer": "Boundary",
                "gate_action": "deterministic_alert",
                "reason": (
                    "Detected potential percent scale mismatch: inputs with '%' semantics appear to be "
                    "in decimal scale (e.g., 0.125 instead of 12.5)."
                ),
                "fic_id": chosen_fic_id,
                "candidate_ids": candidate_ids,
                "support_gap_reason": None,
                "ambiguity_tags": ["percent_scale_mismatch"],
                "clarification_request": None,
                "provided_inputs": provided_inputs,
                "triggered_rule_ids": triggered,
                "normalization_note": normalization_note,
                "repair_hints": [
                    {
                        "rule_id": "auto_percent_scale_mismatch",
                        "title": "Percent Scale Clarification",
                        "user_message": "Please confirm whether rate/return inputs use percentage points or decimals.",
                        "repair_action": {"type": "confirm_scale", "target": "percent_vs_decimal"},
                        "allowed_next_steps": ["ask_followup", "rerun_same_fic"],
                        "ask_user_for": [
                            {
                                "slot": "rate_scale",
                                "label": "Rate Scale",
                                "type": "enum",
                                "required": True,
                                "options": [
                                    {"value": "percent_points", "label": "12.5 means 12.5%"},
                                    {"value": "decimal", "label": "0.125 means 12.5%"},
                                ],
                            }
                        ],
                    }
                ],
            }

    # Pre-compile the FIC's compute function so E-check expressions that call
    # ``compute({...})`` (e.g. plausibility range checks) can be evaluated.
    _fic_compute_fn: Optional[Any] = None
    _fic_code = str((core.get("execution") or {}).get("code", ""))
    if "def compute(inputs)" in _fic_code:
        _safe_builtins_exec = {
            "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
            "all": all, "any": any, "isinstance": isinstance, "bool": bool,
            "str": str, "float": float, "int": int, "pow": pow, "round": round,
            "range": range, "list": list, "dict": dict, "tuple": tuple,
            "set": set, "zip": zip, "enumerate": enumerate,
            "ValueError": ValueError, "TypeError": TypeError, "Exception": Exception,
            "math": math,
        }
        try:
            _exec_env: Dict[str, Any] = {}
            exec(_fic_code, {"__builtins__": _safe_builtins_exec}, _exec_env)  # noqa: S102
            _fic_compute_fn = _exec_env.get("compute")
        except Exception:
            pass  # SyntaxError or other issue — will surface at C-execution stage

    # Classify E-check eval errors: structural TypeError/AttributeError →  F-class;
    # everything else (NameError, SyntaxError, genuine logic errors) → C-class.
    _STRUCTURAL_ERROR_SIGNALS = (
        "is not subscriptable",
        "has no attribute",
        "is not iterable",
        "object is not callable",
        "cannot unpack",
        "must be",          # e.g. 'int' object must be ...
    )

    checks = core.get("diagnostic_checks", [])
    e_checks = [chk for chk in checks if str(chk.get("diagnostic_type", "")).upper() == "E"]
    auto_triggered: List[str] = []
    eval_errors: List[Dict[str, str]] = []        # genuine C-class issues
    structural_errors: List[Dict[str, str]] = []  # input-shape issues → F-class

    for chk in e_checks:
        rule_id = str(chk.get("rule_id", "")).strip()
        ctype = str(chk.get("check_type", "")).strip().lower()
        if ctype in {"deterministic", "normalization"}:
            is_triggered, eval_err = _is_check_triggered(chk, provided_inputs, compute_fn=_fic_compute_fn)
            if eval_err:
                err_record = {
                    "rule_id": rule_id or "<unknown>",
                    "expression": str(chk.get("expression", "")),
                    "error": eval_err,
                }
                # Decide: is this a structural (F) or logic (C) failure?
                if any(sig in eval_err for sig in _STRUCTURAL_ERROR_SIGNALS):
                    structural_errors.append(err_record)
                    _dbg(
                        f"e-check: rule_id={rule_id or '<unknown>'}, triggered=<structural-error[F]>, "
                        f"error={eval_err}"
                    )
                else:
                    eval_errors.append(err_record)
                    _dbg(
                        f"e-check: rule_id={rule_id or '<unknown>'}, triggered=<error[C]>, "
                        f"error={eval_err}"
                    )
                continue
            if is_triggered and rule_id:
                auto_triggered.append(rule_id)
            _dbg(
                f"e-check: rule_id={rule_id or '<unknown>'}, mode={_infer_predicate_mode(chk)}, "
                f"triggered={is_triggered}"
            )

    # Structural errors mean the input shape does not match the FIC schema → F.
    if structural_errors:
        first_s = structural_errors[0]
        missing_structural = [e["rule_id"] for e in structural_errors]
        _dbg(f"early-exit: F (structural E-check error → input shape mismatch on {first_s['rule_id']})")
        return {
            **base,
            "status": "error",
            "diagnostic_type": "F",
            "funnel_layer": "Schema",
            "gate_action": "slot_filling",
            "reason": (
                f"Input structure mismatch detected in rule {first_s['rule_id']}: {first_s['error']}. "
                "One or more inputs have an unexpected type or shape (e.g., a flat list where a "
                "list-of-lists was expected)."
            ),
            "fic_id": chosen_fic_id,
            "candidate_ids": candidate_ids,
            "support_gap_reason": None,
            "ambiguity_tags": ["input_structure_mismatch"],
            "clarification_request": None,
            "provided_inputs": provided_inputs,
            "normalization_note": normalization_note,
            "rule_eval_errors": structural_errors,
            "repair_hints": _summarize_repairs(chosen_fic_id, missing_structural, repair_index),
        }

    if eval_errors:
        first = eval_errors[0]
        _dbg(f"early-exit: C (E-check evaluation error on {first['rule_id']})")
        return {
            **base,
            "status": "error",
            "diagnostic_type": "C",
            "funnel_layer": "Logic",
            "gate_action": "audit_log",
            "reason": f"E-check evaluation error in rule {first['rule_id']}: {first['error']}",
            "fic_id": chosen_fic_id,
            "candidate_ids": candidate_ids,
            "support_gap_reason": None,
            "ambiguity_tags": [],
            "clarification_request": None,
            "provided_inputs": provided_inputs,
            "normalization_note": normalization_note,
            "rule_eval_errors": eval_errors,
        }

    triggered = list(dict.fromkeys(auto_triggered))
    if triggered:
        _dbg(f"early-exit: E (triggered_rule_ids={triggered})")
        reason = "Deterministic E-type checks detected potential inconsistencies."
        return {
            **base,
            "status": "alert",
            "diagnostic_type": "E",
            "funnel_layer": "Boundary",
            "gate_action": "deterministic_alert",
            "reason": reason or "Potential E-type inconsistency detected",
            "fic_id": chosen_fic_id,
            "candidate_ids": candidate_ids,
            "support_gap_reason": None,
            "ambiguity_tags": [],
            "clarification_request": None,
            "provided_inputs": provided_inputs,
            "triggered_rule_ids": triggered,
            "normalization_note": normalization_note,
            "repair_hints": _summarize_repairs(chosen_fic_id, triggered, repair_index),
        }

    semantic_hints = [h for h in (core.get("semantic_hints") or []) if isinstance(h, dict)]
    critic = _critic_check_with_llm(
        client=client,
        model=judge_model,
        question=q,
        context=c,
        semantic_hints=semantic_hints,
        provided_inputs=provided_inputs,
    )
    needs_clarification = bool(critic.get("needs_clarification"))
    soft_warnings: List[Dict[str, Any]] = []
    if needs_clarification:
        triggered_hint_ids = [str(x).strip() for x in critic.get("triggered_hint_ids", []) if str(x).strip()]
        hint_by_id = {str(h.get("id", "")).strip(): h for h in semantic_hints if str(h.get("id", "")).strip()}
        i_tags = [str(x).strip() for x in critic.get("ambiguity_tags", []) if str(x).strip()]
        clar_qs = [str(x).strip() for x in critic.get("clarification_questions", []) if str(x).strip()]
        model_options = [str(x).strip() for x in critic.get("clarification_options", []) if str(x).strip()]
        hard_triggered = [
            hid for hid in triggered_hint_ids if _semantic_hint_level(hint_by_id.get(hid, {})) == "hard"
        ]
        soft_triggered = [
            hid for hid in triggered_hint_ids if _semantic_hint_level(hint_by_id.get(hid, {})) == "soft"
        ]

        open_inferred_level = _infer_open_critic_i_level(critic) if not triggered_hint_ids else None
        is_hard_block = bool(hard_triggered) or (not triggered_hint_ids and open_inferred_level == "hard")
        _dbg(
            "critic: "
            f"needs_clarification={needs_clarification}, triggered_hint_ids={triggered_hint_ids}, "
            f"hard_triggered={hard_triggered}, soft_triggered={soft_triggered}, "
            f"open_inferred_level={open_inferred_level}, is_hard_block={is_hard_block}"
        )

        i_rule_ids = [
            _i_rule_id_from_hint_id(hid, _semantic_hint_level(hint_by_id.get(hid, {})))
            for hid in hard_triggered
        ] if is_hard_block else []
        if is_hard_block and not i_rule_ids:
            i_rule_ids = ["global_i_semantic_ambiguity"]
        i_repairs = _summarize_repairs(chosen_fic_id, i_rule_ids, repair_index) if i_rule_ids else []
        if is_hard_block and not i_repairs:
            i_repairs = _summarize_repairs(chosen_fic_id, ["global_i_semantic_ambiguity"], repair_index)
        options: List[str] = []
        if semantic_hints:
            for hint in semantic_hints:
                if not triggered_hint_ids or str(hint.get("id", "")) in triggered_hint_ids:
                    options.extend([str(x).strip() for x in hint.get("options", []) if str(x).strip()])
        else:
            options.extend(model_options)
        options = list(dict.fromkeys(options))
        if is_hard_block:
            _dbg("early-exit: I_hard (needs clarification)")
            return {
                **base,
                "status": "needs_clarification",
                "diagnostic_type": "I",
                "funnel_layer": "Critic",
                "gate_action": "critic_intervention",
                "reason": str(critic.get("reason", "")).strip() or "Hidden semantic ambiguity detected.",
                "fic_id": chosen_fic_id,
                "candidate_ids": candidate_ids,
                "support_gap_reason": None,
                "ambiguity_tags": i_tags,
                "clarification_request": {
                    "questions": clar_qs or ["Please confirm the intended financial interpretation."],
                    "options": options,
                    "triggered_hint_ids": triggered_hint_ids,
                    "mode": "hint_oriented" if semantic_hints else "open",
                },
                "provided_inputs": provided_inputs,
                "normalization_note": normalization_note,
                "repair_hints": i_repairs,
                "i_level": "hard",
                "has_i_soft": False,
                "soft_warnings": [],
            }

        # I_soft: do not block execution; add warning payload and continue to C-execution.
        soft_warning_ids = soft_triggered if soft_triggered else []
        soft_warnings = _build_soft_warnings(
            critic=critic,
            semantic_hints=semantic_hints,
            triggered_hint_ids=soft_warning_ids,
        )
        _dbg(f"critic: I_soft warnings={len(soft_warnings)}")

    output_value, exec_err = _evaluate_execution(core, provided_inputs)
    if exec_err:
        _dbg(f"early-exit: C (execution error: {exec_err})")
        return {
            **base,
            "status": "error",
            "diagnostic_type": "C",
            "funnel_layer": "Logic",
            "gate_action": "audit_log",
            "reason": exec_err,
            "fic_id": chosen_fic_id,
            "candidate_ids": candidate_ids,
            "support_gap_reason": None,
            "ambiguity_tags": [],
            "clarification_request": None,
            "provided_inputs": provided_inputs,
            "normalization_note": normalization_note,
            "has_i_soft": bool(soft_warnings),
            "soft_warnings": soft_warnings,
        }

    gold = row.get("gold_answer", row.get("answer", row.get("ground_truth")))
    gold_num = _parse_number(gold)
    abs_error, is_correct = _answer_match(q, output_value, gold_num)
    _dbg(f"success: fic_id={chosen_fic_id}, output={output_value}, gold={gold_num}, is_correct={is_correct}")

    return {
        **base,
        "status": "success",
        "diagnostic_type": "None",
        "funnel_layer": "Logic",
        "gate_action": "audit_log",
        "reason": "Execution completed",
        "fic_id": chosen_fic_id,
        "candidate_ids": candidate_ids,
        "support_gap_reason": None,
        "ambiguity_tags": [],
        "clarification_request": None,
        "provided_inputs": provided_inputs,
        "output_var": (core.get("output") or {}).get("name"),
        "output_value": output_value,
        "gold_answer": gold_num if gold_num is not None else gold,
        "abs_error": abs_error,
        "is_correct": is_correct,
        "normalization_note": normalization_note,
        "has_i_soft": bool(soft_warnings),
        "soft_warnings": soft_warnings,
        "execution_trace": {
            "engine": "deterministic_python",
            "entrypoint": "compute",
            "fic_version": core.get("version", "v1"),
        },
    }


class ErrorClassificationAPI:
    """Reusable API wrapper around VerifiQuant diagnostic run_case()."""

    def __init__(
        self,
        *,
        core_by_id: Dict[str, Dict[str, Any]],
        retrieval_cards: List[Dict[str, Any]],
        store: Optional[SQLAlchemyArtifactStore],
        repair_index: Dict[Tuple[str, str], Dict[str, Any]],
        client: Any,
        selector_model: str = "gemini-2.5-flash",
        extractor_model: str = "gemini-2.5-flash",
        judge_model: str = "gemini-2.5-flash",
        top_k: int = 3,
        m_min_top_score: float = 0.05,
    ) -> None:
        self.core_by_id = core_by_id
        self.retrieval_cards = retrieval_cards
        self.store = store
        self.repair_index = repair_index
        self.client = client
        self.selector_model = selector_model
        self.extractor_model = extractor_model
        self.judge_model = judge_model
        self.top_k = top_k
        self.m_min_top_score = m_min_top_score

    @classmethod
    def from_db(
        cls,
        *,
        db_url: str,
        client: Any,
        selector_model: str = "gemini-2.5-flash",
        extractor_model: str = "gemini-2.5-flash",
        judge_model: str = "gemini-2.5-flash",
        top_k: int = 3,
        m_min_top_score: float = 0.05,
    ) -> "ErrorClassificationAPI":
        store = SQLAlchemyArtifactStore(db_url)
        core_by_id = store.load_core_by_id()
        repair_index = store.build_repair_index()
        return cls(
            core_by_id=core_by_id,
            retrieval_cards=[],
            store=store,
            repair_index=repair_index,
            client=client,
            selector_model=selector_model,
            extractor_model=extractor_model,
            judge_model=judge_model,
            top_k=top_k,
            m_min_top_score=m_min_top_score,
        )

    @classmethod
    def from_files(
        cls,
        *,
        core_path: Path,
        retrieval_path: Path,
        repair_path: Path,
        client: Any,
        selector_model: str = "gemini-2.5-flash",
        extractor_model: str = "gemini-2.5-flash",
        judge_model: str = "gemini-2.5-flash",
        top_k: int = 3,
        m_min_top_score: float = 0.05,
    ) -> "ErrorClassificationAPI":
        core_by_id = _load_core(core_path)
        retrieval_cards = _load_retrieval(retrieval_path)
        repair_rows = _load_repair(repair_path)
        validate_artifact_relations(
            core_cards=list(core_by_id.values()),
            retrieval_cards=retrieval_cards,
            repair_rules=repair_rows,
        )
        repair_index = _build_repair_index(repair_rows)
        return cls(
            core_by_id=core_by_id,
            retrieval_cards=retrieval_cards,
            store=None,
            repair_index=repair_index,
            client=client,
            selector_model=selector_model,
            extractor_model=extractor_model,
            judge_model=judge_model,
            top_k=top_k,
            m_min_top_score=m_min_top_score,
        )

    def diagnose_row(
        self,
        row: Dict[str, Any],
        *,
        top_k: Optional[int] = None,
        m_min_top_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        return run_case(
            row=row,
            core_by_id=self.core_by_id,
            retrieval_cards=self.retrieval_cards,
            store=self.store,
            repair_index=self.repair_index,
            client=self.client,
            selector_model=self.selector_model,
            extractor_model=self.extractor_model,
            judge_model=self.judge_model,
            top_k=top_k if top_k is not None else self.top_k,
            m_min_top_score=m_min_top_score if m_min_top_score is not None else self.m_min_top_score,
        )

    def diagnose(
        self,
        *,
        question: str,
        context: str,
        case_id: str = "",
        top_k: Optional[int] = None,
        m_min_top_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        return self.diagnose_row(
            {
                "case_id": case_id,
                "question": question,
                "context": context,
            },
            top_k=top_k,
            m_min_top_score=m_min_top_score,
        )

    def retrieve(
        self,
        *,
        question: str,
        context: str,
        top_k: Optional[int] = None,
        domain: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if self.store is not None:
            rows = self.store.retrieve_candidates(
                query=f"{question}\n{context}",
                top_k=top_k or self.top_k,
                domain=domain,
                topic=topic,
            )
            return rows

        candidates = retrieve_candidates(self.retrieval_cards, f"{question}\n{context}", top_k or self.top_k)
        out: List[Dict[str, Any]] = []
        for cand in candidates:
            row = dict(cand.retrieval)
            row["score"] = cand.score
            out.append(row)
        return out


def create_genai_client_from_env() -> Any:
    if genai is None or genai_types is None:
        raise RuntimeError("Missing dependency: google.genai import failed.")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in environment.")
    return genai.Client(api_key=api_key)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run M/N/F/E/I/C diagnostic pipeline with separated core/retrieval/repair artifacts."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSON/JSONL questions")
    parser.add_argument("--core", type=Path, help="fic_core JSON/JSONL")
    parser.add_argument("--retrieval", type=Path, help="fic_retrieval JSON/JSONL")
    parser.add_argument("--repair", type=Path, help="repair_rule JSON/JSONL")
    parser.add_argument("--db-url", help="Optional SQLAlchemy DB URL. If set, retrieval/core/repair are loaded from DB.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON/JSONL results")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--selector-model", default="gemini-2.5-flash")
    parser.add_argument("--extractor-model", default="gemini-2.5-flash")
    parser.add_argument("--judge-model", default="gemini-2.5-flash")
    parser.add_argument("--m-min-top-score", type=float, default=0.05)
    parser.add_argument(
        "--debug-sanity",
        action="store_true",
        help="Print detailed per-case funnel diagnostics (retrieval, selector, checks, critic).",
    )
    args = parser.parse_args()

    try:
        client = create_genai_client_from_env()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    rows = _load_records(args.input)
    if args.max_records > 0:
        rows = rows[: args.max_records]

    store: Optional[SQLAlchemyArtifactStore] = None
    if args.db_url:
        store = SQLAlchemyArtifactStore(args.db_url)
        stats = store.validate_integrity()
        print(
            "[run_run_error_classification_pipeline] db relation check passed: "
            f"core={stats['core_count']}, retrieval={stats['retrieval_count']}, "
            f"repair={stats['repair_count']}, diagnostic_rules={stats['diagnostic_rule_count']}"
        )
        core_by_id = store.load_core_by_id()
        retrieval_cards: List[Dict[str, Any]] = []
        repair_index = store.build_repair_index()
    else:
        if not (args.core and args.retrieval and args.repair):
            raise ValueError("When --db-url is not provided, --core --retrieval --repair are all required.")
        core_by_id = _load_core(args.core)
        retrieval_cards = _load_retrieval(args.retrieval)
        repair_rows = _load_repair(args.repair)
        validate_artifact_relations(
            core_cards=list(core_by_id.values()),
            retrieval_cards=retrieval_cards,
            repair_rules=repair_rows,
        )
        repair_index = _build_repair_index(repair_rows)
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        cid = row.get("case_id", row.get("question_id", idx))
        print(f"[run_run_error_classification_pipeline] processing {cid}")
        result = run_case(
            row=row,
            core_by_id=core_by_id,
            retrieval_cards=retrieval_cards,
            store=store,
            repair_index=repair_index,
            client=client,
            selector_model=args.selector_model,
            extractor_model=args.extractor_model,
            judge_model=args.judge_model,
            top_k=args.top_k,
            m_min_top_score=args.m_min_top_score,
            debug_sanity=args.debug_sanity,
        )

        for k in ("variant_type", "expected_status", "expected_diagnostic_type", "source_sample_id", "reason"):
            if k in row and k not in result:
                result[f"expected_{k}" if k == "reason" else k] = row[k]
        out.append(result)

    _dump_records(args.output, out)
    print(f"Wrote {len(out)} results to {args.output}")


if __name__ == "__main__":
    main()
