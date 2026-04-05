from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from verifiquant.preprocessing.common import (
    ConversionInput,
    call_gemini_json,
    normalize_input_type,
    normalize_label,
    optional_str,
    require_compute_code,
    safe_id,
)
from verifiquant.taxonomy import TAXONOMY, taxonomy_json, validate_domain_topic

try:
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai_types = None


STAGE_CORE_PROMPT = """Role:
You are a financial software engineer generating a FIC core card.

Task:
Convert one dataset case into a reusable `fic_core` object with:
- fic_id, name, short_description
- domain, topic
- inputs, output
- execution (deterministic python)
- selection_hints
- semantic_hints
- diagnostics.invariants / diagnostics.scale_checks

Critical rules:
1) `python_solution` is PRIMARY for executable logic.
2) `function` is SECONDARY for naming and financial semantics.
3) Do not hardcode case-specific constants into execution; use inputs.

Taxonomy policy:
- Domain MUST be one of taxonomy domains.
- Topic should prefer existing topics under that domain.
- If no existing topic is suitable, you MAY create a concise snake_case topic under that existing domain.
- Never invent a new domain.

Diagnostic rules policy:
- Provide 3 to 8 checks.
- check_type must be one of: deterministic, normalization.
- deterministic checks must include a valid python boolean `expression` using input names.
- For deterministic/normalization checks, include `predicate_mode`:
  - `violation`: expression=True means the rule is triggered (preferred default).
  - `validity`: expression=True means the input is valid, and expression=False triggers the rule.
- In diagnostics checks, use ONLY F or E. Do NOT output M/N/I/C.
- Why: M is handled at retrieval/card-selection stage before card commitment.
- Mapping guidance:
  - Missing required field / invalid type / hard formula precondition -> F
  - Unit/scale/binding/plausibility inconsistency -> E
- severity must be error or alert.
- semantic_hints are for I-gate only (hidden ambiguity checks like FX direction/time basis),
  and must include clarification question plus fixed options.

Input case:
<DEFINITION_JSON>
{definition_json}
</DEFINITION_JSON>

Taxonomy:
<TAXONOMY_JSON>
{taxonomy_json}
</TAXONOMY_JSON>

Output:
Return JSON only.
"""


def stage_core_schema() -> Any:
    if genai_types is None:
        return None
    semantic_hint = genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "id": genai_types.Schema(type=genai_types.Type.STRING),
            "ambiguity_type": genai_types.Schema(type=genai_types.Type.STRING),
            "trigger_signal": genai_types.Schema(type=genai_types.Type.STRING),
            "clarification_question": genai_types.Schema(type=genai_types.Type.STRING),
            "options": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
        },
        required=["id", "ambiguity_type", "clarification_question", "options"],
    )
    check = genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "rule_id": genai_types.Schema(type=genai_types.Type.STRING),
            "name": genai_types.Schema(type=genai_types.Type.STRING),
            "check_type": genai_types.Schema(
                type=genai_types.Type.STRING,
                enum=["deterministic", "normalization"],
            ),
            "diagnostic_type": genai_types.Schema(
                type=genai_types.Type.STRING,
                enum=["F", "E"],
            ),
            "severity": genai_types.Schema(
                type=genai_types.Type.STRING,
                enum=["error", "alert"],
            ),
            "predicate_mode": genai_types.Schema(
                type=genai_types.Type.STRING,
                enum=["violation", "validity"],
            ),
            "expression": genai_types.Schema(type=genai_types.Type.STRING),
            "applies_to": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
            "description": genai_types.Schema(type=genai_types.Type.STRING),
        },
        required=["rule_id", "name", "check_type", "diagnostic_type", "severity", "description"],
    )

    return genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "fic_id": genai_types.Schema(type=genai_types.Type.STRING),
            "name": genai_types.Schema(type=genai_types.Type.STRING),
            "short_description": genai_types.Schema(type=genai_types.Type.STRING),
            "domain": genai_types.Schema(type=genai_types.Type.STRING),
            "topic": genai_types.Schema(type=genai_types.Type.STRING),
            "selection_hints": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
            "semantic_hints": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=semantic_hint,
            ),
            "inputs": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        "name": genai_types.Schema(type=genai_types.Type.STRING),
                        "type": genai_types.Schema(type=genai_types.Type.STRING),
                        "required": genai_types.Schema(type=genai_types.Type.BOOLEAN),
                        "description": genai_types.Schema(type=genai_types.Type.STRING),
                        "unit": genai_types.Schema(type=genai_types.Type.STRING),
                        "aliases": genai_types.Schema(
                            type=genai_types.Type.ARRAY,
                            items=genai_types.Schema(type=genai_types.Type.STRING),
                        ),
                    },
                    required=["name", "type", "required", "description"],
                ),
            ),
            "output": genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "name": genai_types.Schema(type=genai_types.Type.STRING),
                    "type": genai_types.Schema(type=genai_types.Type.STRING),
                    "unit": genai_types.Schema(type=genai_types.Type.STRING),
                    "description": genai_types.Schema(type=genai_types.Type.STRING),
                },
                required=["name", "type"],
            ),
            "execution": genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "language": genai_types.Schema(type=genai_types.Type.STRING),
                    "entrypoint": genai_types.Schema(type=genai_types.Type.STRING),
                    "code": genai_types.Schema(type=genai_types.Type.STRING),
                    "deterministic": genai_types.Schema(type=genai_types.Type.BOOLEAN),
                },
                required=["language", "entrypoint", "code", "deterministic"],
            ),
            "diagnostics": genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "invariants": genai_types.Schema(
                        type=genai_types.Type.ARRAY,
                        items=check,
                    ),
                    "scale_checks": genai_types.Schema(
                        type=genai_types.Type.ARRAY,
                        items=check,
                    ),
                },
                required=["invariants", "scale_checks"],
            ),
            "diagnostic_checks": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=check,
            ),
        },
        required=[
            "fic_id",
            "name",
            "short_description",
            "domain",
            "topic",
            "selection_hints",
            "semantic_hints",
            "inputs",
            "output",
            "execution",
            "diagnostics",
        ],
    )


def build_stage_core_prompt(defn: ConversionInput) -> str:
    definition_json = json.dumps(
        {
            "source_meta": defn.source_meta,
            "function": defn.function,
            "python_solution": defn.python_solution,
            "context": defn.context,
            "question": defn.question,
        },
        ensure_ascii=False,
        indent=2,
    )
    return STAGE_CORE_PROMPT.format(
        definition_json=definition_json,
        taxonomy_json=taxonomy_json(indent=2),
    )


def _normalize_input_item(item: Dict[str, Any]) -> Dict[str, Any]:
    aliases = item.get("aliases", [])
    if not isinstance(aliases, list):
        aliases = []
    return {
        "name": normalize_label(item.get("name", "")),
        "type": normalize_input_type(item.get("type", "number")),
        "required": bool(item.get("required", True)),
        "description": str(item.get("description", "")).strip(),
        "unit": optional_str(item.get("unit")),
        "aliases": [normalize_label(a) for a in aliases if normalize_label(a)],
    }


def _normalize_output(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": normalize_label(item.get("name", "")),
        "type": normalize_input_type(item.get("type", "number")),
        "unit": optional_str(item.get("unit")),
        "description": optional_str(item.get("description")) or "",
    }


def _infer_predicate_mode_from_expression(expression: str) -> str:
    expr = "".join(str(expression or "").strip().lower().split())
    if not expr:
        return "violation"
    if expr.startswith("not(") or expr.startswith("not"):
        return "violation"
    if any(tok in expr for tok in ("<0", "<=0", "<=-1", "==0", "!=", "isnone")):
        return "violation"
    if any(tok in expr for tok in (">=0", ">0", ">-1", "<=1", "<=10")):
        return "validity"
    if "<=" in expr or ">=" in expr:
        return "validity"
    if "<" in expr and "<0" not in expr:
        return "validity"
    return "violation"


def _normalize_check(check: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "rule_id": normalize_label(check.get("rule_id", "")),
        "name": str(check.get("name", "")).strip(),
        "check_type": str(check.get("check_type", "")).strip().lower(),
        "diagnostic_type": str(check.get("diagnostic_type", "")).strip().upper(),
        "severity": str(check.get("severity", "")).strip().lower(),
        "predicate_mode": str(check.get("predicate_mode", "")).strip().lower(),
        "expression": str(check.get("expression", "") or "").strip(),
        "applies_to": [normalize_label(x) for x in (check.get("applies_to") or []) if normalize_label(x)],
        "description": str(check.get("description", "")).strip(),
    }

    if not out["rule_id"]:
        out["rule_id"] = f"rule_{normalize_label(out['name']) or 'generated'}"
    if out["check_type"] not in {"deterministic", "normalization"}:
        out["check_type"] = "normalization"
    # Core checks are post-card-commit guards, so they should be F/E only.
    if out["diagnostic_type"] == "M":
        if out["check_type"] == "normalization":
            out["diagnostic_type"] = "E"
        else:
            out["diagnostic_type"] = "F"
    elif out["diagnostic_type"] not in {"F", "E"}:
        out["diagnostic_type"] = "E"
    if out["severity"] not in {"error", "alert"}:
        out["severity"] = "alert"
    if out["predicate_mode"] not in {"violation", "validity"}:
        out["predicate_mode"] = _infer_predicate_mode_from_expression(out["expression"])
    if not out["expression"]:
        raise ValueError(f"check '{out['rule_id']}' missing expression")
    if not out["description"]:
        out["description"] = out["name"] or out["rule_id"]
    return out


def _norm_str_list(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for item in items:
        s = str(item or "").strip()
        if s:
            out.append(s)
    return out


def _normalize_semantic_hint(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    hint_id = normalize_label(item.get("id", "")) or f"semantic_hint_{idx}"
    ambiguity_type = normalize_label(item.get("ambiguity_type", "")) or "generic_ambiguity"
    trigger_signal = str(item.get("trigger_signal", "") or "").strip()
    clarification_question = str(item.get("clarification_question", "") or "").strip()
    options = _norm_str_list(item.get("options", []))
    if not clarification_question:
        clarification_question = "Please clarify the intended financial interpretation."
    if not options:
        options = ["Option A", "Option B"]
    return {
        "id": hint_id,
        "ambiguity_type": ambiguity_type,
        "trigger_signal": trigger_signal,
        "clarification_question": clarification_question,
        "options": options,
    }


def _collect_diagnostic_checks(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    diagnostics = raw.get("diagnostics")
    checks_raw: List[Dict[str, Any]] = []
    if isinstance(diagnostics, dict):
        for key, default_type in (("invariants", "deterministic"), ("scale_checks", "normalization")):
            rows = diagnostics.get(key, [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                fixed = dict(row)
                fixed["check_type"] = str(fixed.get("check_type", "")).strip().lower() or default_type
                checks_raw.append(fixed)

    # Backward compatibility with v1 generation output.
    if not checks_raw and isinstance(raw.get("diagnostic_checks"), list):
        checks_raw = [x for x in raw.get("diagnostic_checks", []) if isinstance(x, dict)]
    return [_normalize_check(chk) for chk in checks_raw]


def formalize_core_payload(
    raw: Dict[str, Any],
    *,
    source_meta: Dict[str, Any],
    fallback_id: str,
    allow_new_topic: bool,
) -> Dict[str, Any]:
    fic_id = safe_id(raw.get("fic_id") or raw.get("id") or fallback_id)
    name = str(raw.get("name", "")).strip()
    short_description = str(raw.get("short_description", "")).strip()
    domain, topic, is_new_topic = validate_domain_topic(
        str(raw.get("domain", "")),
        str(raw.get("topic", "")),
        allow_new_topic=allow_new_topic,
    )

    inputs_raw = raw.get("inputs")
    if not isinstance(inputs_raw, list) or not inputs_raw:
        raise ValueError("Stage core requires non-empty inputs.")
    inputs = [_normalize_input_item(x) for x in inputs_raw]
    if any(not inp["name"] for inp in inputs):
        raise ValueError("Every input must include a normalized name.")

    output = _normalize_output(raw.get("output") or {"name": raw.get("output_var")})
    if not output["name"]:
        raise ValueError("output.name is required")

    execution = dict(raw.get("execution") or {})
    execution["language"] = "python"
    execution["entrypoint"] = "compute"
    execution["deterministic"] = True
    execution["code"] = require_compute_code(execution.get("code", ""))

    checks = _collect_diagnostic_checks(raw)
    if not checks:
        raise ValueError("diagnostics checks must be a non-empty list")

    selection_hints = _norm_str_list(raw.get("selection_hints", []))
    if not selection_hints:
        selection_hints = [name, short_description, topic]

    semantic_hints_raw = raw.get("semantic_hints", [])
    if not isinstance(semantic_hints_raw, list):
        semantic_hints_raw = []
    semantic_hints = [
        _normalize_semantic_hint(h, idx)
        for idx, h in enumerate(semantic_hints_raw, start=1)
        if isinstance(h, dict)
    ]

    if not name:
        raise ValueError("name is required")
    if not short_description:
        raise ValueError("short_description is required")

    invariants = [x for x in checks if x.get("check_type") == "deterministic"]
    scale_checks = [x for x in checks if x.get("check_type") == "normalization"]

    return {
        "fic_id": fic_id,
        "name": name,
        "short_description": short_description,
        "domain": domain,
        "topic": topic,
        "topic_extension": is_new_topic,
        "version": "v2",
        "source_meta": source_meta,
        "selection_hints": selection_hints,
        "semantic_hints": semantic_hints,
        "inputs": inputs,
        "output": output,
        "execution": execution,
        "diagnostics": {
            "invariants": invariants,
            "scale_checks": scale_checks,
        },
        # Keep v1 compatibility for existing downstream consumers.
        "diagnostic_checks": checks,
    }


def generate_core(
    *,
    client: Any,
    model: str,
    defn: ConversionInput,
    allow_new_topic: bool,
) -> Dict[str, Any]:
    if not defn.python_solution.strip():
        raise ValueError("Record missing python_solution")
    schema = stage_core_schema()
    if schema is None:
        raise RuntimeError("Stage core schema is unavailable because google.genai is missing.")

    fallback_id = f"{defn.source_meta.get('function_id') or 'unknown'}_{defn.source_meta.get('question_id') or 'unknown'}"
    base_prompt = build_stage_core_prompt(defn)
    corrective_suffix = ""
    last_error: Optional[Exception] = None

    # Some model responses still violate compute(inputs) despite schema constraints.
    # Retry with progressively stricter corrective hints before failing hard.
    for attempt in range(1, 4):
        prompt = f"{base_prompt}\n{corrective_suffix}" if corrective_suffix else base_prompt
        raw = call_gemini_json(
            client,
            model=model,
            prompt=prompt,
            schema=schema,
        )
        try:
            return formalize_core_payload(
                raw,
                source_meta=defn.source_meta,
                fallback_id=fallback_id,
                allow_new_topic=allow_new_topic,
            )
        except ValueError as err:
            last_error = err
            corrective_suffix = (
                "CRITICAL CORRECTION:\n"
                f"- Previous output failed validation: {err}\n"
                "- Ensure domain is from taxonomy.\n"
                "- Ensure topic is non-empty snake_case.\n"
                "- Ensure execution.code contains exactly def compute(inputs):\n"
                "- Ensure diagnostics checks have valid expression and predicate_mode.\n"
                f"- This is retry #{attempt} with stricter validation requirements."
            )

    raise ValueError(
        f"Stage core failed after retries for record {fallback_id}: {last_error}"
    )


def default_topic_for_domain(domain: str) -> Optional[str]:
    topics = TAXONOMY.get(domain, [])
    if not topics:
        return None
    return topics[0]
