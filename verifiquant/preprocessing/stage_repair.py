from __future__ import annotations

import json
from typing import Any, Dict, List

from verifiquant.preprocessing.common import call_gemini_json, normalize_label

try:
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai_types = None


REPAIR_ACTION_TYPES = {
    "request_missing_fields",
    "confirm_assumption",
    "confirm_unit_conversion",
    "swap_suspected_fields",
    "select_alternative_fic",
    "rephrase_task_intent",
    "declare_scope_boundary",
    "present_clarification_options",
}

ACTION_ALIASES = {
    "manual_input_missing_fields": "request_missing_fields",
    "confirm_unit_conversion": "confirm_unit_conversion",
    "select_alternative_fic": "select_alternative_fic",
    "swap_suspected_fields": "swap_suspected_fields",
    "rephrase_task_intent": "rephrase_task_intent",
    "declare_scope_boundary": "declare_scope_boundary",
    "present_clarification_options": "present_clarification_options",
}

NEXT_STEPS = {
    "rerun_same_fic",
    "select_alternative_fic",
    "ask_followup",
    "stop_with_refusal",
}


STAGE_REPAIR_PROMPT = """Role:
You are a product workflow designer creating repair rules for diagnostic checks.

Task:
For each diagnostic check in `fic_core.diagnostic_checks`, generate a stable frontend repair rule.

Rules:
1) Each `rule_id` in diagnostic_checks must appear exactly once in output rules.
2) Keep actions deterministic and UI-friendly.
3) Focus on what to ask user and what next action to take.
4) Do not modify diagnostic_type/severity; they are inherited from core.
5) repair_action.type must be one of:
   request_missing_fields, confirm_assumption, confirm_unit_conversion,
   swap_suspected_fields, select_alternative_fic, rephrase_task_intent
6) allowed_next_steps must be from:
   rerun_same_fic, select_alternative_fic, ask_followup, stop_with_refusal

Input:
<CORE_JSON>
{core_json}
</CORE_JSON>

Output JSON shape:
{{
  "rules": [
    {{
      "rule_id": "...",
      "title": "...",
      "user_message": "...",
      "explanation": "...",
      "ask_user_for": [
        {{"slot": "...", "label": "...", "type": "text|number|enum|boolean", "required": true, "options": [{{"value":"...", "label":"..."}}]}}
      ],
      "repair_action": {{"type": "...", "target": "..."}},
      "allowed_next_steps": ["..."]
    }}
  ]
}}
"""


def stage_repair_schema() -> Any:
    if genai_types is None:
        return None
    option_schema = genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "value": genai_types.Schema(type=genai_types.Type.STRING),
            "label": genai_types.Schema(type=genai_types.Type.STRING),
        },
        required=["value", "label"],
    )

    ask_schema = genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "slot": genai_types.Schema(type=genai_types.Type.STRING),
            "label": genai_types.Schema(type=genai_types.Type.STRING),
            "type": genai_types.Schema(
                type=genai_types.Type.STRING,
                enum=["text", "number", "enum", "boolean"],
            ),
            "required": genai_types.Schema(type=genai_types.Type.BOOLEAN),
            "options": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=option_schema,
            ),
        },
        required=["slot", "label", "type", "required", "options"],
    )

    rule_schema = genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "rule_id": genai_types.Schema(type=genai_types.Type.STRING),
            "title": genai_types.Schema(type=genai_types.Type.STRING),
            "user_message": genai_types.Schema(type=genai_types.Type.STRING),
            "explanation": genai_types.Schema(type=genai_types.Type.STRING),
            "ask_user_for": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=ask_schema,
            ),
            "repair_action": genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "type": genai_types.Schema(type=genai_types.Type.STRING),
                    "target": genai_types.Schema(type=genai_types.Type.STRING),
                },
                required=["type", "target"],
            ),
            "allowed_next_steps": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
        },
        required=[
            "rule_id",
            "title",
            "user_message",
            "explanation",
            "ask_user_for",
            "repair_action",
            "allowed_next_steps",
        ],
    )

    return genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "rules": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=rule_schema,
            )
        },
        required=["rules"],
    )


def build_stage_repair_prompt(core: Dict[str, Any]) -> str:
    return STAGE_REPAIR_PROMPT.format(
        core_json=json.dumps(core, ensure_ascii=False, indent=2),
    )


def _normalize_action(raw_action: str, diagnostic_type: str) -> str:
    key = normalize_label(raw_action)
    mapped = ACTION_ALIASES.get(key, key)
    if mapped in REPAIR_ACTION_TYPES:
        return mapped
    if diagnostic_type == "F":
        return "request_missing_fields"
    if diagnostic_type == "N":
        return "declare_scope_boundary"
    if diagnostic_type == "I":
        return "present_clarification_options"
    if diagnostic_type == "M":
        return "rephrase_task_intent"
    return "confirm_assumption"


def _default_repair_for_check(core: Dict[str, Any], check: Dict[str, Any]) -> Dict[str, Any]:
    d_type = check.get("diagnostic_type", "E")
    action_type = _normalize_action("", d_type)
    return {
        "rule_id": check["rule_id"],
        "fic_id": core["fic_id"],
        "diagnostic_type": d_type,
        "severity": check.get("severity", "alert"),
        "title": check.get("name") or check["rule_id"],
        "user_message": check.get("description") or "Potential issue detected.",
        "explanation": check.get("description") or "Please review this input and confirm intended values.",
        "ask_user_for": [],
        "repair_action": {
            "type": action_type,
            "target": ",".join(check.get("applies_to", [])) or "inputs",
        },
        "allowed_next_steps": ["ask_followup", "rerun_same_fic"],
    }


def _normalize_ask_user_for(items: Any) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        slot = normalize_label(item.get("slot", ""))
        label = str(item.get("label", "")).strip()
        value_type = str(item.get("type", "")).strip().lower()
        required = bool(item.get("required", True))
        options = []
        for opt in item.get("options", []):
            if not isinstance(opt, dict):
                continue
            ov = str(opt.get("value", "")).strip()
            ol = str(opt.get("label", "")).strip()
            if ov and ol:
                options.append({"value": ov, "label": ol})

        if not slot or not label:
            continue
        if value_type not in {"text", "number", "enum", "boolean"}:
            value_type = "text"
        out.append(
            {
                "slot": slot,
                "label": label,
                "type": value_type,
                "required": required,
                "options": options,
            }
        )
    return out


def formalize_repair_payload(raw: Dict[str, Any], core: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks_by_id = {chk["rule_id"]: chk for chk in core.get("diagnostic_checks", [])}
    raw_rules = raw.get("rules", []) if isinstance(raw, dict) else []

    normalized: Dict[str, Dict[str, Any]] = {}
    for item in raw_rules:
        if not isinstance(item, dict):
            continue
        rid = normalize_label(item.get("rule_id", ""))
        if rid not in checks_by_id:
            continue
        chk = checks_by_id[rid]
        action = item.get("repair_action", {}) or {}
        action_type = _normalize_action(action.get("type", ""), chk.get("diagnostic_type", "E"))
        next_steps = [str(x).strip() for x in item.get("allowed_next_steps", []) if str(x).strip() in NEXT_STEPS]
        if not next_steps:
            next_steps = ["ask_followup", "rerun_same_fic"]

        normalized[rid] = {
            "rule_id": rid,
            "fic_id": core["fic_id"],
            "diagnostic_type": chk.get("diagnostic_type", "E"),
            "severity": chk.get("severity", "alert"),
            "title": str(item.get("title", "")).strip() or chk.get("name") or rid,
            "user_message": str(item.get("user_message", "")).strip() or chk.get("description", ""),
            "explanation": str(item.get("explanation", "")).strip() or chk.get("description", ""),
            "ask_user_for": _normalize_ask_user_for(item.get("ask_user_for", [])),
            "repair_action": {
                "type": action_type,
                "target": str(action.get("target", "")).strip() or ",".join(chk.get("applies_to", [])) or "inputs",
            },
            "allowed_next_steps": next_steps,
        }

    # Every diagnostic rule must map to one stable repair rule.
    out: List[Dict[str, Any]] = []
    for rid, chk in checks_by_id.items():
        out.append(normalized.get(rid) or _default_repair_for_check(core, chk))
    out.extend(_global_v2_rules(core))
    return out


def _global_v2_rules(core: Dict[str, Any]) -> List[Dict[str, Any]]:
    selection_hints = [str(x).strip() for x in (core.get("selection_hints") or []) if str(x).strip()]
    semantic_hints = [x for x in (core.get("semantic_hints") or []) if isinstance(x, dict)]
    i_options: List[Dict[str, str]] = []
    for hint in semantic_hints:
        for opt in hint.get("options", []):
            text = str(opt).strip()
            if text:
                i_options.append({"value": text, "label": text})
    if not i_options:
        i_options = [
            {"value": "option_a", "label": "Option A"},
            {"value": "option_b", "label": "Option B"},
        ]

    return [
        {
            "rule_id": "global_n_not_supported",
            "fic_id": core["fic_id"],
            "diagnostic_type": "N",
            "severity": "error",
            "title": "Out-of-Scope Request",
            "user_message": "This request appears outside the currently supported formula scope.",
            "explanation": "The system can explain the current scope boundary and suggest supported alternatives.",
            "ask_user_for": [
                {
                    "slot": "target_metric",
                    "label": "Select a supported metric",
                    "type": "enum",
                    "required": True,
                    "options": [{"value": x, "label": x} for x in (selection_hints or ["npv", "irr", "payback_period"])],
                }
            ],
            "repair_action": {
                "type": "declare_scope_boundary",
                "target": "scope_boundary",
            },
            "allowed_next_steps": ["select_alternative_fic", "ask_followup", "stop_with_refusal"],
        },
        {
            "rule_id": "global_i_semantic_ambiguity",
            "fic_id": core["fic_id"],
            "diagnostic_type": "I",
            "severity": "alert",
            "title": "Semantic Ambiguity Clarification",
            "user_message": "A hidden ambiguity was detected (for example FX direction or time basis).",
            "explanation": "Ask a constrained clarification question before deterministic execution.",
            "ask_user_for": [
                {
                    "slot": "clarification_choice",
                    "label": "Clarification option",
                    "type": "enum",
                    "required": True,
                    "options": i_options,
                }
            ],
            "repair_action": {
                "type": "present_clarification_options",
                "target": "semantic_hints",
            },
            "allowed_next_steps": ["ask_followup", "rerun_same_fic"],
        },
    ]


def generate_repair_rules(*, client: Any, model: str, core: Dict[str, Any]) -> List[Dict[str, Any]]:
    schema = stage_repair_schema()
    if schema is None:
        raise RuntimeError("Stage repair schema is unavailable because google.genai is missing.")
    raw = call_gemini_json(
        client,
        model=model,
        prompt=build_stage_repair_prompt(core),
        schema=schema,
    )
    return formalize_repair_payload(raw, core)
