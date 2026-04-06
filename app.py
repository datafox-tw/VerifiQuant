from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template, request

from verifiquant.card_store import SQLAlchemyArtifactStore
from verifiquant.pipeline import run_error_classification_pipeline as diag


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_URL = os.environ.get(
    "VERIFIQUANT_DB_URL",
    "sqlite:///verifiquant/data/runs/demo_v2/cards.db",
)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _to_context_dict(context: str) -> Dict[str, Any]:
    raw = _clean_text(context)
    if not raw:
        return {}

    parsed: Dict[str, Any] = {}
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            for k, v in obj.items():
                parsed[str(k).strip()] = v
            return parsed
    except Exception:
        pass

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]\s*(.+)\s*$", line)
        if m:
            parsed[m.group(1)] = m.group(2).strip()
            continue
        if "," in line:
            for part in line.split(","):
                p = part.strip()
                m2 = re.match(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]\s*(.+)\s*$", p)
                if m2:
                    parsed[m2.group(1)] = m2.group(2).strip()
    return parsed


def _extract_inputs(
    *,
    inputs_spec: List[Dict[str, Any]],
    question: str,
    context: str,
) -> Tuple[Dict[str, Any], List[str]]:
    parsed_context = _to_context_dict(context)
    combined_text = f"{question}\n{context}"
    provided_inputs: Dict[str, Any] = {}
    missing_required: List[str] = []

    for item in inputs_spec:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        aliases = [str(x).strip() for x in item.get("aliases", []) if str(x).strip()]
        keys = [name] + aliases

        raw_value: Optional[Any] = None
        for key in keys:
            if key in parsed_context:
                raw_value = parsed_context[key]
                break

        if raw_value is None:
            for key in keys:
                pattern = rf"\b{re.escape(key)}\b\s*[:=]\s*([^\n,;]+)"
                match = re.search(pattern, combined_text, flags=re.IGNORECASE)
                if match:
                    raw_value = match.group(1).strip()
                    break

        parsed = None
        if raw_value is not None:
            parsed = diag._parse_typed_value(raw_value, str(item.get("type", "number")))

        if parsed is None and bool(item.get("required", True)):
            missing_required.append(name)
        elif parsed is not None:
            provided_inputs[name] = parsed

    # Heuristic fallback for narrative context where slots are not explicit.
    # Example: effective duration story text with "current price", "decrease", "increase", and yield shift.
    if any(str(i.get("name")) == "p0" for i in inputs_spec):
        p0_match = re.search(
            r"(?:current market price|original price)[^$\d\-]*\$?\s*([0-9]+(?:\.[0-9]+)?)",
            combined_text,
            flags=re.IGNORECASE,
        )
        dec_clause = re.search(
            r"(yield\s+decreases?.{0,140})",
            combined_text,
            flags=re.IGNORECASE,
        )
        inc_clause = re.search(
            r"(yield\s+increases?.{0,140})",
            combined_text,
            flags=re.IGNORECASE,
        )
        p1_raw: Optional[str] = None
        p2_raw: Optional[str] = None
        if dec_clause:
            dec_prices = re.findall(r"\$?\s*([0-9]+(?:\.[0-9]+)?)", dec_clause.group(1))
            for token in dec_prices:
                # Skip yield-like percent scales for price slot.
                try:
                    if float(token) <= 1.0:
                        continue
                except Exception:
                    continue
                p1_raw = token
                break
        if inc_clause:
            inc_prices = re.findall(r"\$?\s*([0-9]+(?:\.[0-9]+)?)", inc_clause.group(1))
            for token in inc_prices:
                try:
                    if float(token) <= 1.0:
                        continue
                except Exception:
                    continue
                p2_raw = token
                break
        if p1_raw is None:
            p1_match = re.search(
                r"(?:rise to|increase to|up to)[^$\d\-]*\$?\s*([0-9]+(?:\.[0-9]+)?)",
                combined_text,
                flags=re.IGNORECASE,
            )
            p1_raw = p1_match.group(1) if p1_match else None
        if p2_raw is None:
            p2_match = re.search(
                r"(?:drop to|fall to|decrease to|down to)[^$\d\-]*\$?\s*([0-9]+(?:\.[0-9]+)?)",
                combined_text,
                flags=re.IGNORECASE,
            )
            p2_raw = p2_match.group(1) if p2_match else None
        y_match = re.search(
            r"(?:yield[^0-9\-+]{0,40}(?:decrease|increase)[^0-9\-+]{0,40})([0-9]+(?:\.[0-9]+)?\s*%)",
            combined_text,
            flags=re.IGNORECASE,
        ) or re.search(
            r"([0-9]+(?:\.[0-9]+)?\s*%)",
            combined_text,
            flags=re.IGNORECASE,
        )

        heuristics = {
            "p0": p0_match.group(1) if p0_match else None,
            "p1": p1_raw,
            "p2": p2_raw,
            "y": y_match.group(1) if y_match else None,
        }
        type_by_name = {str(i.get("name")): str(i.get("type", "number")) for i in inputs_spec}
        for key, raw in heuristics.items():
            if key in provided_inputs or raw is None:
                continue
            parsed = diag._parse_typed_value(raw, type_by_name.get(key, "number"))
            if parsed is not None:
                provided_inputs[key] = parsed

        # recompute missing after heuristic fill
        required_names = [str(i.get("name")) for i in inputs_spec if bool(i.get("required", True))]
        missing_required = [n for n in required_names if n not in provided_inputs]

    return provided_inputs, missing_required


def _build_repaired_context(original_context: str, updates: Dict[str, Any]) -> str:
    merged = _to_context_dict(original_context)
    for k, v in updates.items():
        if _clean_text(k):
            merged[str(k).strip()] = v
    if not merged:
        return original_context
    return json.dumps(merged, ensure_ascii=False, indent=2)


def _should_trigger_i_gate(question: str, context: str) -> bool:
    q = _clean_text(question).lower()
    has_ambiguity_signal = any(
        token in q
        for token in [" or ", "either", "which one", "which basis", "which direction", "或", "還是", "哪一種", "方向"]
    )
    context_slots = len(_to_context_dict(context))
    return has_ambiguity_signal and context_slots <= 1


def _format_output_for_question(question: str, value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    q = _clean_text(question).lower()
    if "two decimal" in q or "2 decimal" in q or "小數點後兩位" in q or "兩位小數" in q:
        return round(float(value), 2)
    return value


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=os.path.join(ROOT_DIR, "templates"),
        static_folder=os.path.join(ROOT_DIR, "static"),
    )
    store = SQLAlchemyArtifactStore(DEFAULT_DB_URL)
    core_by_id = store.load_core_by_id()
    repair_index = store.build_repair_index()

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/api/cards/overview")
    def cards_overview() -> Any:
        rows = store.load_retrieval_cards()
        grouped: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            domain = _clean_text(row.get("domain")) or "unknown"
            topic = _clean_text(row.get("topic")) or "unknown"
            domain_entry = grouped.setdefault(domain, {"total": 0, "topics": {}})
            topic_entry = domain_entry["topics"].setdefault(topic, {"count": 0, "cards": []})
            topic_entry["count"] += 1
            domain_entry["total"] += 1
            topic_entry["cards"].append(
                {
                    "fic_id": row.get("fic_id"),
                    "title": row.get("title"),
                    "summary": row.get("summary"),
                    "keywords": row.get("keywords", []),
                }
            )
        return jsonify(
            {
                "status": "ok",
                "domain_count": len(grouped),
                "card_count": len(rows),
                "grouped": grouped,
            }
        )

    @app.post("/api/repair/compose")
    def compose_repair() -> Any:
        payload = request.get_json(silent=True) or {}
        question = _clean_text(payload.get("question"))
        context = str(payload.get("context") or "")
        updates = payload.get("updates") or {}
        if not isinstance(updates, dict):
            return jsonify({"status": "error", "message": "updates must be an object"}), 400
        repaired_context = _build_repaired_context(context, updates)
        return jsonify(
            {
                "status": "ok",
                "repaired_question": question,
                "repaired_context": repaired_context,
            }
        )

    @app.post("/api/diagnose")
    def diagnose() -> Any:
        payload = request.get_json(silent=True) or {}
        question = _clean_text(payload.get("question"))
        context = str(payload.get("context") or "")
        domain = _clean_text(payload.get("domain")) or None
        topic = _clean_text(payload.get("topic")) or None

        if not question:
            return jsonify({"status": "error", "message": "question is required"}), 400

        candidates = store.retrieve_candidates(
            query=f"{question}\n{context}",
            top_k=5,
            domain=domain,
            topic=topic,
        )
        if not candidates:
            return jsonify(
                {
                    "status": "refusal",
                    "diagnostic_type": "N",
                    "funnel_layer": "Scope",
                    "gate_action": "graceful_exit",
                    "reason": "No candidate cards retrieved",
                    "repair_hints": [],
                    "prompt": {"question": question, "context": context},
                }
            )

        chosen = candidates[0]
        fic_id = str(chosen.get("fic_id", "")).strip()
        core = core_by_id.get(fic_id)
        if not core:
            return jsonify(
                {
                    "status": "refusal",
                    "diagnostic_type": "N",
                    "funnel_layer": "Scope",
                    "gate_action": "graceful_exit",
                    "reason": f"Chosen fic_id '{fic_id}' has no matching core card",
                    "repair_hints": [],
                    "prompt": {"question": question, "context": context},
                }
            )

        provided_inputs, missing_required = _extract_inputs(
            inputs_spec=core.get("inputs", []),
            question=question,
            context=context,
        )
        candidate_ids = [c.get("fic_id") for c in candidates]

        if missing_required:
            ask = [
                {"slot": name, "label": name, "type": "text", "required": True, "options": []}
                for name in missing_required
            ]
            return jsonify(
                {
                    "status": "error",
                    "diagnostic_type": "F",
                    "funnel_layer": "Schema",
                    "gate_action": "slot_filling",
                    "reason": "Missing or unparsable required inputs",
                    "fic_id": fic_id,
                    "candidate_ids": candidate_ids,
                    "provided_inputs": provided_inputs,
                    "requested_fields": missing_required,
                    "repair_hints": [
                        {
                            "rule_id": "local_missing_fields",
                            "title": "Missing required fields",
                            "user_message": "請補齊必要欄位，系統才能繼續執行。",
                            "repair_action": {"type": "request_missing_fields", "target": ",".join(missing_required)},
                            "allowed_next_steps": ["ask_followup", "rerun_same_fic"],
                            "ask_user_for": ask,
                        }
                    ],
                    "prompt": {"question": question, "context": context},
                }
            )

        checks = core.get("diagnostic_checks", [])
        e_checks = [chk for chk in checks if str(chk.get("diagnostic_type", "")).upper() == "E"]
        triggered: List[str] = []
        for chk in e_checks:
            is_triggered, eval_err = diag._is_check_triggered(chk, provided_inputs)
            if eval_err:
                return jsonify(
                    {
                        "status": "error",
                        "diagnostic_type": "C",
                        "funnel_layer": "Logic",
                        "gate_action": "audit_log",
                        "reason": f"E-check evaluation error: {eval_err}",
                        "fic_id": fic_id,
                        "candidate_ids": candidate_ids,
                        "provided_inputs": provided_inputs,
                        "repair_hints": [],
                        "prompt": {"question": question, "context": context},
                    }
                )
            if is_triggered:
                rid = _clean_text(chk.get("rule_id"))
                if rid:
                    triggered.append(rid)

        if triggered:
            repair_hints = diag._summarize_repairs(fic_id, triggered, repair_index)
            return jsonify(
                {
                    "status": "alert",
                    "diagnostic_type": "E",
                    "funnel_layer": "Boundary",
                    "gate_action": "deterministic_alert",
                    "reason": "Deterministic E-type checks detected potential inconsistencies.",
                    "fic_id": fic_id,
                    "candidate_ids": candidate_ids,
                    "provided_inputs": provided_inputs,
                    "triggered_rule_ids": triggered,
                    "repair_hints": repair_hints,
                    "prompt": {"question": question, "context": context},
                }
            )

        if _should_trigger_i_gate(question, context):
            i_repairs = diag._summarize_repairs(fic_id, ["global_i_semantic_ambiguity"], repair_index)
            if i_repairs:
                return jsonify(
                    {
                        "status": "needs_clarification",
                        "diagnostic_type": "I",
                        "funnel_layer": "Critic",
                        "gate_action": "critic_intervention",
                        "reason": "Potential semantic ambiguity detected. Please clarify before execution.",
                        "fic_id": fic_id,
                        "candidate_ids": candidate_ids,
                        "provided_inputs": provided_inputs,
                        "repair_hints": i_repairs,
                        "prompt": {"question": question, "context": context},
                    }
                )

        output_value, exec_err = diag._evaluate_execution(core, provided_inputs)
        if exec_err:
            return jsonify(
                {
                    "status": "error",
                    "diagnostic_type": "C",
                    "funnel_layer": "Logic",
                    "gate_action": "audit_log",
                    "reason": exec_err,
                    "fic_id": fic_id,
                    "candidate_ids": candidate_ids,
                    "provided_inputs": provided_inputs,
                    "repair_hints": [],
                    "prompt": {"question": question, "context": context},
                }
            )

        final_output = _format_output_for_question(question, output_value)
        return jsonify(
            {
                "status": "success",
                "diagnostic_type": "None",
                "funnel_layer": "Logic",
                "gate_action": "audit_log",
                "reason": "Execution completed",
                "fic_id": fic_id,
                "candidate_ids": candidate_ids,
                "provided_inputs": provided_inputs,
                "output_var": (core.get("output") or {}).get("name"),
                "output_value": final_output,
                "repair_hints": [],
                "prompt": {"question": question, "context": context},
            }
        )

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=int(os.environ.get("PORT", 6222)), debug=True)
