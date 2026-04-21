"""LangGraph refactor of the VerifiQuant diagnostic pipeline.

Replaces the monolithic run_case() with a typed StateGraph DAG.
Public interface: ErrorClassificationLG (mirrors ErrorClassificationAPI).

Graph topology:
  START
    │
  [retrieve] ──N──────────────────────────────────────────► [exit_n]
    │                                                             │
  [mn_select] ──M──► [exit_m]                                    │
    │          ──N──► [exit_n]                                   │
    │                                                            │
  [extract] ──F────► [exit_f]                                   │
    │                                                            │
  [fe_checks] ──F──► [exit_f]                                   │
    │           ──C──► [exit_c]                                  │
    │           ──E──► [exit_e]                                  │
    │                                                            │
  [i_gate] ──I─────► [exit_i]                                   │
    │                                                            │
  [execute] ──C────► [exit_c]                                   │
    │                    │                                       │
  [exit_success]   (all exit_* nodes converge)                  │
    │                         │                                  │
    └─────────────────────────▼──────────────────────────────────┘
                         [finalize] → END

HITL (v1 — stateless rerun):
  exit_m / exit_f / exit_e / exit_i → hitl_context.needs_hitl=True
  Caller merges clarification into question+context, reruns from START.
  exit_n (scope) / exit_c → hitl_context.needs_hitl=False
"""
from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Tuple

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from verifiquant.card_store import SQLAlchemyArtifactStore
from verifiquant.preprocessing.stage_repair import GLOBAL_N_NOT_SUPPORTED_RULE, GLOBAL_SCOPE_FIC_ID
from verifiquant.preprocessing.validate_relations import validate_artifact_relations

# Re-use all pure helpers from the existing pipeline — no duplication.
from verifiquant.pipeline.run_error_classification_pipeline import (
    RetrievalCandidate,
    retrieve_candidates,
    retrieve_candidates_from_store,
    _select_card_with_llm,
    _extract_inputs_with_llm,
    coerce_input,
    _resolve_schema_scale,
    _is_check_triggered,
    _critic_check_with_llm,
    _evaluate_execution,
    _answer_match,
    _summarize_repairs,
    _best_scope_repair,
    _i_rule_id_from_hint_id,
    _semantic_hint_level,
    _build_soft_warnings,
    _infer_open_critic_i_level,
    _load_core,
    _load_retrieval,
    _load_repair,
    _build_repair_index,
)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_STRUCTURAL_ERROR_SIGNALS = (
    "is not subscriptable",
    "has no attribute",
    "is not iterable",
    "object is not callable",
    "cannot unpack",
    "must be",
)

_SAFE_BUILTINS_EXEC: Dict[str, Any] = {
    "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
    "all": all, "any": any, "isinstance": isinstance, "bool": bool,
    "str": str, "float": float, "int": int, "pow": pow, "round": round,
    "range": range, "list": list, "dict": dict, "tuple": tuple,
    "set": set, "zip": zip, "enumerate": enumerate,
    "ValueError": ValueError, "TypeError": TypeError, "Exception": Exception,
    "math": math,
}

# Global M repair rule (no FIC-specific rules exist for M-class)
_GLOBAL_M_RULE: Dict[str, Any] = {
    "rule_id": "global_m_clarify",
    "diagnostic_type": "M",
    "severity": "error",
    "title": "Intent Clarification Required",
    "user_message": "The task intent is ambiguous. Please specify which financial metric to calculate.",
    "explanation": "The system could not determine a unique financial formula from your question.",
    "ask_user_for": [
        {
            "slot": "target_metric",
            "label": "Which metric do you want to calculate? (e.g. NPV, IRR, Payback Period)",
            "type": "text",
            "required": True,
            "options": [],
        }
    ],
    "repair_action": {"type": "rephrase_task_intent", "target": "question"},
    "allowed_next_steps": ["ask_followup", "rerun_same_fic"],
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. STATE
# ══════════════════════════════════════════════════════════════════════════════

class PipelineState(TypedDict):
    """Single immutable-ish state flowing through all nodes.

    Each node returns only the keys it writes; LangGraph merges the partial
    update into the canonical state.
    pipeline_logs uses operator.add so every node appends without clobbering.
    """

    # ── Input ────────────────────────────────────────────────────────────────
    case_id: str
    question: str
    context: str
    domain: Optional[str]
    topic: Optional[str]
    gold_answer: Optional[Any]

    # ── node_retrieve ────────────────────────────────────────────────────────
    candidates: List[Any]               # List[RetrievalCandidate]
    selection_trace: Dict[str, Any]     # retrieval_candidates written first

    # ── node_mn_select ───────────────────────────────────────────────────────
    decision: str                       # "select_card" | "abstain_m" | "abstain_n"
    chosen_fic_id: str
    candidate_ids: List[str]
    ambiguity_tags: List[str]
    clarification_questions: List[str]
    support_gap_reason: str
    core: Optional[Dict[str, Any]]      # chosen FIC core card

    # ── node_extract ─────────────────────────────────────────────────────────
    provided_inputs: Dict[str, Any]
    missing: List[str]
    ambiguous_fields: List[str]
    schema_contradiction_fields: List[str]
    normalization_note: str
    coercion_notes: List[str]
    extraction_trace: Dict[str, Any]

    # ── node_fe_checks ───────────────────────────────────────────────────────
    auto_triggered_e: List[str]
    structural_errors: List[Dict[str, Any]]
    eval_errors: List[Dict[str, Any]]
    echeck_trace: Dict[str, Any]

    # ── node_i_gate ──────────────────────────────────────────────────────────
    needs_clarification: bool
    i_level: str                        # "hard" | "soft" | ""
    i_hard_triggered: List[str]         # hint_ids that caused I_hard block
    i_soft_triggered: List[str]
    soft_warnings: List[Dict[str, Any]]
    critic_raw: Dict[str, Any]          # full LLM critic response
    critic_trace: Dict[str, Any]

    # ── node_execute ─────────────────────────────────────────────────────────
    output_value: Optional[float]
    exec_err: Optional[str]
    exec_trace: Dict[str, Any]
    is_correct: Optional[bool]
    abs_error: Optional[float]
    scale_note: Optional[str]

    # ── Routing signal (set by processing nodes, consumed by routers) ────────
    exit_gate: Optional[str]            # "M"|"N"|"F"|"E"|"I"|"C"|"success"

    # ── Exit payload (populated by exit_* nodes, assembled by node_finalize) ─
    exit_reason: str
    repair_hints: List[Dict[str, Any]]
    clarification_request: Optional[Dict[str, Any]]

    # ── HITL context (v1 — stateless rerun) ─────────────────────────────────
    # Shape: {needs_hitl, hitl_type, user_message, ask_user_for,
    #         allowed_next_steps, rerun_strategy}
    hitl_context: Optional[Dict[str, Any]]

    # ── Accumulated debug logs (auto-appended) ───────────────────────────────
    pipeline_logs: Annotated[List[str], operator.add]

    # ── Final assembled output ───────────────────────────────────────────────
    result: Dict[str, Any]


# ══════════════════════════════════════════════════════════════════════════════
# 2. DEPS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineDeps:
    """External resources shared across all nodes. Build once, bind everywhere."""
    client: Any
    store: Optional[SQLAlchemyArtifactStore]
    core_by_id: Dict[str, Dict[str, Any]]
    retrieval_cards: List[Dict[str, Any]]
    repair_index: Dict[Tuple[str, str], Dict[str, Any]]

    selector_model: str = "gemini-2.5-flash"
    extractor_model: str = "gemini-2.5-flash"
    judge_model: str = "gemini-2.5-flash"
    top_k: int = 3
    m_min_top_score: float = 0.05


# ══════════════════════════════════════════════════════════════════════════════
# 3. PROCESSING NODES
# ══════════════════════════════════════════════════════════════════════════════

def node_retrieve(state: PipelineState, *, deps: PipelineDeps) -> dict:
    """RAG: retrieve top-k FIC candidate cards from store or flat list."""
    q, c = state["question"], state["context"]
    domain, topic = state.get("domain"), state.get("topic")

    if deps.store is not None:
        candidates = retrieve_candidates_from_store(
            deps.store, query=f"{q}\n{c}", top_k=deps.top_k,
            domain=domain, topic=topic,
        )
    else:
        candidates = retrieve_candidates(deps.retrieval_cards, f"{q}\n{c}", top_k=deps.top_k)

    top_rows = [
        {"fic_id": str(c.retrieval.get("fic_id", "")).strip(),
         "score": round(float(c.score), 6),
         "title": c.retrieval.get("title"),
         "topic": c.retrieval.get("topic"),
         "domain": c.retrieval.get("domain")}
        for c in candidates
    ]
    log = "[retrieve] " + (
        ", ".join(f"{r['fic_id']}@{r['score']:.3f}" for r in top_rows)
        if top_rows else "no candidates"
    )
    update: dict = {
        "candidates": candidates,
        "selection_trace": {"retrieval_candidates": top_rows},
        "pipeline_logs": [log],
    }
    if not candidates:
        update["exit_gate"] = "N"
        update["support_gap_reason"] = "no_candidate_cards"
        update["pipeline_logs"] = [log, "[retrieve] exit: N (no candidates)"]
    return update


def node_mn_select(state: PipelineState, *, deps: PipelineDeps) -> dict:
    """LLM selector: commit to one FIC card or abstain (M / N)."""
    q, c = state["question"], state["context"]
    candidates: List[RetrievalCandidate] = state["candidates"]

    sel = _select_card_with_llm(deps.client, deps.selector_model, q, c, candidates)
    decision = str(sel.get("decision", "")).strip()
    chosen_fic_id = str(sel.get("chosen_fic_id", "")).strip()
    candidate_ids = [str(cand.retrieval.get("fic_id", "")) for cand in candidates]
    support_gap_reason = str(sel.get("support_gap_reason", "")).strip()
    ambiguity_tags = [str(x).strip() for x in sel.get("ambiguity_tags", []) if str(x).strip()]
    clar_qs = [str(x).strip() for x in sel.get("clarification_questions", []) if str(x).strip()]

    selector_trace = {
        "decision": decision, "chosen_fic_id": chosen_fic_id,
        "reason": str(sel.get("reason", "")).strip(),
        "support_gap_reason": support_gap_reason,
        "ambiguity_tags": ambiguity_tags, "clarification_questions": clar_qs,
    }
    log = f"[mn_select] decision={decision} fic_id={chosen_fic_id or '<none>'}"
    update: dict = {
        "decision": decision, "chosen_fic_id": chosen_fic_id,
        "candidate_ids": candidate_ids, "support_gap_reason": support_gap_reason,
        "ambiguity_tags": ambiguity_tags, "clarification_questions": clar_qs,
        "selection_trace": {**state.get("selection_trace", {}), "selector": selector_trace},
        "pipeline_logs": [log],
    }

    if decision == "abstain_m":
        update["exit_gate"] = "M"
        update["pipeline_logs"] = [log, "[mn_select] exit: M (abstain_m)"]
        return update
    if decision == "abstain_n":
        update["exit_gate"] = "N"
        update["pipeline_logs"] = [log, "[mn_select] exit: N (abstain_n)"]
        return update
    if decision != "select_card" or not chosen_fic_id:
        update["exit_gate"] = "M"
        update["decision"] = "abstain_m"
        update["pipeline_logs"] = [log, "[mn_select] exit: M (no valid card committed)"]
        return update

    top_score = candidates[0].score if candidates else 0.0
    if top_score < deps.m_min_top_score:
        update["exit_gate"] = "N"
        update["support_gap_reason"] = "low_retrieval_confidence"
        update["pipeline_logs"] = [log, f"[mn_select] exit: N (low_confidence score={top_score:.3f})"]
        return update

    core = deps.core_by_id.get(chosen_fic_id)
    if core is None:
        update["exit_gate"] = "N"
        update["support_gap_reason"] = "missing_core_card"
        update["pipeline_logs"] = [log, f"[mn_select] exit: N (missing core for {chosen_fic_id})"]
        return update

    update["core"] = core
    return update


def node_extract(state: PipelineState, *, deps: PipelineDeps) -> dict:
    """LLM extractor: pull and coerce input values from question/context."""
    q, c = state["question"], state["context"]
    core: Dict[str, Any] = state["core"]  # guaranteed non-None here

    extraction = _extract_inputs_with_llm(deps.client, deps.extractor_model, q, c, core)
    normalization_note = str(extraction.get("normalization_note", "") or "")

    core_inputs = core.get("inputs", [])
    type_map  = {str(i.get("name")): str(i.get("type", "number"))  for i in core_inputs}
    unit_map  = {str(i.get("name")): i.get("unit")                  for i in core_inputs}
    desc_map  = {str(i.get("name")): str(i.get("description", "")) for i in core_inputs}
    required_names = [str(i.get("name")) for i in core_inputs if bool(i.get("required", True))]
    extracted = {str(x.get("name")): x for x in extraction.get("inputs", []) if x.get("name")}

    missing: List[str] = []
    provided_inputs: Dict[str, Any] = {}
    ambiguous_fields: List[str] = []
    schema_contradiction_fields: List[str] = []
    coercion_notes: List[str] = []

    def _bind(name: str, item: Dict[str, Any]) -> Optional[Any]:
        result = coerce_input(
            item.get("value", ""), type_map.get(name, "number"),
            unit=unit_map.get(name), description=desc_map.get(name, ""),
            source_scale=str(item.get("source_scale") or "inferred"),
        )
        if result.note:
            coercion_notes.append(f"[{name}] {result.note}")
        if result.confidence == "ambiguous":
            ambiguous_fields.append(name)
            schema_scale = _resolve_schema_scale(unit_map.get(name), desc_map.get(name, ""))
            if schema_scale == "percent_point" and str(item.get("source_scale") or "") == "explicit_decimal":
                schema_contradiction_fields.append(name)
        return result.value

    for name in required_names:
        item = extracted.get(name)
        if not item or item.get("status") != "provided":
            missing.append(name)
            continue
        val = _bind(name, item)
        if val is None:
            missing.append(name)
        else:
            provided_inputs[name] = val

    for name, item in extracted.items():
        if name in provided_inputs:
            continue
        val = _bind(name, item)
        if val is not None:
            provided_inputs[name] = val

    log = f"[extract] required={required_names} missing={missing} provided={sorted(provided_inputs)}"
    update: dict = {
        "provided_inputs": provided_inputs, "missing": missing,
        "ambiguous_fields": ambiguous_fields,
        "schema_contradiction_fields": schema_contradiction_fields,
        "normalization_note": normalization_note, "coercion_notes": coercion_notes,
        "extraction_trace": {
            "required_names": required_names,
            "llm_extracted_inputs": extraction.get("inputs", []),
            "normalization_note": normalization_note,
            "missing_fields": missing,
            "provided_keys": sorted(provided_inputs.keys()),
            "ambiguous_fields": ambiguous_fields,
            "schema_contradiction_fields": schema_contradiction_fields,
            "coercion_notes": coercion_notes,
        },
        "pipeline_logs": [log],
    }
    if missing:
        update["exit_gate"] = "F"
        update["pipeline_logs"] = [log, f"[extract] exit: F (missing {missing})"]
    return update


def node_fe_checks(state: PipelineState, *, deps: PipelineDeps) -> dict:
    """Deterministic F/E boundary checks against extracted inputs."""
    core: Dict[str, Any] = state["core"]
    provided_inputs = state["provided_inputs"]

    # Pre-compile FIC compute fn for E-checks that call compute({...})
    fic_compute_fn: Optional[Any] = None
    fic_code = str((core.get("execution") or {}).get("code", ""))
    if "def compute(inputs)" in fic_code:
        try:
            _env: Dict[str, Any] = {}
            exec(fic_code, {"__builtins__": _SAFE_BUILTINS_EXEC}, _env)  # noqa: S102
            fic_compute_fn = _env.get("compute")
        except Exception:
            pass  # surface at C-execution stage

    checks = core.get("diagnostic_checks", [])
    e_checks = [chk for chk in checks if str(chk.get("diagnostic_type", "")).upper() == "E"]
    auto_triggered: List[str] = []
    eval_errors: List[Dict[str, Any]] = []
    structural_errors: List[Dict[str, Any]] = []
    logs: List[str] = []

    for chk in e_checks:
        rule_id = str(chk.get("rule_id", "")).strip()
        ctype = str(chk.get("check_type", "")).strip().lower()
        if ctype not in {"deterministic", "normalization"}:
            continue
        is_triggered, eval_err = _is_check_triggered(chk, provided_inputs, compute_fn=fic_compute_fn)
        if eval_err:
            rec = {"rule_id": rule_id or "<unknown>",
                   "expression": str(chk.get("expression", "")), "error": eval_err}
            if any(sig in eval_err for sig in _STRUCTURAL_ERROR_SIGNALS):
                structural_errors.append(rec)
                logs.append(f"[fe_checks] {rule_id}: structural-F error={eval_err}")
            else:
                eval_errors.append(rec)
                logs.append(f"[fe_checks] {rule_id}: eval-C error={eval_err}")
            continue
        if is_triggered and rule_id:
            auto_triggered.append(rule_id)
            logs.append(f"[fe_checks] {rule_id}: E triggered")

    auto_triggered = list(dict.fromkeys(auto_triggered))
    echeck_trace = {
        "total_checks": len(checks), "e_checks_count": len(e_checks),
        "auto_triggered_rule_ids": auto_triggered,
        "structural_errors": structural_errors, "evaluation_errors": eval_errors,
    }
    update: dict = {
        "auto_triggered_e": auto_triggered,
        "structural_errors": structural_errors,
        "eval_errors": eval_errors,
        "echeck_trace": echeck_trace,
        "pipeline_logs": logs or ["[fe_checks] all checks passed"],
    }
    if structural_errors:
        update["exit_gate"] = "F"
    elif eval_errors:
        update["exit_gate"] = "C"
    elif auto_triggered:
        update["exit_gate"] = "E"
    return update


def node_i_gate(state: PipelineState, *, deps: PipelineDeps) -> dict:
    """Critic agent: detect hidden semantic ambiguity before execution."""
    q, c = state["question"], state["context"]
    core: Dict[str, Any] = state["core"]
    provided_inputs = state["provided_inputs"]
    semantic_hints = [h for h in (core.get("semantic_hints") or []) if isinstance(h, dict)]

    critic = _critic_check_with_llm(
        client=deps.client, model=deps.judge_model,
        question=q, context=c,
        semantic_hints=semantic_hints, provided_inputs=provided_inputs,
    )
    needs_clarification = bool(critic.get("needs_clarification"))

    if not needs_clarification:
        return {
            "needs_clarification": False, "i_level": "",
            "i_hard_triggered": [], "i_soft_triggered": [],
            "soft_warnings": [], "critic_raw": critic,
            "critic_trace": {"semantic_hints_count": len(semantic_hints), "raw_critic": critic},
            "pipeline_logs": ["[i_gate] no clarification needed"],
        }

    triggered_hint_ids = [str(x).strip() for x in critic.get("triggered_hint_ids", []) if str(x).strip()]
    hint_by_id = {str(h.get("id", "")).strip(): h for h in semantic_hints if str(h.get("id", "")).strip()}
    hard_triggered = [hid for hid in triggered_hint_ids if _semantic_hint_level(hint_by_id.get(hid, {})) == "hard"]
    soft_triggered = [hid for hid in triggered_hint_ids if _semantic_hint_level(hint_by_id.get(hid, {})) == "soft"]
    open_inferred_level = _infer_open_critic_i_level(critic) if not triggered_hint_ids else None
    is_hard_block = bool(hard_triggered) or (not triggered_hint_ids and open_inferred_level == "hard")

    log = (f"[i_gate] needs_clarification=True hard={hard_triggered} "
           f"soft={soft_triggered} open_level={open_inferred_level} hard_block={is_hard_block}")

    update: dict = {
        "needs_clarification": True,
        "i_hard_triggered": hard_triggered,
        "i_soft_triggered": soft_triggered,
        "critic_raw": critic,
        "critic_trace": {"semantic_hints_count": len(semantic_hints), "raw_critic": critic},
        "pipeline_logs": [log],
    }

    if is_hard_block:
        update["exit_gate"] = "I"
        update["i_level"] = "hard"
        update["soft_warnings"] = []
    else:
        # I_soft: attach warnings but continue to execute
        update["i_level"] = "soft"
        update["soft_warnings"] = _build_soft_warnings(
            critic=critic, semantic_hints=semantic_hints,
            triggered_hint_ids=soft_triggered if soft_triggered else [],
        )
    return update


def node_execute(state: PipelineState, *, deps: PipelineDeps) -> dict:
    """Deterministic Python execution via FIC compute(inputs)."""
    core: Dict[str, Any] = state["core"]
    provided_inputs = state["provided_inputs"]

    output_value, exec_err, exec_trace = _evaluate_execution(core, provided_inputs)
    exec_trace["fic_id"] = state.get("chosen_fic_id")
    exec_trace["fic_version"] = core.get("version", "v1")

    if exec_err:
        return {
            "output_value": None, "exec_err": exec_err, "exec_trace": exec_trace,
            "is_correct": None, "abs_error": None, "scale_note": None,
            "exit_gate": "C",
            "pipeline_logs": [f"[execute] exit: C (exec_err={exec_err})"],
        }

    gold_num = state.get("gold_answer")
    if not isinstance(gold_num, float):
        from verifiquant.pipeline.run_error_classification_pipeline import _parse_number
        gold_num = _parse_number(gold_num)

    abs_error, is_correct, scale_note = _answer_match(state["question"], output_value, gold_num)
    log = f"[execute] output={output_value} gold={gold_num} is_correct={is_correct}"
    return {
        "output_value": output_value, "exec_err": None, "exec_trace": exec_trace,
        "is_correct": is_correct, "abs_error": abs_error, "scale_note": scale_note,
        "exit_gate": "success",
        "pipeline_logs": [log],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. EXIT NODES  — each prepares exit_reason + repair_hints + hitl_context
# ══════════════════════════════════════════════════════════════════════════════

def node_exit_m(state: PipelineState, *, deps: PipelineDeps) -> dict:
    """M-class: intent ambiguous before card commitment."""
    clar_qs = state.get("clarification_questions") or [
        "Please specify the target metric (e.g. NPV, IRR, Payback Period)."
    ]
    reason = (state.get("selection_trace", {}).get("selector", {}).get("reason")
              or "Ambiguous task intent before card commitment")
    return {
        "exit_reason": reason,
        "repair_hints": [_GLOBAL_M_RULE],
        "clarification_request": {"questions": clar_qs, "options": []},
        "hitl_context": {
            "needs_hitl": True,
            "hitl_type": "M_clarify",
            "user_message": "Please clarify which financial metric you want to calculate.",
            "ask_user_for": _GLOBAL_M_RULE["ask_user_for"],
            "allowed_next_steps": ["ask_followup", "rerun_same_fic"],
            "rerun_strategy": "append_to_context",
        },
        "pipeline_logs": ["[exit_m] M-class exit prepared"],
    }


def node_exit_n(state: PipelineState, *, deps: PipelineDeps) -> dict:
    """N-class: intent clear but no matching formula in library."""
    support_gap = state.get("support_gap_reason") or "unsupported_formula_family"
    candidate_ids = state.get("candidate_ids", [])
    repair_hints = _best_scope_repair(candidate_ids, deps.repair_index)

    # low_retrieval_confidence: user might reframe query; scope gap: user cannot fix
    needs_hitl = support_gap == "low_retrieval_confidence"
    return {
        "exit_reason": f"Request out of supported scope (support_gap={support_gap})",
        "repair_hints": repair_hints,
        "clarification_request": None,
        "hitl_context": {
            "needs_hitl": needs_hitl,
            "hitl_type": "N_scope",
            "user_message": (
                "Retrieval confidence was low. Try rephrasing or adding more context."
                if needs_hitl else
                "This request is outside the currently supported formula scope."
            ),
            "ask_user_for": [],
            "allowed_next_steps": ["select_alternative_fic", "ask_followup", "stop_with_refusal"],
            "rerun_strategy": "append_to_context" if needs_hitl else "none",
        },
        "pipeline_logs": [f"[exit_n] N-class exit (support_gap={support_gap})"],
    }


def node_exit_f(state: PipelineState, *, deps: PipelineDeps) -> dict:
    """F-class: missing / structurally mismatched required inputs."""
    chosen_fic_id = state.get("chosen_fic_id", "")
    missing = state.get("missing", [])
    structural_errors = state.get("structural_errors", [])

    if structural_errors:
        first = structural_errors[0]
        exit_reason = (
            f"Input structure mismatch in rule {first['rule_id']}: {first['error']}."
        )
        rule_ids = [e["rule_id"] for e in structural_errors]
        repair_hints = _summarize_repairs(chosen_fic_id, rule_ids, deps.repair_index)
        ask_user_for: List[Dict] = []
        user_msg = f"Input shape mismatch detected: {first['error']}"
    else:
        exit_reason = f"Missing or unparsable required inputs: {missing}"
        repair_hints = [
            {
                "repair_action": {"type": "request_missing_fields", "target": ",".join(missing)},
                "ask_user_for": [
                    {"slot": m, "label": m, "type": "text", "required": True, "options": []}
                    for m in missing
                ],
                "allowed_next_steps": ["ask_followup", "rerun_same_fic"],
            }
        ]
        ask_user_for = [
            {"slot": m, "label": f"Please provide a value for: {m}", "type": "text",
             "required": True, "options": []}
            for m in missing
        ]
        user_msg = f"The following required fields are missing: {', '.join(missing)}"

    return {
        "exit_reason": exit_reason,
        "repair_hints": repair_hints,
        "clarification_request": None,
        "hitl_context": {
            "needs_hitl": True,
            "hitl_type": "F_slot_fill",
            "user_message": user_msg,
            "ask_user_for": ask_user_for,
            "allowed_next_steps": ["ask_followup", "rerun_same_fic"],
            "rerun_strategy": "append_to_context",
        },
        "pipeline_logs": [f"[exit_f] F-class exit (missing={missing} structural={len(structural_errors)})"],
    }


def node_exit_e(state: PipelineState, *, deps: PipelineDeps) -> dict:
    """E-class: deterministic boundary violation."""
    chosen_fic_id = state.get("chosen_fic_id", "")
    triggered = state.get("auto_triggered_e", [])
    repair_hints = _summarize_repairs(chosen_fic_id, triggered, deps.repair_index)

    user_msgs = [h.get("user_message", "") for h in repair_hints if h.get("user_message")]
    ask_user_for: List[Dict] = []
    for h in repair_hints:
        ask_user_for.extend(h.get("ask_user_for", []))

    return {
        "exit_reason": "Deterministic E-type checks detected potential inconsistencies.",
        "repair_hints": repair_hints,
        "clarification_request": None,
        "hitl_context": {
            "needs_hitl": True,
            "hitl_type": "E_alert",
            "user_message": " ".join(user_msgs) or "Please review and correct the flagged input values.",
            "ask_user_for": ask_user_for,
            "allowed_next_steps": ["ask_followup", "rerun_same_fic"],
            "rerun_strategy": "append_to_context",
        },
        "pipeline_logs": [f"[exit_e] E-class exit (triggered={triggered})"],
    }


def node_exit_i(state: PipelineState, *, deps: PipelineDeps) -> dict:
    """I-class (hard): hidden semantic ambiguity blocks execution."""
    chosen_fic_id = state.get("chosen_fic_id", "")
    core = state.get("core") or {}
    semantic_hints = [h for h in (core.get("semantic_hints") or []) if isinstance(h, dict)]
    critic = state.get("critic_raw", {})
    hard_triggered = state.get("i_hard_triggered", [])

    # Clarification questions and options from critic / semantic hints
    clar_qs = [str(x).strip() for x in critic.get("clarification_questions", []) if str(x).strip()]
    model_options = [str(x).strip() for x in critic.get("clarification_options", []) if str(x).strip()]
    hint_by_id = {str(h.get("id", "")).strip(): h for h in semantic_hints if str(h.get("id", "")).strip()}
    options: List[str] = []
    if semantic_hints:
        for hint in semantic_hints:
            if not hard_triggered or str(hint.get("id", "")) in hard_triggered:
                options.extend([str(x).strip() for x in hint.get("options", []) if str(x).strip()])
    else:
        options.extend(model_options)
    options = list(dict.fromkeys(options))

    # Build repair hints from I-class rules
    i_rule_ids = [_i_rule_id_from_hint_id(hid, _semantic_hint_level(hint_by_id.get(hid, {})))
                  for hid in hard_triggered]
    if not i_rule_ids:
        i_rule_ids = ["global_i_semantic_ambiguity"]
    repair_hints = _summarize_repairs(chosen_fic_id, i_rule_ids, deps.repair_index)
    if not repair_hints:
        repair_hints = _summarize_repairs(chosen_fic_id, ["global_i_semantic_ambiguity"], deps.repair_index)

    ask_user_for: List[Dict] = []
    for h in repair_hints:
        ask_user_for.extend(h.get("ask_user_for", []))
    if not ask_user_for and options:
        ask_user_for = [
            {"slot": "clarification_choice", "label": clar_qs[0] if clar_qs else "Please clarify.",
             "type": "enum", "required": True,
             "options": [{"value": o, "label": o} for o in options]}
        ]

    return {
        "exit_reason": str(critic.get("reason", "")).strip() or "Hidden semantic ambiguity detected.",
        "repair_hints": repair_hints,
        "clarification_request": {
            "questions": clar_qs or ["Please confirm the intended financial interpretation."],
            "options": options,
            "triggered_hint_ids": hard_triggered,
            "mode": "hint_oriented" if semantic_hints else "open",
        },
        "hitl_context": {
            "needs_hitl": True,
            "hitl_type": "I_ambiguity",
            "user_message": clar_qs[0] if clar_qs else "Please clarify the intended interpretation.",
            "ask_user_for": ask_user_for,
            "allowed_next_steps": ["ask_followup", "rerun_same_fic"],
            "rerun_strategy": "append_to_context",
        },
        "pipeline_logs": [f"[exit_i] I-class exit (hard_triggered={hard_triggered})"],
    }


def node_exit_c(state: PipelineState, *, deps: PipelineDeps) -> dict:
    """C-class: Python execution error or E-check eval error. System-side fault."""
    eval_errors = state.get("eval_errors", [])
    exec_err = state.get("exec_err", "")
    reason = exec_err or (eval_errors[0]["error"] if eval_errors else "Unknown execution error")
    return {
        "exit_reason": reason,
        "repair_hints": [],           # C is a system error — user cannot self-repair
        "clarification_request": None,
        "hitl_context": {
            "needs_hitl": False,
            "hitl_type": "C_logic",
            "user_message": "An internal calculation error occurred. The FIC card may need to be updated.",
            "ask_user_for": [],
            "allowed_next_steps": ["stop_with_refusal"],
            "rerun_strategy": "none",
        },
        "pipeline_logs": [f"[exit_c] C-class exit (reason={reason[:120]})"],
    }


def node_exit_success(state: PipelineState, *, deps: PipelineDeps) -> dict:
    """Success path: all gates passed, output is valid."""
    return {
        "exit_reason": "Execution completed",
        "repair_hints": [],
        "clarification_request": None,
        "hitl_context": {"needs_hitl": False, "hitl_type": None,
                          "user_message": "", "ask_user_for": [],
                          "allowed_next_steps": [], "rerun_strategy": "none"},
        "pipeline_logs": [f"[exit_success] output={state.get('output_value')} is_correct={state.get('is_correct')}"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. FINALIZE NODE
# ══════════════════════════════════════════════════════════════════════════════

_FUNNEL_META: Dict[str, Tuple[str, str, str]] = {
    "M":       ("refusal",             "Intent",   "refusal"),
    "N":       ("refusal",             "Scope",    "graceful_exit"),
    "F":       ("error",               "Schema",   "slot_filling"),
    "E":       ("alert",               "Boundary", "deterministic_alert"),
    "I":       ("needs_clarification", "Critic",   "critic_intervention"),
    "C":       ("error",               "Logic",    "audit_log"),
    "success": ("success",             "Logic",    "audit_log"),
}


def _build_result(state: PipelineState) -> Dict[str, Any]:
    """Assemble the final result dict from state. Single switch on exit_gate."""
    gate = state.get("exit_gate") or "C"
    status, funnel_layer, gate_action = _FUNNEL_META.get(gate, ("error", "Unknown", "audit_log"))

    result: Dict[str, Any] = {
        "case_id":               state.get("case_id", ""),
        "status":                status,
        "diagnostic_type":       gate if gate != "success" else "None",
        "funnel_layer":          funnel_layer,
        "gate_action":           gate_action,
        "reason":                state.get("exit_reason", ""),
        "fic_id":                state.get("chosen_fic_id") or None,
        "candidate_ids":         state.get("candidate_ids", []),
        "support_gap_reason":    state.get("support_gap_reason") or None,
        "ambiguity_tags":        state.get("ambiguity_tags", []),
        "clarification_request": state.get("clarification_request"),
        "repair_hints":          state.get("repair_hints", []),
        "hitl_context":          state.get("hitl_context"),
        "pipeline_logs":         state.get("pipeline_logs", []),
        # backward-compat traces
        "selection_trace":       state.get("selection_trace", {}),
        "extraction_trace":      state.get("extraction_trace", {}),
        "echeck_trace":          state.get("echeck_trace", {}),
        "critic_trace":          state.get("critic_trace", {}),
        "execution_trace":       state.get("exec_trace", {}),
    }

    if gate in ("F", "E", "I", "C", "success"):
        result["fic_id"]             = state.get("chosen_fic_id")
        result["provided_inputs"]    = state.get("provided_inputs", {})
        result["normalization_note"] = state.get("normalization_note", "")

    if gate == "success":
        result["output_var"]    = (state.get("core") or {}).get("output", {}).get("name")
        result["output_value"]  = state.get("output_value")
        result["gold_answer"]   = state.get("gold_answer")
        result["is_correct"]    = state.get("is_correct")
        result["abs_error"]     = state.get("abs_error")
        result["has_i_soft"]    = state.get("i_level") == "soft"
        result["soft_warnings"] = state.get("soft_warnings", [])
        if state.get("scale_note"):
            result["output_scale_note"] = state["scale_note"]

    if gate == "C":
        result["exec_err"] = state.get("exec_err") or state.get("exit_reason")

    return result


def node_finalize(state: PipelineState, *, deps: PipelineDeps) -> dict:
    return {"result": _build_result(state)}


# ══════════════════════════════════════════════════════════════════════════════
# 6. ROUTERS
# ══════════════════════════════════════════════════════════════════════════════

def _route_retrieve(state: PipelineState) -> str:
    return "exit_n" if state.get("exit_gate") else "mn_select"

def _route_mn_select(state: PipelineState) -> str:
    gate = state.get("exit_gate")
    return {"M": "exit_m", "N": "exit_n"}.get(gate, "extract")  # type: ignore[return-value]

def _route_extract(state: PipelineState) -> str:
    return "exit_f" if state.get("exit_gate") == "F" else "fe_checks"

def _route_fe_checks(state: PipelineState) -> str:
    return {"F": "exit_f", "C": "exit_c", "E": "exit_e"}.get(
        state.get("exit_gate", ""), "i_gate"   # type: ignore[arg-type]
    )

def _route_i_gate(state: PipelineState) -> str:
    return "exit_i" if state.get("exit_gate") == "I" else "execute"

def _route_execute(state: PipelineState) -> str:
    return "exit_c" if state.get("exit_gate") == "C" else "exit_success"


# ══════════════════════════════════════════════════════════════════════════════
# 7. GRAPH FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline(deps: PipelineDeps):
    def _bind(fn):
        return partial(fn, deps=deps)

    g = StateGraph(PipelineState)

    # Processing nodes
    for name, fn in [
        ("retrieve",     node_retrieve),
        ("mn_select",    node_mn_select),
        ("extract",      node_extract),
        ("fe_checks",    node_fe_checks),
        ("i_gate",       node_i_gate),
        ("execute",      node_execute),
    ]:
        g.add_node(name, _bind(fn))

    # Exit nodes
    for name, fn in [
        ("exit_m",       node_exit_m),
        ("exit_n",       node_exit_n),
        ("exit_f",       node_exit_f),
        ("exit_e",       node_exit_e),
        ("exit_i",       node_exit_i),
        ("exit_c",       node_exit_c),
        ("exit_success", node_exit_success),
    ]:
        g.add_node(name, _bind(fn))

    g.add_node("finalize", _bind(node_finalize))

    # Wiring
    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve",  _route_retrieve,  {"mn_select": "mn_select", "exit_n": "exit_n"})
    g.add_conditional_edges("mn_select", _route_mn_select, {"extract": "extract", "exit_m": "exit_m", "exit_n": "exit_n"})
    g.add_conditional_edges("extract",   _route_extract,   {"fe_checks": "fe_checks", "exit_f": "exit_f"})
    g.add_conditional_edges("fe_checks", _route_fe_checks, {"i_gate": "i_gate", "exit_f": "exit_f", "exit_c": "exit_c", "exit_e": "exit_e"})
    g.add_conditional_edges("i_gate",    _route_i_gate,    {"execute": "execute", "exit_i": "exit_i"})
    g.add_conditional_edges("execute",   _route_execute,   {"exit_success": "exit_success", "exit_c": "exit_c"})

    for exit_node in ("exit_m", "exit_n", "exit_f", "exit_e", "exit_i", "exit_c", "exit_success"):
        g.add_edge(exit_node, "finalize")
    g.add_edge("finalize", END)

    return g.compile()


# ══════════════════════════════════════════════════════════════════════════════
# 8. PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

_EMPTY_STATE: PipelineState = {          # type: ignore[assignment]
    "case_id": "", "question": "", "context": "",
    "domain": None, "topic": None, "gold_answer": None,
    "candidates": [], "selection_trace": {},
    "decision": "", "chosen_fic_id": "", "candidate_ids": [],
    "ambiguity_tags": [], "clarification_questions": [], "support_gap_reason": "",
    "core": None, "provided_inputs": {}, "missing": [],
    "ambiguous_fields": [], "schema_contradiction_fields": [],
    "normalization_note": "", "coercion_notes": [], "extraction_trace": {},
    "auto_triggered_e": [], "structural_errors": [], "eval_errors": [], "echeck_trace": {},
    "needs_clarification": False, "i_level": "",
    "i_hard_triggered": [], "i_soft_triggered": [],
    "soft_warnings": [], "critic_raw": {}, "critic_trace": {},
    "output_value": None, "exec_err": None, "exec_trace": {},
    "is_correct": None, "abs_error": None, "scale_note": None,
    "exit_gate": None, "exit_reason": "", "repair_hints": [],
    "clarification_request": None, "hitl_context": None,
    "pipeline_logs": [], "result": {},
}


class ErrorClassificationLG:
    """Drop-in replacement for ErrorClassificationAPI backed by LangGraph."""

    def __init__(self, deps: PipelineDeps) -> None:
        self.deps = deps
        self._app = build_pipeline(deps)

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
    ) -> "ErrorClassificationLG":
        store = SQLAlchemyArtifactStore(db_url)
        return cls(PipelineDeps(
            client=client, store=store,
            core_by_id=store.load_core_by_id(),
            retrieval_cards=[],
            repair_index=store.build_repair_index(),
            selector_model=selector_model,
            extractor_model=extractor_model,
            judge_model=judge_model,
            top_k=top_k, m_min_top_score=m_min_top_score,
        ))

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
    ) -> "ErrorClassificationLG":
        core_by_id = _load_core(core_path)
        retrieval_cards = _load_retrieval(retrieval_path)
        repair_rows = _load_repair(repair_path)
        validate_artifact_relations(
            core_cards=list(core_by_id.values()),
            retrieval_cards=retrieval_cards,
            repair_rules=repair_rows,
        )
        return cls(PipelineDeps(
            client=client, store=None,
            core_by_id=core_by_id,
            retrieval_cards=retrieval_cards,
            repair_index=_build_repair_index(repair_rows),
            selector_model=selector_model,
            extractor_model=extractor_model,
            judge_model=judge_model,
            top_k=top_k, m_min_top_score=m_min_top_score,
        ))

    def diagnose(
        self,
        *,
        question: str,
        context: str,
        case_id: str = "",
        gold_answer: Optional[Any] = None,
        domain: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> Dict[str, Any]:
        initial = {**_EMPTY_STATE,
                   "case_id": case_id, "question": question, "context": context,
                   "domain": domain, "topic": topic, "gold_answer": gold_answer}
        return self._app.invoke(initial)["result"]

    def diagnose_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return self.diagnose(
            question=str(row.get("question", "")),
            context=str(row.get("context", "")),
            case_id=str(row.get("case_id") or row.get("question_id") or ""),
            gold_answer=row.get("gold_answer", row.get("answer", row.get("ground_truth"))),
            domain=str(row.get("domain", "") or "") or None,
            topic=str(row.get("topic", "") or "") or None,
        )
