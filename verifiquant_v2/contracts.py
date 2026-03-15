from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional


RefusalCategory = Literal["M", "F", "E", "C", "Unknown"]
DiagnosticStatus = Literal["success", "refusal", "error", "alert", "needs_clarification"]
Severity = Literal["error", "alert"]

REFUSAL_CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "M": "Task misunderstanding or ambiguity (semantic mismatch, unclear intent).",
    "F": "Formula/spec mismatch or missing required specification/inputs.",
    "E": "Input binding, unit, or logical inconsistency detected.",
    "C": "Calculation/runtime error (expected to be rare under deterministic engine).",
    "Unknown": "Unclassified diagnostic outcome.",
}

USER_REPAIR_OPTIONS = [
    "manual_input_missing_fields",
    "confirm_unit_conversion",
    "select_alternative_fic",
    "force_run_with_override",
    "swap_suspected_fields",
    "rephrase_task_intent",
]


@dataclass
class DiagnosticFinding:
    id: str
    category: RefusalCategory
    severity: Severity
    message: str
    rule: Optional[str] = None
    evidence: Optional[Dict[str, Any]] = None
    suggested_fix: Optional[str] = None
    ui_action: Optional[str] = None


@dataclass
class DiagnosticReport:
    status: DiagnosticStatus
    diagnostic_type: RefusalCategory = "Unknown"
    reason_code: Optional[str] = None
    message: str = ""
    findings: List[DiagnosticFinding] = field(default_factory=list)
    requested_fields: List[str] = field(default_factory=list)
    reframe_suggestion: Optional[str] = None
    confidence: Optional[float] = None
    recommended_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["findings"] = [asdict(f) for f in self.findings]
        return payload


DIAGNOSTIC_REPORT_SCHEMA_EXAMPLE: Dict[str, Any] = {
    "status": "alert",
    "diagnostic_type": "E",
    "reason_code": "invariant_violation",
    "message": "Potential logical inconsistency detected.",
    "findings": [
        {
            "id": "inv_rf_lt_rm",
            "category": "E",
            "severity": "alert",
            "message": "Risk-free rate exceeds market return.",
            "rule": "risk_free_rate < market_return",
            "evidence": {"risk_free_rate": 0.085, "market_return": 0.03},
            "suggested_fix": "Verify whether Rf and Rm were swapped.",
            "ui_action": "swap_suspected_fields",
        }
    ],
    "requested_fields": [],
    "reframe_suggestion": None,
    "confidence": 0.92,
    "recommended_actions": ["swap_suspected_fields", "force_run_with_override"],
}

