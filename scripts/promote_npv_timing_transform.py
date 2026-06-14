#!/usr/bin/env python3
"""Inject the verified begin/end Atomic Transform into the NPV card (fic_article_2940).

The demo's NPV hero needs the cash-flow-timing I_HARD choice to actually CHANGE the
number via a verifiable atomic transform (end-of-period -> beginning-of-period),
the same way the loan hero (fic_article_2164) does. The paper_v1 card shipped this
hint with bare-string options and an empty repair_action (no transform_map), so
picking "Beginning of each period" fell back to a context re-diagnose and the FIC
recomputed end-of-period -> identical number.

This script edits, in verifiquant/data/runs/paper_v1/fic/cards.db (DEMO copy only):
  - core_cards.semantic_hints_json : cash_flow_timing options -> structured dicts;
    the beginning option carries a verified transform_spec (sets has_transform=true
    so the frontend routes to /api/transform/apply).
  - repair_rules (i_cash_flow_timing_hard): diagnostic_type -> I_HARD and
    repair_action_json gains transform_map[beginning_of_period] = same spec
    (the documented runtime contract get_transform_spec_for_choice reads).

Verified transform (end -> beginning):
  npv_begin = (npv_end + I)*(1+r) - I
  invariant: (result_new + I) == (result_old + I)*(1+r)
Idempotent; safe to re-run after a card-store rebuild. Demo theater, NOT a change
to any reported paper result.
"""
from __future__ import annotations
import argparse, json, os, sqlite3

FIC_ID = "fic_article_2940"
HINT_ID = "cash_flow_timing"
RULE_ID = "i_cash_flow_timing_hard"
END_VAL, BEGIN_VAL = "end_of_period", "beginning_of_period"

TRANSFORM_SPEC = {
    "patch_type": "result_postprocess",
    "result_expr": "(result + initial_investment) * (1 + discount_rate) - initial_investment",
    "max_expr_nodes": 18,
    "invariant": "(result_new + initial_investment) == (result_old + initial_investment) * (1 + discount_rate)",
    "affected_inputs": ["initial_investment", "discount_rate"],
}

STRUCTURED_OPTIONS = [
    {"label": "End of each period (standard for this formula)", "value": END_VAL, "is_default": True},
    {"label": "Beginning of each period", "value": BEGIN_VAL, "is_default": False,
     "transform_spec": TRANSFORM_SPEC},
]

DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "verifiquant", "data", "runs", "paper_v1", "fic", "cards.db",
)


def promote(db_path: str) -> None:
    con = sqlite3.connect(db_path); con.row_factory = sqlite3.Row
    try:
        # 1) semantic_hints: replace cash_flow_timing options with structured dicts
        row = con.execute("SELECT semantic_hints_json FROM core_cards WHERE fic_id=?", (FIC_ID,)).fetchone()
        if not row:
            raise SystemExit(f"core card {FIC_ID} not found in {db_path}")
        hints = json.loads(row["semantic_hints_json"] or "[]")
        found = False
        for h in hints:
            if str(h.get("id", "")) == HINT_ID:
                h["options"] = STRUCTURED_OPTIONS
                h["i_level"] = "hard"
                found = True
        if not found:
            raise SystemExit(f"hint {HINT_ID} not found on {FIC_ID}")
        con.execute("UPDATE core_cards SET semantic_hints_json=? WHERE fic_id=?",
                    (json.dumps(hints, ensure_ascii=False), FIC_ID))
        print(f"  semantic_hints: {HINT_ID} options -> structured (+transform_spec on {BEGIN_VAL})")

        # 2) repair rule: I_HARD + transform_map
        rule = con.execute(
            "SELECT id, repair_action_json, ask_user_for_json FROM repair_rules WHERE fic_id=? AND rule_id=?",
            (FIC_ID, RULE_ID)).fetchone()
        if not rule:
            raise SystemExit(f"repair rule {RULE_ID} not found for {FIC_ID}")
        ra = json.loads(rule["repair_action_json"] or "{}")
        ra["type"] = "present_clarification_options"
        ra["target"] = HINT_ID
        ra["transform_map"] = {BEGIN_VAL: TRANSFORM_SPEC}
        # keep ask_user_for_json option values in sync with the structured values
        ask = json.loads(rule["ask_user_for_json"] or "[]")
        for slot in ask:
            if isinstance(slot, dict) and slot.get("options"):
                slot["options"] = [
                    {"value": END_VAL, "label": "End of each period (standard for this formula)"},
                    {"value": BEGIN_VAL, "label": "Beginning of each period"},
                ]
        con.execute(
            "UPDATE repair_rules SET diagnostic_type=?, repair_action_json=?, ask_user_for_json=? WHERE id=?",
            ("I_HARD", json.dumps(ra, ensure_ascii=False), json.dumps(ask, ensure_ascii=False), rule["id"]))
        print(f"  repair rule {RULE_ID}: diagnostic_type -> I_HARD, transform_map[{BEGIN_VAL}] added")

        con.commit()
        print("done.")
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DEFAULT_DB)
    args = ap.parse_args()
    print(f"Injecting NPV begin/end transform in {args.db}")
    promote(args.db)


if __name__ == "__main__":
    main()
