#!/usr/bin/env python3
"""Promote the loan-payment timing ambiguity (fic_article_2164) to I_HARD.

The demo's hero case needs the begin/end timing ambiguity to *intercept* (I_HARD: no
number until the human clarifies) rather than answer-then-offer (I_soft). The fair
paper_v1 run ships this hint as soft, so this script flips it for the DEMO database only.

Idempotent: safe to re-run after a card-store rebuild. It edits, in
verifiquant/data/runs/paper_v1/fic/cards.db:
  - core_cards.semantic_hints_json: payment_timing hint i_level "soft" -> "hard"
  - repair_rules: rule i_payment_timing_soft -> rule_id i_payment_timing_hard,
    diagnostic_type I_HARD (transform_map preserved verbatim)

This is demo theater on a copy of the cards, NOT a change to the paper's reported results.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3

FIC_ID = "fic_article_2164"
HINT_ID = "payment_timing"
OLD_RULE_ID = "i_payment_timing_soft"
NEW_RULE_ID = "i_payment_timing_hard"

DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "verifiquant", "data", "runs", "paper_v1", "fic", "cards.db",
)


def promote(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        # 1) semantic_hints: payment_timing i_level -> hard
        row = con.execute(
            "SELECT semantic_hints_json FROM core_cards WHERE fic_id = ?", (FIC_ID,)
        ).fetchone()
        if not row:
            raise SystemExit(f"core card {FIC_ID} not found in {db_path}")
        hints = json.loads(row["semantic_hints_json"] or "[]")
        changed = False
        for hint in hints:
            if str(hint.get("id", "")) == HINT_ID and str(hint.get("i_level", "")) != "hard":
                hint["i_level"] = "hard"
                changed = True
        if changed:
            con.execute(
                "UPDATE core_cards SET semantic_hints_json = ? WHERE fic_id = ?",
                (json.dumps(hints, ensure_ascii=False), FIC_ID),
            )
            print(f"  semantic_hints: {HINT_ID} i_level -> hard")
        else:
            print(f"  semantic_hints: {HINT_ID} already hard (no change)")

        # 2) repair rule: soft -> hard (transform_map preserved)
        rule = con.execute(
            "SELECT id, rule_id, diagnostic_type FROM repair_rules WHERE fic_id = ? AND rule_id IN (?, ?)",
            (FIC_ID, OLD_RULE_ID, NEW_RULE_ID),
        ).fetchone()
        if not rule:
            raise SystemExit(f"repair rule {OLD_RULE_ID}/{NEW_RULE_ID} not found for {FIC_ID}")
        if rule["rule_id"] == NEW_RULE_ID and str(rule["diagnostic_type"]).upper() == "I_HARD":
            print(f"  repair rule already {NEW_RULE_ID} / I_HARD (no change)")
        else:
            con.execute(
                "UPDATE repair_rules SET rule_id = ?, diagnostic_type = ? WHERE id = ?",
                (NEW_RULE_ID, "I_HARD", rule["id"]),
            )
            print(f"  repair rule: {rule['rule_id']}/{rule['diagnostic_type']} -> {NEW_RULE_ID}/I_HARD")

        con.commit()
        print("done.")
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DEFAULT_DB, help="path to cards.db (default: paper_v1)")
    args = ap.parse_args()
    print(f"Promoting loan timing to I_HARD in {args.db}")
    promote(args.db)


if __name__ == "__main__":
    main()
