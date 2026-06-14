#!/usr/bin/env python3
"""Import project-valuation cards from the 0415 run into paper_v1 (demo only).

paper_v1 has return-metric cards (ROI/ROA/Alpha) but no NPV/payback, so an ambiguous
"is this project profitable?" question can't produce a real M abstain ("which method?").
This copies NPV / NPVGO / Discounted-Payback from demo_50q_0415 into paper_v1's card store
so the M-class demo shows multiple competing contracts and the selector asks which one.

Uses the card_store API (ingest_artifacts) so the retrieval FTS index stays in sync.
Idempotent (upsert). Demo theater on the demo DB — NOT a change to the paper's results.
"""
from __future__ import annotations

import argparse
import os

from verifiquant.card_store import SQLAlchemyArtifactStore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DB = os.path.join(ROOT, "verifiquant", "data", "runs", "demo_50q_0415", "cards.db")
DST_DB = os.path.join(ROOT, "verifiquant", "data", "runs", "paper_v1", "fic", "cards.db")

# NPV, NPVGO, Discounted Payback Period — competing project-valuation methods.
IMPORT_IDS = {"fic_article_2940", "fic_article_2942", "fic_article_1733"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=SRC_DB)
    ap.add_argument("--dst", default=DST_DB)
    args = ap.parse_args()

    src = SQLAlchemyArtifactStore(f"sqlite:///{args.src}")
    dst = SQLAlchemyArtifactStore(f"sqlite:///{args.dst}")

    core = [c for c in src.load_core_by_id().values() if c.get("fic_id") in IMPORT_IDS]
    retrieval = [r for r in src.load_retrieval_cards() if r.get("fic_id") in IMPORT_IDS]
    repair = [r for r in src.load_repair_rules() if r.get("fic_id") in IMPORT_IDS]

    found = {c["fic_id"] for c in core}
    missing = IMPORT_IDS - found
    if missing:
        print(f"WARNING: not found in source: {sorted(missing)}")
    if not core:
        raise SystemExit("nothing to import")

    print(f"Importing {len(core)} core / {len(retrieval)} retrieval / {len(repair)} repair "
          f"rows into {args.dst}")
    report = dst.ingest_artifacts(
        core_cards=core,
        retrieval_cards=retrieval,
        repair_rules=repair,
        validate_relations=False,
    )
    print("done. card_count now:", report.get("core_card_count") or report.get("card_count"))


if __name__ == "__main__":
    main()
