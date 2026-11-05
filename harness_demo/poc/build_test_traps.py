"""Held-out trap TEST set: the paper's generator (build_trap_set_v2.py) on the base questions the 80 dev traps
do not use, with out-of-scope (N) asks the dev set never saw.

The 80 traps in datasets/03_traps_80 become the DEV set: the harness may be changed after looking at them.
This set is frozen before any harness change is tested on it and is only scored, never inspected per item
to tune the harness. Same operators, families and quotas as the dev set; no LLM, no gold answer in any label.

    python3 -m harness_demo.poc.build_test_traps      -> harness_demo/poc/traps_test/trap_set.jsonl + manifest
"""

from __future__ import annotations

import json
import re

from verifiquant.pipeline import build_trap_set_v2 as gen

from .data import QUESTIONS, ROOT

DEV = ROOT / "datasets/03_traps_80/trap_set.jsonl"
OUT = ROOT / "harness_demo/poc/traps_test"
CANONICAL = ROOT / "verifiquant/data/runs/paper_v2_250/results/vq_flash/output.jsonl"
CARDS_DB = ROOT / "verifiquant/data/runs/paper_v2_250/fic/cards.db"

# none of these appear in the dev templates; the near-miss metrics have no card in the 250 library
# (checked against retrieval_cards titles and summaries on 2026-11-05)
TEST_TEMPLATES = [
    ("exotic_lsm", "Price the Bermudan feature of this contract with a Longstaff-Schwartz least-squares "
                   "Monte-Carlo and report its value."),
    ("exotic_garch", "Fit a GARCH(1,1) model to this scenario's returns and forecast next month's volatility."),
    ("exotic_hullwhite", "Calibrate a Hull-White one-factor short-rate model to this scenario and report the "
                         "mean-reversion speed."),
    ("exotic_es", "Estimate the 97.5% Expected Shortfall of this position with filtered historical simulation."),
    ("exotic_ml_pd", "Train a gradient-boosted classifier on this information and report the predicted "
                     "probability of default."),
    ("near_miss_sortino", "Report the Sortino ratio of this investment against a 2% minimum acceptable return."),
    ("near_miss_calmar", "Report the Calmar ratio of this investment over the last three years."),
]


def construction_errors(t: dict) -> list[str]:
    """Structural, label-independent validity check (the 80 dev traps, which were reviewed by hand, have none).
    Decided on 2026-11-05 after three F traps turned out to still state the value they claim to remove."""
    text = t["context"] + " " + t["question"]
    errs = []
    if re.search(r"\d\.\[REDACTED\]|\[REDACTED\]\d", text):
        errs.append("redaction landed inside another number")
    if t["operator"] == "F":
        v = t["grounding"]["removed_value"]
        if any(re.search(r"(?<![\d.])" + re.escape(f) + r"(?!\d)", text) for f in gen._value_forms(v)):
            errs.append(f"removed value {v} is still stated in the text")
    return errs


def main():
    dev_bases = {json.loads(l)["base_question_id"] for l in DEV.read_text().splitlines() if l.strip()}
    OUT.mkdir(parents=True, exist_ok=True)
    pool = OUT / "questions_excluding_dev_bases.jsonl"
    rows = [json.loads(l) for l in QUESTIONS.read_text().splitlines() if l.strip()]
    keep = [r for r in rows if r["question_id"] not in dev_bases]
    pool.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep))
    gen._N_TEMPLATES = TEST_TEMPLATES
    manifest = gen.build(str(pool), str(CANONICAL), str(CARDS_DB), str(OUT), per_operator=20, max_ops_per_base=2)
    manifest.update(protocol="harness_poc_test_traps_v1", split="test",
                    excluded_dev_bases=sorted(dev_bases), n_pool=len(keep),
                    n_templates=[k for k, _ in TEST_TEMPLATES])
    (OUT / "trap_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    test = [json.loads(l) for l in (OUT / "trap_set.jsonl").read_text().splitlines() if l.strip()]
    dropped = {t["trap_id"]: construction_errors(t) for t in test if construction_errors(t)}
    test = [t for t in test if t["trap_id"] not in dropped]
    (OUT / "trap_set.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in test))
    manifest.update(dropped_construction_errors=dropped, total_traps=len(test),
                    trap_set_sha256=gen._sha256(str(OUT / "trap_set.jsonl")))
    (OUT / "trap_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    assert not {t["base_question_id"] for t in test} & dev_bases, "a test trap reuses a dev base question"
    print(f"pool {len(keep)} questions (excluded {len(dev_bases)} dev bases); test traps {len(test)}; "
          f"dropped {len(dropped)} for construction errors: {dropped}")


if __name__ == "__main__":
    main()
