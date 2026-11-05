"""FNAR with and without the harness, split by trap family (the tables in FNAR_ANALYSIS.md).

    python3 -m harness_demo.poc.fnar_breakdown               # dev split, the paper's 80 traps
    python3 -m harness_demo.poc.fnar_breakdown --split test  # held-out traps (needs the R-PAL / VQ test runs)

Every system is scored from its own rows with the paper's `score_traps`; a run that does not cover every
trap is left out, because the scorer counts a missing row as safe.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys

from .data import ROOT, RUNS
from . import run_traps

sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

G = ROOT / "verifiquant/data/runs/governance_confirmatory"
G1014 = ROOT / "verifiquant/data/runs/governance_confirmatory_1014"
RT = RUNS / "rpal_test_traps"
SYSTEMS = {
    "dev": [("R-PAL plain", "rpal", [G / "conf_plain_run1.jsonl"]),
            ("R-PAL+", "rpal", [G / f"conf_prompted_run{i}.jsonl" for i in (1, 2, 3)]),
            ("harness", "harness", [RUNS / t / "results.jsonl" for t in ("traps_r1", "traps_r2", "traps_r3", "traps_v5_r1")]),
            ("VQ", "vq", [G1014 / f"vq_k1_top4_run{i}.jsonl" for i in (1, 2, 3)])],
    "test": [("R-PAL plain", "rpal", [RT / "rpal_plain_test_run1.jsonl"]),
             ("R-PAL+", "rpal", [RT / f"rpal_prompted_test_run{i}.jsonl" for i in (1, 2, 3)]),
             ("harness", "harness", [RUNS / f"test_traps_v5_r{i}" / "results.jsonl" for i in (1, 2, 3)]),
             ("VQ", "vq", [RUNS / f"vq_test_traps/vq_k1_top4_test_run{i}.jsonl" for i in (1, 2, 3)])],
}
FAMILIES = [("F", "F 缺欄位"), ("E", "E 輸入不合理"), ("I_HARD", "I_HARD 困難歧義"), ("I_SOFT", "I_SOFT 輕微歧義"),
            ("N", "N 範圍外")]


def flags_for(kind, path, ts):
    from scripts.analyze_governance_eval import load_jsonl, normalize_rpal, normalize_vq, score_traps

    ids = {t["trap_id"] for t in ts}
    if kind == "harness":
        rows = [run_traps.as_rpal(json.loads(l)) for l in path.read_text().splitlines() if l.strip()]
        norm = normalize_rpal
    else:
        rows, norm = load_jsonl(path), (normalize_vq if kind == "vq" else normalize_rpal)
    rows = [r for r in rows if r.get("case_id") in ids]
    if {r["case_id"] for r in rows} != ids:
        return None
    return {d["case_id"]: int(d["unsafe_numeric"]) for d in score_traps(ts, rows, norm)["details"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=sorted(SYSTEMS), default="dev")
    args = ap.parse_args()
    run_traps.split = args.split
    ts = run_traps.traps()
    fam = {t["trap_id"]: t["expected_diagnostic"] for t in ts}
    n_fam = collections.Counter(fam.values())
    hard = [i for i in fam if fam[i] != "I_SOFT"]
    L = [f"# FNAR 拆解（{args.split}，{len(ts)} 題）", "",
         "| 系統 | 完整執行次數 | FNAR | 不含 I_SOFT | " + " | ".join(name for _, name in FAMILIES) + " | 每次漏幾題 |",
         "|---|---|---|---|" + "---|" * len(FAMILIES) + "---|"]
    for name, kind, paths in SYSTEMS[args.split]:
        runs = [f for f in (flags_for(kind, p, ts) for p in paths if p.exists()) if f is not None]
        if not runs:
            L.append(f"| {name} | 0（尚未執行或未完成） |" + " |" * (len(FAMILIES) + 3))
            continue
        fnar = statistics.mean(sum(f.values()) / len(ts) for f in runs)
        nosoft = statistics.mean(sum(f[i] for i in hard) / len(hard) for f in runs)
        per = {k: statistics.mean(sum(f[i] for i in fam if fam[i] == k) for f in runs) for k, _ in FAMILIES}
        cells = " | ".join(f"{per[k]:.2f} / {n_fam[k]}" for k, _ in FAMILIES)
        L.append(f"| {name} | {len(runs)} | {fnar:.1%} | {nosoft:.1%} | {cells} | "
                 f"{statistics.mean(sum(f.values()) for f in runs):.2f} |")
    L += ["", "類別欄：每次執行平均漏幾題 / 該類題數。"]
    print("\n".join(L))


if __name__ == "__main__":
    main()
