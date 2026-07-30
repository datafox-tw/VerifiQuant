"""
analyze_governance_confirmatory.py
----------------------------------
Focused analyzer for the P1 confirmatory 80-trap run. Reuses the EXACT scoring
functions from analyze_governance_eval.py (normalize_vq / normalize_rpal /
score_traps) so metric definitions match the pilot. Trap-only: the clean axis is
covered separately by the audited 250Q R-PAL, so we do not couple to clean files.

Outputs:
  - per-system overall FNAR / diagnostic-accuracy / action-accuracy
    (prompted R-PAL = mean over its 3 runs, with range)
  - per-operator FNAR (F/E/I/N) and the I_HARD vs I_SOFT split
  - base-question-clustered bootstrap 95% CI on the (prompted - VQ) FNAR gap
  - go/no-go vs H1: gap >= 10 points AND CI lower bound > 0
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_governance_eval import (  # noqa: E402
    load_jsonl, normalize_vq, normalize_rpal, score_traps,
)

G = ROOT / "verifiquant/data/runs/governance_confirmatory"


def _flags_by_trap(traps: Sequence[Mapping[str, Any]], rows, normalizer) -> Dict[str, int]:
    """trap_id -> unsafe_numeric (0/1) using the shared scorer's details."""
    scored = score_traps(traps, rows, normalizer)
    return {d["case_id"]: int(d["unsafe_numeric"]) for d in scored["details"]}, scored


def _fnar(flags: Mapping[str, float], ids: Sequence[str]) -> float:
    return sum(flags[i] for i in ids) / len(ids) if ids else 0.0


def main() -> None:
    traps = load_jsonl(G / "trap_set.jsonl")
    trap_ids = [str(t["trap_id"]) for t in traps]
    base_of = {str(t["trap_id"]): t["base_question_id"] for t in traps}
    expdt_of = {str(t["trap_id"]): t["expected_diagnostic"] for t in traps}
    op_of = {str(t["trap_id"]): t["operator"] for t in traps}

    vq_flags, vq_scored = _flags_by_trap(traps, load_jsonl(G / "vq_k1_traps.jsonl"), normalize_vq)
    plain_flags, plain_scored = _flags_by_trap(traps, load_jsonl(G / "conf_plain_run1.jsonl"), normalize_rpal)
    prompted_runs = []
    prompted_scored = []
    for i in (1, 2, 3):
        f, s = _flags_by_trap(traps, load_jsonl(G / f"conf_prompted_run{i}.jsonl"), normalize_rpal)
        prompted_runs.append(f)
        prompted_scored.append(s)
    prompted_mean = {tid: statistics.mean(r[tid] for r in prompted_runs) for tid in trap_ids}

    # ---- overall + per-operator (F/E/I/N) ----
    def op_fnar(flags):
        out = {}
        for op in ("F", "E", "I", "N"):
            ids = [t for t in trap_ids if op_of[t] == op]
            out[op] = round(_fnar(flags, ids), 4)
        return out

    def isplit(flags):
        out = {}
        for lvl in ("I_HARD", "I_SOFT"):
            ids = [t for t in trap_ids if expdt_of[t] == lvl]
            out[lvl] = {"n": len(ids), "fnar": round(_fnar(flags, ids), 4)}
        return out

    overall = {
        "vq_k1": {
            "fnar": round(_fnar(vq_flags, trap_ids), 4),
            "diagnostic_accuracy": round(vq_scored["overall"]["diagnostic_accuracy"], 4),
            "action_accuracy": round(vq_scored["overall"]["action_accuracy"], 4),
        },
        "rpal_plain": {"fnar": round(_fnar(plain_flags, trap_ids), 4)},
        "rpal_prompted_mean": {
            "fnar": round(_fnar(prompted_mean, trap_ids), 4),
            "fnar_per_run": [round(_fnar(r, trap_ids), 4) for r in prompted_runs],
            "diagnostic_accuracy": round(
                statistics.mean(s["overall"]["diagnostic_accuracy"] for s in prompted_scored), 4),
            "action_accuracy": round(
                statistics.mean(s["overall"]["action_accuracy"] for s in prompted_scored), 4),
        },
    }
    per_operator = {"vq_k1": op_fnar(vq_flags), "rpal_plain": op_fnar(plain_flags),
                    "rpal_prompted_mean": op_fnar(prompted_mean)}
    i_split = {"vq_k1": isplit(vq_flags), "rpal_prompted_mean": isplit(prompted_mean)}

    # ---- base-clustered bootstrap on (prompted_mean - vq) FNAR gap ----
    clusters = defaultdict(list)
    for tid in trap_ids:
        clusters[base_of[tid]].append(tid)
    cluster_keys = list(clusters)
    rng = random.Random(20260729)
    point_gap = _fnar(prompted_mean, trap_ids) - _fnar(vq_flags, trap_ids)
    boots = []
    for _ in range(10000):
        sample_ids: List[str] = []
        for _ in cluster_keys:
            sample_ids.extend(clusters[rng.choice(cluster_keys)])
        boots.append(_fnar(prompted_mean, sample_ids) - _fnar(vq_flags, sample_ids))
    boots.sort()
    lo, hi = boots[249], boots[9750]

    h1_gap_ok = point_gap >= 0.10
    h1_ci_ok = lo > 0.0
    verdict = "GO" if (h1_gap_ok and h1_ci_ok) else ("NARROW" if point_gap >= 0.10 else "NO-GO")

    summary = {
        "n_traps": len(trap_ids),
        "n_base_clusters": len(cluster_keys),
        "overall": overall,
        "per_operator_fnar": per_operator,
        "i_hard_soft_fnar": i_split,
        "prompted_minus_vq_fnar_gap": {
            "point": round(point_gap, 4),
            "ci95_base_clustered": [round(lo, 4), round(hi, 4)],
        },
        "h1": {"gap_ge_10pts": h1_gap_ok, "ci_lower_gt_0": h1_ci_ok, "verdict": verdict},
    }
    out = G / "confirmatory_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # console
    print(f"n_traps={len(trap_ids)}  base_clusters={len(cluster_keys)}")
    print("\nOverall FNAR:")
    print(f"  VQ K=1          {overall['vq_k1']['fnar']:.3f}  "
          f"diag={overall['vq_k1']['diagnostic_accuracy']:.3f} action={overall['vq_k1']['action_accuracy']:.3f}")
    print(f"  R-PAL plain     {overall['rpal_plain']['fnar']:.3f}")
    print(f"  R-PAL prompted  {overall['rpal_prompted_mean']['fnar']:.3f}  "
          f"(runs {overall['rpal_prompted_mean']['fnar_per_run']}) "
          f"diag={overall['rpal_prompted_mean']['diagnostic_accuracy']:.3f} "
          f"action={overall['rpal_prompted_mean']['action_accuracy']:.3f}")
    print("\nPer-operator FNAR (F/E/I/N):")
    for sysname in ("vq_k1", "rpal_plain", "rpal_prompted_mean"):
        print(f"  {sysname:20s} {per_operator[sysname]}")
    print("\nI_HARD / I_SOFT FNAR:")
    for sysname in ("vq_k1", "rpal_prompted_mean"):
        print(f"  {sysname:20s} {i_split[sysname]}")
    print(f"\n(prompted - VQ) FNAR gap: {point_gap:+.3f}  "
          f"95% CI [{lo:+.3f}, {hi:+.3f}]  -> {verdict}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
