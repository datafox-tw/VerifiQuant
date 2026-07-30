#!/usr/bin/env python3
"""Aggregate the library-size sweep and draw the coverage-tracking curve.

For each sweep point (lib_10/25/50/75/100) computes, from the point's results jsonl
(summary json preferred when present):
  ID accuracy, ID abstain, OOD abstain, OOD silent-wrong, selectivity gap
and draws them vs library size. Output: docs/figures/libsweep_250q.png + a
machine-readable aggregate at verifiquant/data/runs/libsweep/aggregate.json.

Robust to partial points (uses whatever rows exist; marks them as partial).
"""
from __future__ import annotations
import glob, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP = os.path.join(ROOT, "verifiquant/data/runs/libsweep")
FRACS = [10, 25, 50, 75, 100]


def point_stats(frac: int):
    d = os.path.join(SWEEP, f"results_{frac}")
    rows = []
    files = sorted(glob.glob(os.path.join(d, "ood_results_*.jsonl")), key=os.path.getmtime)
    if not files:
        return None
    for line in open(files[-1]):
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        return None
    sums = sorted(glob.glob(os.path.join(d, "ood_summary_*.json")), key=os.path.getmtime)
    complete = bool(sums)

    def _num(x):
        try:
            return float(str(x).replace(",", "").replace("%", "").strip())
        except (TypeError, ValueError):
            return None

    def scale_near_miss(r):
        """Output equals gold up to a decimal<->percent factor (harness scores strictly;
        the full pipeline's output-scale normalization handles these)."""
        a, b = _num(r.get("output_value")), _num(r.get("gold"))
        if a is None or b is None:
            return False
        return any(abs(a * f - b) <= max(1e-9, abs(b) * 0.01) for f in (100.0, 0.01))

    def rate(split, verdict, normalized=False):
        n = sum(1 for r in rows if r["split"] == split)
        if not n:
            return None, 0
        if verdict == "silent_wrong" and normalized:
            k = sum(1 for r in rows if r["split"] == split and r["verdict"] == verdict
                    and not scale_near_miss(r))
        elif verdict == "correct" and normalized:
            k = sum(1 for r in rows if r["split"] == split
                    and (r["verdict"] == verdict
                         or (r["verdict"] == "silent_wrong" and scale_near_miss(r))))
        else:
            k = sum(1 for r in rows if r["split"] == split and r["verdict"] == verdict)
        return k / n, n

    id_acc, n_id = rate("ID", "correct", normalized=True)
    id_abst, _ = rate("ID", "abstain")
    id_sw, _ = rate("ID", "silent_wrong", normalized=True)
    ood_abst, n_ood = rate("OOD", "abstain")
    ood_sw, _ = rate("OOD", "silent_wrong", normalized=True)
    gap = (ood_abst - id_abst) if (ood_abst is not None and id_abst is not None) else None
    return {"frac": frac, "n_id": n_id, "n_ood": n_ood, "complete": complete,
            "id_acc": id_acc, "id_abstain": id_abst, "id_sw": id_sw,
            "ood_abstain": ood_abst, "ood_sw": ood_sw, "selectivity_gap": gap}


def main():
    pts = [p for p in (point_stats(f) for f in FRACS) if p]
    if not pts:
        raise SystemExit("no sweep results found")
    agg = {"points": pts}
    json.dump(agg, open(os.path.join(SWEEP, "aggregate.json"), "w"), indent=2)

    def series(key):
        xs = [p["frac"] for p in pts if p.get(key) is not None]
        ys = [100 * p[key] for p in pts if p.get(key) is not None]
        return xs, ys

    plt.rcParams.update({
        "font.size": 17, "axes.titlesize": 18, "axes.labelsize": 17,
        "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 15,
    })
    plt.figure(figsize=(7.4, 5.2))
    for key, label, style, color in [
        ("ood_abstain", "OOD abstention (families not in library)", "-o", "#1f77b4"),
        ("id_abstain", "ID abstention (covered families)", "-s", "#aec7e8"),
        ("id_acc", "ID accuracy (unit-normalized)", "-^", "#2ca02c"),
        ("ood_sw", "OOD silent-wrong (unit-normalized)", "-v", "#d62728"),
        ("id_sw", "ID silent-wrong (unit-normalized)", "--x", "#ff9896"),
    ]:
        xs, ys = series(key)
        if xs:
            plt.plot(xs, ys, style, color=color, label=label, markersize=7)
    xs, ys = series("selectivity_gap")
    if xs:
        plt.plot(xs, ys, ":d", color="#9467bd", label="selectivity gap (OOD − ID abstain)", markersize=7)
    plt.xlabel("Library size (% of the 250 formula families)")
    plt.ylabel("Rate (%)")
    plt.title("Abstention tracks library coverage (250Q, K=1)")
    plt.xticks(FRACS)
    plt.ylim(-2, 102)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=15, loc="center right")
    plt.tight_layout()
    out = os.path.join(ROOT, "docs/figures/libsweep_250q.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.08)
    print("figure:", out)
    for p in pts:
        tag = "" if p["complete"] else "  (PARTIAL)"
        print(f"  lib_{p['frac']:>3}: ID acc={_pc(p['id_acc'])} ID abst={_pc(p['id_abstain'])} "
              f"ID SW={_pc(p['id_sw'])} | OOD abst={_pc(p['ood_abstain'])} OOD SW={_pc(p['ood_sw'])} "
              f"| gap={_pc(p['selectivity_gap'])}  n={p['n_id']}+{p['n_ood']}{tag}")


def _pc(x):
    return "  n/a" if x is None else f"{100*x:5.1f}%"


if __name__ == "__main__":
    main()
