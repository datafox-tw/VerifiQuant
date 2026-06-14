"""
run_multik_refusal.py
---------------------
Multi-K blind self-correction on the refusal-leveled CoT (clean 50Q).

Runs each cell ONCE at max K, then truncates the per-turn history to recover the
whole K=1..max curve (same trick as paper §5.9), so we pay for one run, not K of
them. Blind protocol: the oracle reviews every turn (no GT gating) and may only
use ground-truth *code* logic — see run_cot_self_improve_pipeline.

Cells (per teacher's pick): GPT medium L0 & L3, Gemini flash L0 & L3, K=6.

Usage:
    python3 scripts/run_multik_refusal.py
    python3 scripts/run_multik_refusal.py --aggregate-only
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "verifiquant/data/runs/paper_v1/questions_50.jsonl"
RUN_DIR = ROOT / "verifiquant/data/runs/refusal_ablation_multik"
PIPELINE = ROOT / "verifiquant/pipeline/run_refusal_ablation.py"
MAX_K = 6

# (cell, provider, model, effort, level)
GRID = [
    ("gpt_medium_L0", "openai", "gpt-5.2",          "medium", 0),
    ("gpt_medium_L1", "openai", "gpt-5.2",          "medium", 1),
    ("gpt_medium_L3", "openai", "gpt-5.2",          "medium", 3),
    ("gem_L0",        "gemini", "gemini-2.5-flash", "low",    0),
    ("gem_L1",        "gemini", "gemini-2.5-flash", "low",    1),
    ("gem_L3",        "gemini", "gemini-2.5-flash", "low",    3),
]


def run_grid(force: bool) -> None:
    for name, provider, model, effort, level in GRID:
        out = RUN_DIR / name / "output.jsonl"
        summ = RUN_DIR / name / "summary.json"
        if out.exists() and not force:
            print(f"[skip] {name}")
            continue
        cmd = [
            sys.executable, str(PIPELINE), "--input", str(INPUT),
            "--provider", provider, "--model", model,
            "--refusal-level", str(level), "--max-turns", str(MAX_K),
            "--output", str(out), "--summary-output", str(summ),
        ]
        if provider == "openai":
            cmd += ["--reasoning-effort", effort]
        print(f"\n[run] {name} (K={MAX_K})\n  $ " + " ".join(cmd))
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"  [warn] {name} returncode={rc}")


def _verdict_of(turn_state: dict) -> str:
    if turn_state["decision"] == "refuse" and turn_state["parsed_answer"] is None:
        return "safe_refusal"
    return "correct" if turn_state["is_correct"] is True else "silent_wrong"


def _state_at_k(history: list, k: int) -> dict:
    # natural termination may stop before k; use the last available turn
    idx = min(k, len(history)) - 1
    return history[idx]


def aggregate() -> dict:
    out = {"max_k": MAX_K, "cells": {}}
    for name, provider, model, effort, level in GRID:
        path = RUN_DIR / name / "output.jsonl"
        if not path.exists():
            continue
        cases = [json.loads(l)["refusal_ablation"] for l in open(path, encoding="utf-8") if l.strip()]
        n = len(cases)
        curve = []
        k1_verdict = [_verdict_of(_state_at_k(c["history"], 1)) for c in cases]
        for k in range(1, MAX_K + 1):
            vs = [_verdict_of(_state_at_k(c["history"], k)) for c in cases]
            correct = sum(v == "correct" for v in vs)
            sw = sum(v == "silent_wrong" for v in vs)
            sr = sum(v == "safe_refusal" for v in vs)
            answered = correct + sw
            recovered = sum(1 for a, b in zip(k1_verdict, vs) if a != "correct" and b == "correct")
            broken = sum(1 for a, b in zip(k1_verdict, vs) if a == "correct" and b != "correct")
            curve.append({
                "k": k, "correct": correct, "silent_wrong": sw, "safe_refusal": sr,
                "accuracy": round(correct / n, 4) if n else None,
                "swr": round(sw / answered, 4) if answered else None,
                "recovered_vs_k1": recovered, "broken_vs_k1": broken,
            })
        out["cells"][name] = {
            "provider": provider, "model": model, "effort": effort, "level": level,
            "n": n, "curve": curve,
            "avg_rounds": round(sum(c["rounds"] for c in cases) / n, 2) if n else None,
        }
    return out


def _print(report: dict) -> None:
    for name, c in report["cells"].items():
        print(f"\n{'='*72}\n  {name}  ({c['model']}/{c['effort']}/L{c['level']}, "
              f"avg_rounds={c['avg_rounds']})\n{'='*72}")
        print(f"  {'K':>2}{'corr':>6}{'SW':>5}{'refuse':>8}{'acc':>7}{'SWR':>7}"
              f"{'recov':>7}{'broke':>7}")
        for r in c["curve"]:
            acc = f"{r['accuracy']:.0%}" if r["accuracy"] is not None else "-"
            swr = f"{r['swr']:.0%}" if r["swr"] is not None else "-"
            print(f"  {r['k']:>2}{r['correct']:>6}{r['silent_wrong']:>5}{r['safe_refusal']:>8}"
                  f"{acc:>7}{swr:>7}{r['recovered_vs_k1']:>7}{r['broken_vs_k1']:>7}")
    print("\n  recov/broke = vs K=1 (recovered: not-correct@K1 -> correct@K; broken: correct@K1 -> not@K)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    if not args.aggregate_only:
        run_grid(args.force)
    report = aggregate()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "multik_curves.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print(report)
    print(f"\n  curves -> {RUN_DIR / 'multik_curves.json'}")


if __name__ == "__main__":
    main()
