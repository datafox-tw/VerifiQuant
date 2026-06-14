"""
run_refusal_ablation_grid.py
----------------------------
Run the prompt-controlled refusal ablation grid on the clean 50Q set and build
the master risk-coverage table.

Grid (priority order; a hard question budget stops the run before overspend):
  OpenAI gpt-5.2 : level {0,1,2,3} x reasoning_effort {low, medium}   (8 cells)
  Gemini  flash  : level {0,3,1,2}                                     (4 cells)

Each cell = 50 questions. The --max-questions budget (default 500 = "10x 50Q")
caps total questions answered across all cells; cells are run in the order below
and the grid stops cleanly once the next cell would exceed the budget. The L0
baselines (per provider/effort) run first so refusal quality is always scorable.

Master table per cell:
  correct / silent_wrong / safe_refusal, coverage, selective_acc, SWR,
  plus refusal-quality vs the matching L0 baseline (same provider & effort):
    good_refusal  = L0 got it wrong  AND this cell refused   (SWR rescued)
    over_refusal  = L0 got it right  AND this cell refused   (accuracy lost)

Usage:
    python3 scripts/run_refusal_ablation_grid.py
    python3 scripts/run_refusal_ablation_grid.py --aggregate-only
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "verifiquant/data/runs/paper_v1/questions_50.jsonl"
RUN_DIR = ROOT / "verifiquant/data/runs/refusal_ablation"
PIPELINE = ROOT / "verifiquant/pipeline/run_refusal_ablation.py"

# (cell_name, provider, model, effort, level) — priority order. L0 baselines first.
GRID = [
    ("gpt_low_L0",    "openai", "gpt-5.2",          "low",    0),
    ("gpt_medium_L0", "openai", "gpt-5.2",          "medium", 0),
    ("gpt_low_L1",    "openai", "gpt-5.2",          "low",    1),
    ("gpt_low_L2",    "openai", "gpt-5.2",          "low",    2),
    ("gpt_low_L3",    "openai", "gpt-5.2",          "low",    3),
    ("gpt_medium_L1", "openai", "gpt-5.2",          "medium", 1),
    ("gpt_medium_L2", "openai", "gpt-5.2",          "medium", 2),
    ("gpt_medium_L3", "openai", "gpt-5.2",          "medium", 3),
    ("gem_L0",        "gemini", "gemini-2.5-flash", "low",    0),
    ("gem_L3",        "gemini", "gemini-2.5-flash", "low",    3),
    ("gem_L1",        "gemini", "gemini-2.5-flash", "low",    1),
    ("gem_L2",        "gemini", "gemini-2.5-flash", "low",    2),
]

N_PER_CELL = 50


def _cell_dir(name: str) -> Path:
    return RUN_DIR / name


def run_grid(force: bool, max_questions: int) -> None:
    spent = 0
    for name, provider, model, effort, level in GRID:
        out = _cell_dir(name) / "output.jsonl"
        summ = _cell_dir(name) / "summary.json"
        if out.exists() and not force:
            print(f"[skip] {name} (output exists)")
            continue
        if spent + N_PER_CELL > max_questions:
            print(f"\n[budget] stopping: next cell {name} would push answered "
                  f"questions to {spent + N_PER_CELL} > cap {max_questions}.")
            break
        cmd = [
            sys.executable, str(PIPELINE),
            "--input", str(INPUT),
            "--provider", provider, "--model", model,
            "--refusal-level", str(level),
            "--output", str(out), "--summary-output", str(summ),
        ]
        if provider == "openai":
            cmd += ["--reasoning-effort", effort]
        print(f"\n[run] {name}  (spent={spent}, +{N_PER_CELL})")
        print("  $ " + " ".join(cmd))
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"  [warn] {name} returncode={rc}; continuing.")
            continue
        spent += N_PER_CELL
    print(f"\n[budget] total questions answered this run: {spent}")


def _baseline_key(provider: str, effort: str) -> str:
    return f"{provider}:{effort}:L0"


def aggregate() -> dict:
    # 1. load every cell that has output
    cells = {}
    for name, provider, model, effort, level in GRID:
        out = _cell_dir(name) / "output.jsonl"
        summ = _cell_dir(name) / "summary.json"
        if not out.exists():
            continue
        rows = {}
        with open(out, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[r["case_id"]] = r["refusal_ablation"]
        summary = json.loads(summ.read_text()) if summ.exists() else {}
        cells[name] = {
            "provider": provider, "model": model, "effort": effort,
            "level": level, "rows": rows, "summary": summary,
        }

    # 2. index L0 baselines by (provider, effort)
    baselines = {}
    for name, c in cells.items():
        if c["level"] == 0:
            baselines[_baseline_key(c["provider"], c["effort"])] = c["rows"]

    # 3. per-cell refusal-quality vs its L0 baseline
    def verdict(rec):
        if rec["decision"] == "refuse" and rec["final_answer"] is None:
            return "safe_refusal"
        return "correct" if rec["is_correct"] is True else "silent_wrong"

    table = []
    for name, c in cells.items():
        s = c["summary"]
        base = baselines.get(_baseline_key(c["provider"], c["effort"]))
        good = over = unknown = 0
        if base is not None and c["level"] != 0:
            for cid, rec in c["rows"].items():
                if verdict(rec) != "safe_refusal":
                    continue
                b = base.get(cid)
                if b is None:
                    unknown += 1
                elif b.get("is_correct") is True:
                    over += 1     # L0 answered correctly; refusing here lost accuracy
                else:
                    good += 1     # L0 was (silent) wrong; refusing here rescued an SWR
        table.append({
            "cell": name, "provider": c["provider"], "effort": c["effort"],
            "level": c["level"],
            "correct": s.get("correct"), "silent_wrong": s.get("silent_wrong"),
            "safe_refusal": s.get("safe_refusal"),
            "accuracy": s.get("accuracy"), "coverage": s.get("coverage"),
            "selective_accuracy": s.get("selective_accuracy"), "swr": s.get("swr"),
            "good_refusal": good, "over_refusal": over,
            "refusal_vs_baseline_unknown": unknown,
        })
    table.sort(key=lambda r: (r["provider"], r["effort"], r["level"]))
    return {"cells_run": list(cells.keys()), "table": table}


def _print_table(report: dict) -> None:
    print(f"\n{'='*108}\n  Refusal Prompt Ablation — clean 50Q\n{'='*108}")
    hdr = (f"  {'cell':<16}{'prov':<8}{'eff':<8}{'L':<3}"
           f"{'corr':>5}{'SW':>4}{'refuse':>8}{'acc':>7}{'SWR':>7}{'good':>6}{'over':>6}")
    print(hdr); print("  " + "-"*104)
    for r in report["table"]:
        acc = f"{r['accuracy']:.0%}" if r["accuracy"] is not None else "-"
        swr = f"{r['swr']:.0%}" if r["swr"] is not None else "-"
        print(f"  {r['cell']:<16}{r['provider']:<8}{r['effort']:<8}{r['level']:<3}"
              f"{r['correct']:>5}{r['silent_wrong']:>4}{r['safe_refusal']:>8}"
              f"{acc:>7}{swr:>7}{r['good_refusal']:>6}{r['over_refusal']:>6}")
    print("\n  good = L0-wrong & refused (SWR rescued)   over = L0-correct & refused (acc lost)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--max-questions", type=int, default=500,
                    help="hard cap on total questions answered (default 500 = 10x 50Q).")
    args = ap.parse_args()

    if not args.aggregate_only:
        run_grid(args.force, args.max_questions)

    report = aggregate()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "master_table.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_table(report)
    print(f"\n  master table -> {RUN_DIR / 'master_table.json'}")


if __name__ == "__main__":
    main()
