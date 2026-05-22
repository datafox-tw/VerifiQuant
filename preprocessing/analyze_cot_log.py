from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


PROCESSING_RE = re.compile(r"\[verifiquant-lite-cot-self-improve\]\s+processing\s+(\S+)")
TURN_RE = re.compile(r"Turn\s+(\d+):.*?\bis_correct=(True|False|None)")


def analyze_log(path: Path, *, streak: int = 3) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        m_case = PROCESSING_RE.search(line)
        if m_case:
            if current is not None:
                cases.append(current)
            current = {"case_id": m_case.group(1), "turns": []}
            continue

        m_turn = TURN_RE.search(line)
        if m_turn and current is not None:
            current["turns"].append(
                {
                    "turn": int(m_turn.group(1)),
                    "is_correct": m_turn.group(2),
                }
            )

    if current is not None:
        cases.append(current)

    consecutive_false_cases = []
    for case in cases:
        values = [t["is_correct"] for t in case["turns"]]
        max_run = 0
        run = 0
        for value in values:
            if value == "False":
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        if max_run >= streak:
            consecutive_false_cases.append(case["case_id"])

    return {
        "log_path": str(path),
        "total_cases_seen": len(cases),
        "streak": streak,
        "consecutive_false_count": len(consecutive_false_cases),
        "consecutive_false_cases": consecutive_false_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze VerifiQuant-lite CoT console logs.")
    parser.add_argument("log", type=Path, help="Path to log.txt")
    parser.add_argument("--streak", type=int, default=3, help="Minimum consecutive False turns to count")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    summary = analyze_log(args.log, streak=max(1, args.streak))
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print(f"Log: {summary['log_path']}")
    print(f"Total cases seen: {summary['total_cases_seen']}")
    print(f"Consecutive False streak >= {summary['streak']}: {summary['consecutive_false_count']}")
    if summary["consecutive_false_cases"]:
        print("Cases: " + ", ".join(summary["consecutive_false_cases"]))


if __name__ == "__main__":
    main()
