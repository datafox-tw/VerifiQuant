#!/usr/bin/env python3
"""Offline validation of the literal-grounding audit (strict_binding) against the
stored 250Q run — zero LLM calls.

Replays audit_binding over every committed turn's (question, context, provided_inputs)
from the canonical vq_flash run and reports:
  - flag rate on CORRECT cases  (= coverage cost if the guard were enabled)
  - flag rate on SILENT-WRONG cases (= catch potential on the binding-slip class)
  - the flagged-correct examples (to tune the normalizer)
"""
from __future__ import annotations
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from verifiquant.pipeline.strict_binding import audit_binding  # noqa: E402

F = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "verifiquant/data/runs/paper_v2_250/results/vq_flash/output.jsonl")


def main() -> None:
    stats = {"correct": [0, 0], "silent_wrong": [0, 0]}  # class -> [flagged, total]
    flagged_examples = []
    for line in open(F):
        d = json.loads(line)["framework_guided"]
        # last committed turn (the one that produced the final answer)
        turn = None
        for h in d.get("history", []):
            diag = h.get("diagnostic", {})
            if isinstance(diag, dict) and diag.get("provided_inputs"):
                turn = h
        if turn is None:
            continue
        diag = turn["diagnostic"]
        committed = d.get("final_output_value") is not None or d.get("final_status") == "success"
        if not committed:
            continue
        cls = "correct" if d.get("final_is_correct") else "silent_wrong"
        _, ung = audit_binding(diag["provided_inputs"], turn.get("question", ""), turn.get("context", ""))
        stats[cls][1] += 1
        if ung:
            stats[cls][0] += 1
            if cls == "correct" and len(flagged_examples) < 15:
                flagged_examples.append((diag.get("case_id"), ung,
                                         {k: diag["provided_inputs"][k] for k in ung}))
            if cls == "silent_wrong":
                print(f"  SW-case {diag.get('case_id')}: ungrounded={ung} "
                      f"values={ {k: diag['provided_inputs'][k] for k in ung} }")

    for cls, (fl, tot) in stats.items():
        pct = 100 * fl / tot if tot else 0
        print(f"{cls:12}: flagged {fl}/{tot} ({pct:.1f}%)")
    print("\nflagged-correct examples (coverage cost — tune normalizer against these):")
    for cid, names, vals in flagged_examples:
        print(f"  {cid}: {vals}")


if __name__ == "__main__":
    main()
