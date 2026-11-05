"""Freeze the 50-question sample before any system runs on it.

Stratified by level (medium 36, hard 14, the 180:70 ratio of the 250), at most two
questions per topic, fixed seed. Never chosen by any system's outcome.
"""

from __future__ import annotations

import collections
import json
import random

from .data import SAMPLE, fic_id, questions, topics

SEED = 20261031
QUOTA = {"medium": 36, "hard": 14}
PER_TOPIC = 2


def draw() -> list[str]:
    rng = random.Random(SEED)
    qs, tp = questions(), topics()
    picked, per_topic = [], collections.Counter()
    for level, n in QUOTA.items():
        pool = sorted(qid for qid, q in qs.items() if q["level"] == level)
        rng.shuffle(pool)
        got = 0
        for qid in pool:
            t = tp.get(fic_id(qs[qid]), "?")
            if per_topic[t] >= PER_TOPIC:
                continue
            picked.append(qid)
            per_topic[t] += 1
            got += 1
            if got == n:
                break
    return picked


if __name__ == "__main__":
    if SAMPLE.exists():
        raise SystemExit(f"{SAMPLE.name} already frozen; delete it by hand to redraw")
    ids = draw()
    qs, tp = questions(), topics()
    SAMPLE.write_text(json.dumps({
        "seed": SEED, "quota": QUOTA, "per_topic": PER_TOPIC, "question_ids": ids,
        "topics": len({tp.get(fic_id(qs[i])) for i in ids}),
    }, indent=1) + "\n")
    print(len(ids), "questions,", collections.Counter(qs[i]["level"] for i in ids))
