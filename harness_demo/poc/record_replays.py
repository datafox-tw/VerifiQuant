"""Record complete harness runs (every step body and the audit) for the demo's PoC replay cases.

The demo site cannot run the PoC itself (no 250-card library, no auto cards in the image), so it replays
runs recorded here with the admitted v6c settings: auto cards with self-check, rounding gate on, inspector
independent of the card. Each case is recorded once and shown as it ran; its story says how often that
outcome occurred across the earlier runs, so a replay is never a hand-picked lucky draw.

    python3 -m harness_demo.poc.record_replays          -> harness_demo/replays/<case id>.json
"""

from __future__ import annotations

import json

from .. import harness
from .data import ROOT, questions, text_of
from .provider import PocProvider
from . import run_traps

OUT = ROOT / "harness_demo/replays"
# case id -> (question or trap id, split for traps or None)
CASES = {
    "poc_amortization": ("test-2218", None),
    "poc_keltner": ("test-2236", None),
    "poc_inspector": ("test-1588", None),
    "poc_lwma": ("test-1801", None),
    "poc_trap_missing": ("F-test-1023", "dev"),
    "poc_trap_scope": ("N-test-1086", "dev"),
    "poc_trap_absurd": ("E-test-1006", "dev"),
}


def record(case_id: str) -> dict:
    qid, split = CASES[case_id]
    if split:
        run_traps.split = split
        q = next(t for t in run_traps.traps() if t["trap_id"] == qid)
    else:
        q = questions()[qid]
    PocProvider.self_check, PocProvider.verify_inspector, PocProvider.reject_rounding = True, False, True
    res = harness.run(PocProvider(), text_of(q), {})
    res["replay"] = {"source": qid, "settings": "v6c", "gold": q.get("ground_truth")}
    OUT.mkdir(exist_ok=True)
    (OUT / f"{case_id}.json").write_text(json.dumps(res, ensure_ascii=False))
    return res


if __name__ == "__main__":
    import sys

    for cid in sys.argv[1:] or CASES:
        r = record(cid)
        d = r["decision"]
        print(f"{cid:20s} {d['action']:8s} {d.get('headline', '')[:70]}")
