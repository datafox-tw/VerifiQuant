"""
build_nearood_probes.py
------------------------
Offline probe-set builder for the P2 retrieval near-OOD experiment. No LLM calls.

For each probe question we record its target formula family (the fic_id the clean
canonical run selected) and that target's nearest retrieval neighbors. At eval time
`nearood_common.exclusion_for` turns this into the `bare` and `nearmiss` conditions.

Selection criterion: keep a question only if its target fic is the RANK-1 retrieval
hit for the clean query. That guarantees the target is genuinely in-library and
retrievable, so removing it is a real OOD ablation rather than a pre-existing miss.
Each formula family is used at most once, for family diversity.

Usage:
    python3 scripts/build_nearood_probes.py \
        --questions verifiquant/data/runs/paper_v2_250/questions_250.jsonl \
        --canonical verifiquant/data/runs/paper_v2_250/results/vq_flash/output.jsonl \
        --db        verifiquant/data/runs/paper_v2_250/fic/cards.db \
        --out       verifiquant/data/runs/nearood/probes.jsonl \
        --n 24 --neighbors 6
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verifiquant.card_store import SQLAlchemyArtifactStore  # noqa: E402
from verifiquant.pipeline.nearood_common import retrieve_pool  # noqa: E402
from verifiquant.pipeline.build_trap_set_v2 import _load_jsonl, _load_canonical_map, _sha256  # noqa: E402


def build(questions_path: str, canonical_path: str, db_path: str,
          out_path: str, n: int, neighbors: int) -> Dict[str, Any]:
    cases = _load_jsonl(questions_path)
    canon = _load_canonical_map(canonical_path)
    store = SQLAlchemyArtifactStore(f"sqlite:///{db_path}")
    cases.sort(key=lambda c: c.get("question_id", ""))

    probes: List[Dict[str, Any]] = []
    used_families = set()
    skipped_not_rank1 = 0
    for case in cases:
        if len(probes) >= n:
            break
        cid = case.get("question_id")
        target = canon.get(cid, {}).get("fic_id")
        if not target or target in used_families:
            continue
        query = f"{case['question']}\n{case.get('context', '')}"
        pool = retrieve_pool(store, query, k=neighbors + 8)
        if not pool or pool[0] != target:
            skipped_not_rank1 += 1
            continue                                   # target not the top hit -> skip
        nbrs = [f for f in pool if f != target][:neighbors]
        probes.append({
            "probe_id": f"nearood-{cid}",
            "question_id": cid,
            "target_fic": target,
            "neighbors": nbrs,
            "question": case["question"],
            "context": case.get("context", ""),
            "function": case.get("function", ""),
            "ground_truth": case.get("ground_truth"),
        })
        used_families.add(target)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for p in probes:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    manifest = {
        "protocol": "nearood_probes_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_probes": len(probes),
        "neighbors_per_probe": neighbors,
        "selection": "target_fic == rank-1 retrieval hit; one probe per family",
        "skipped_not_rank1": skipped_not_rank1,
        "conditions": {
            "bare": "exclude target + neighbors (nothing adjacent remains)",
            "nearmiss": "exclude target only (adjacent neighbors remain)",
        },
        "sources": {
            "questions": {"path": questions_path, "sha256": _sha256(questions_path)},
            "canonical": {"path": canonical_path, "sha256": _sha256(canonical_path)},
            "cards_db": {"path": db_path, "sha256": _sha256(db_path)},
        },
        "probes_sha256": _sha256(str(out)),
    }
    with open(out.parent / "probes_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[nearood] wrote {len(probes)} probes -> {out}")
    print(f"  neighbors/probe: {neighbors}  | skipped (target not rank-1): {skipped_not_rank1}")
    print(f"  distinct target families: {len(used_families)}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Build near-OOD probe set (offline).")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--neighbors", type=int, default=6)
    args = ap.parse_args()
    build(args.questions, args.canonical, args.db, args.out, args.n, args.neighbors)


if __name__ == "__main__":
    main()
