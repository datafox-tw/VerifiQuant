"""
nearood_common.py
-----------------
Shared machinery for the P2 retrieval near-OOD experiment.

Idea: when a question's target formula family is NOT in the library, does the
system (a) abstain / refuse safely (N), or (b) force-fit a program onto a
semantically-adjacent-but-inapplicable neighbor and emit a confident wrong number?

We simulate OOD by a per-question *retrieval exclusion set* rather than rebuilding
DBs. Two conditions:

  bare      exclude = {target_fic} U nearest_neighbors   -> nothing adjacent remains
  nearmiss  exclude = {target_fic}                        -> neighbors remain (harder)

`ExclusionStore` intercepts retrieval for BOTH systems: VerifiQuant reads
`api.store.retrieve_candidates(...)` and the R-PAL baseline calls the same method,
so setting `store.current_exclude` before each case ablates the library identically
for both. No LLM calls live here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from verifiquant.card_store import SQLAlchemyArtifactStore


class ExclusionStore(SQLAlchemyArtifactStore):
    """A card store that hides a per-query set of fic_ids from retrieval.

    Set `current_exclude` before each retrieval. It over-fetches so that after
    dropping the excluded ids `top_k` real candidates still remain.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.current_exclude: Set[str] = set()

    def retrieve_candidates(
        self,
        *,
        query: str,
        top_k: int = 5,
        domain: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        pad = len(self.current_exclude) + 6
        rows = super().retrieve_candidates(
            query=query, top_k=top_k + pad, domain=domain, topic=topic
        )
        kept = [r for r in rows if str(r.get("fic_id", "")).strip() not in self.current_exclude]
        return kept[:top_k]


def retrieve_pool(store: SQLAlchemyArtifactStore, query: str, k: int = 30) -> List[str]:
    """Ranked fic_ids for a query (full library, no exclusion) -- used offline to
    read a target's nearest neighbors when building the probe manifest."""
    rows = store.retrieve_candidates(query=query, top_k=k)
    out: List[str] = []
    for r in rows:
        fid = str(r.get("fic_id", "")).strip()
        if fid and fid not in out:
            out.append(fid)
    return out


def exclusion_for(condition: str, target_fic: str, neighbors: List[str]) -> Set[str]:
    """Exclusion set for a probe under a condition."""
    if condition == "bare":
        return {target_fic, *neighbors}
    if condition == "nearmiss":
        return {target_fic}
    raise ValueError(f"unknown condition {condition!r} (use 'bare' or 'nearmiss')")
