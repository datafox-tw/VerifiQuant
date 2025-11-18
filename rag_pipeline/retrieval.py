from dataclasses import dataclass
from typing import List, Optional

from card_store import CardRecord, DefinitionStore


@dataclass
class RetrievalCandidate:
    card: CardRecord
    score: float

    def as_context(self) -> str:
        return f"Score: {self.score:.3f}\n{self.card.context_snippet()}"


def fetch_candidates(
    store: DefinitionStore,
    query: str,
    top_k: int = 3,
    alpha: float = 0.4,
    *,
    domain: Optional[str] = None,
    topic: Optional[str] = None,
) -> List[RetrievalCandidate]:
    results = store.retrieve_top_k(
        query=query,
        top_k=top_k,
        alpha=alpha,
        domain=domain,
        topic=topic,
    )
    return [
        RetrievalCandidate(card=record, score=score)
        for record, score in results
    ]

