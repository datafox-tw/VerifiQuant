import argparse
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def _simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class CardRecord:
    """Container for a single data card."""

    data: Dict[str, Any]
    source_path: Path

    @property
    def id(self) -> str:
        return str(self.data.get("id", ""))

    @property
    def domain(self) -> str:
        return str(self.data.get("domain", "")).strip()

    @property
    def topic(self) -> str:
        return str(self.data.get("topic", "")).strip()

    def context_snippet(self) -> str:
        """Return a rich text representation for downstream LLM prompts."""
        inputs_desc = ", ".join(
            f"{inp.get('name')}: {inp.get('description')}"
            for inp in self.data.get("inputs", [])
        )
        formulas = "; ".join(
            f"{f.get('variable')} = {f.get('formula')}"
            for f in self.data.get("sympy_formulas", [])
        )
        return (
            f"[{self.id}] {self.data.get('name','')} — {self.data.get('short_description','')}\n"
            f"Domain: {self.domain} | Topic: {self.topic}\n"
            f"Inputs: {inputs_desc}\n"
            f"Output: {self.data.get('output_var','')}\n"
            f"Formulas: {formulas}\n"
            f"Tags: {', '.join(self.data.get('tags', []))}"
        )

    def searchable_text(self) -> str:
        parts: List[str] = []
        for key in (
            "id",
            "name",
            "short_description",
            "domain",
            "topic",
        ):
            value = self.data.get(key)
            if isinstance(value, str):
                parts.append(value)
        for inp in self.data.get("inputs", []):
            parts.append(f"{inp.get('name','')}: {inp.get('description','')}")
        for formula in self.data.get("sympy_formulas", []):
            parts.append(f"{formula.get('variable','')}: {formula.get('formula','')}")
        if "tags" in self.data:
            parts.extend(self.data["tags"])
        return " ".join(parts)


class DefinitionStore:
    """
    Hybrid retriever blending BM25 keyword search with embedding cosine similarity.
    """

    def __init__(
        self,
        cards: Sequence[CardRecord],
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        precomputed_embeddings: Optional[np.ndarray] = None,
        precomputed_tokens: Optional[List[List[str]]] = None,
    ) -> None:
        if not cards:
            raise ValueError("Cannot build DefinitionStore with empty cards.")
        self.cards: List[CardRecord] = list(cards)
        self._model_name = embedding_model
        self._encoder = SentenceTransformer(embedding_model)

        self._corpus = [card.searchable_text() for card in self.cards]
        self._tokenized_docs = (
            precomputed_tokens
            if precomputed_tokens is not None
            else [_simple_tokenize(doc) for doc in self._corpus]
        )
        self._embeddings = (
            precomputed_embeddings
            if precomputed_embeddings is not None
            else self._encoder.encode(
                self._corpus, convert_to_numpy=True, show_progress_bar=False
            )
        )
        self._bm25 = BM25Okapi(self._tokenized_docs)

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        min_s = scores.min()
        max_s = scores.max()
        if np.isclose(max_s, min_s):
            return np.ones_like(scores)
        return (scores - min_s) / (max_s - min_s)

    def retrieve_top_k(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        *,
        domain: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> List[Tuple[CardRecord, float]]:
        """Return top_k cards ranked by hybrid score."""
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be between 0 and 1.")
        candidate_indices = self._filter_indices(domain=domain, topic=topic)
        if not candidate_indices:
            return []

        query_tokens = _simple_tokenize(query)
        bm25_scores_all = np.array(self._bm25.get_scores(query_tokens))
        query_vec = self._encoder.encode([query], convert_to_numpy=True)[0]
        cos_scores = np.array(
            (self._embeddings @ query_vec)
            / (
                np.linalg.norm(self._embeddings, axis=1)
                * (np.linalg.norm(query_vec) + 1e-8)
            )
        )
        bm25_scores = bm25_scores_all[candidate_indices]
        cos_scores = cos_scores[candidate_indices]

        bm25_norm = self._normalize(bm25_scores)
        cos_norm = self._normalize(cos_scores)
        hybrid = alpha * bm25_norm + (1 - alpha) * cos_norm

        ranked_local = np.argsort(hybrid)[::-1][:top_k]
        ranked_idx = [candidate_indices[i] for i in ranked_local]
        return [(self.cards[i], float(hybrid_local)) for i, hybrid_local in zip(ranked_idx, hybrid[ranked_local])]

    def _filter_indices(
        self,
        *,
        domain: Optional[str],
        topic: Optional[str],
    ) -> List[int]:
        indices = []
        for idx, card in enumerate(self.cards):
            if domain and card.domain.lower() != domain.lower():
                continue
            if topic and card.topic.lower() != topic.lower():
                continue
            indices.append(idx)
        if not domain and not topic:
            return list(range(len(self.cards)))
        return indices

    def save(self, path: Path) -> None:
        payload = {
            "model": self._model_name,
            "cards": [card.data for card in self.cards],
            "sources": [str(card.source_path) for card in self.cards],
            "embeddings": self._embeddings,
            "tokenized_docs": self._tokenized_docs,
        }
        with path.open("wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, path: Path) -> "DefinitionStore":
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        cards = [
            CardRecord(data=data, source_path=Path(src))
            for data, src in zip(payload["cards"], payload["sources"])
        ]
        return cls(
            cards,
            embedding_model=payload["model"],
            precomputed_embeddings=payload.get("embeddings"),
            precomputed_tokens=payload.get("tokenized_docs"),
        )


def load_cards_from_dir(
    data_dir: Path,
    *,
    domain: Optional[str] = None,
    topic: Optional[str] = None,
) -> List[CardRecord]:
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)

    dedup: Dict[str, CardRecord] = {}
    for json_path in sorted(data_dir.rglob("*.json")):
        try:
            entries = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(entries, dict):
            entries = [entries]
        for entry in entries:
            card_id = entry.get("id")
            if not card_id:
                continue
            record = CardRecord(entry, json_path)
            if domain and record.domain.lower() != domain.lower():
                continue
            if topic and record.topic.lower() != topic.lower():
                continue
            dedup[card_id] = record
    return list(dedup.values())


def build_store_cli(
    data_dir: Path,
    output_path: Path,
    embedding_model: str,
    domain: Optional[str],
    topic: Optional[str],
) -> None:
    cards = load_cards_from_dir(data_dir, domain=domain, topic=topic)
    if not cards:
        raise ValueError(
            f"No cards found under {data_dir} with filters domain={domain}, topic={topic}"
        )
    store = DefinitionStore(cards, embedding_model=embedding_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    store.save(output_path)
    print(
        f"DefinitionStore built with {len(cards)} cards and saved to {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a DefinitionStore from generated data cards."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root directory containing domain/topic JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/definition_store.pkl"),
        help="Path to persist the serialized DefinitionStore.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformers model for embeddings.",
    )
    parser.add_argument(
        "--domain",
        help="Optional domain filter (e.g., 'Investment Analysis').",
    )
    parser.add_argument(
        "--topic",
        help="Optional topic filter (e.g., 'Net Present Value').",
    )
    args = parser.parse_args()
    build_store_cli(
        args.data_dir,
        args.output,
        args.embedding_model,
        args.domain,
        args.topic,
    )


if __name__ == "__main__":
    main()

