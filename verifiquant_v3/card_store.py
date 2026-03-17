from __future__ import annotations

import argparse
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency guard
    np = None
    BM25Okapi = None
    SentenceTransformer = None


def _simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


@dataclass
class FICRecord:
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
        inputs_desc = ", ".join(
            f"{inp.get('name')}: {inp.get('description')}"
            for inp in self.data.get("inputs", [])
        )
        return (
            f"[{self.id}] {self.data.get('name','')} — {self.data.get('short_description','')}\n"
            f"Domain: {self.domain} | Topic: {self.topic}\n"
            f"Inputs: {inputs_desc}\n"
            f"Output: {self.data.get('output_var','')}\n"
            f"Selection hints: {self.data.get('selection_hints',{}).get('self_description','')}"
        )

    def searchable_text(self) -> str:
        parts: List[str] = []
        for key in ("id", "name", "short_description", "domain", "topic", "output_var"):
            value = self.data.get(key)
            if isinstance(value, str):
                parts.append(value)
        for inp in self.data.get("inputs", []):
            parts.append(f"{inp.get('name','')}: {inp.get('description','')}")
        diagnostics = self.data.get("diagnostics", {}) or {}
        for inv in diagnostics.get("invariants", []):
            parts.append(str(inv.get("message", "")))
            parts.append(str(inv.get("rule", "")))
        for chk in diagnostics.get("scale_checks", []):
            parts.append(str(chk.get("message", "")))
            parts.append(str(chk.get("expected", "")))
        hints = self.data.get("selection_hints", {}) or {}
        parts.append(str(hints.get("self_description", "")))
        for k in ("applicable_when", "not_applicable_when", "required_input_summary", "disambiguation_prompts"):
            v = hints.get(k, [])
            if isinstance(v, list):
                parts.extend(str(x) for x in v)
        for conf in hints.get("common_confusions", []):
            if isinstance(conf, dict):
                parts.append(str(conf.get("label", "")))
                parts.append(str(conf.get("difference", "")))
        return " ".join(parts)


class FICStore:
    """Hybrid retriever for FIC v2 cards (BM25 + embedding cosine)."""

    def __init__(
        self,
        cards: Sequence[FICRecord],
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        precomputed_embeddings: Optional[np.ndarray] = None,
        precomputed_tokens: Optional[List[List[str]]] = None,
    ) -> None:
        if np is None or BM25Okapi is None or SentenceTransformer is None:
            raise RuntimeError(
                "Missing dependencies for FICStore. Install numpy, rank_bm25, and sentence-transformers."
            )
        if not cards:
            raise ValueError("Cannot build FICStore with empty cards.")
        self.cards: List[FICRecord] = list(cards)
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
            else self._encoder.encode(self._corpus, convert_to_numpy=True, show_progress_bar=False)
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
        alpha: float = 0.4,
        *,
        domain: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> List[Tuple[FICRecord, float]]:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be between 0 and 1.")
        idxs = self._filter_indices(domain=domain, topic=topic)
        if not idxs:
            return []

        q_tokens = _simple_tokenize(query)
        bm25_all = np.array(self._bm25.get_scores(q_tokens))
        q_vec = self._encoder.encode([query], convert_to_numpy=True)[0]
        cos_all = np.array(
            (self._embeddings @ q_vec)
            / ((np.linalg.norm(self._embeddings, axis=1) * (np.linalg.norm(q_vec) + 1e-8)))
        )

        bm25 = bm25_all[idxs]
        cos = cos_all[idxs]
        hybrid = alpha * self._normalize(bm25) + (1 - alpha) * self._normalize(cos)
        ranked_local = np.argsort(hybrid)[::-1][:top_k]
        ranked_idxs = [idxs[i] for i in ranked_local]
        return [(self.cards[i], float(hybrid_local)) for i, hybrid_local in zip(ranked_idxs, hybrid[ranked_local])]

    def _filter_indices(self, *, domain: Optional[str], topic: Optional[str]) -> List[int]:
        out = []
        for i, c in enumerate(self.cards):
            if domain and c.domain.lower() != domain.lower():
                continue
            if topic and c.topic.lower() != topic.lower():
                continue
            out.append(i)
        if not domain and not topic:
            return list(range(len(self.cards)))
        return out

    def save(self, path: Path) -> None:
        payload = {
            "model": self._model_name,
            "cards": [c.data for c in self.cards],
            "sources": [str(c.source_path) for c in self.cards],
            "embeddings": self._embeddings,
            "tokenized_docs": self._tokenized_docs,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, path: Path) -> "FICStore":
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        cards = [
            FICRecord(data=data, source_path=Path(src))
            for data, src in zip(payload["cards"], payload["sources"])
        ]
        return cls(
            cards,
            embedding_model=payload["model"],
            precomputed_embeddings=payload.get("embeddings"),
            precomputed_tokens=payload.get("tokenized_docs"),
        )


def load_cards_from_file(path: Path, *, domain: Optional[str] = None, topic: Optional[str] = None) -> List[FICRecord]:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        rows = payload if isinstance(payload, list) else [payload]

    out: List[FICRecord] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rec = FICRecord(row, source_path=path)
        if domain and rec.domain.lower() != domain.lower():
            continue
        if topic and rec.topic.lower() != topic.lower():
            continue
        out.append(rec)
    return out


def build_store_from_cards(
    cards: Iterable[Dict[str, Any]],
    *,
    output_path: Path,
    source_path: Path,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    records = [FICRecord(dict(c), source_path=source_path) for c in cards]
    store = FICStore(records, embedding_model=embedding_model)
    store.save(output_path)
    print(f"FICStore built with {len(records)} cards and saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a persistent FICStore (.pkl) from FIC v2 cards JSON/JSONL.")
    parser.add_argument("--input", required=True, type=Path, help="Input FIC cards JSON/JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Output .pkl path")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--domain")
    parser.add_argument("--topic")
    args = parser.parse_args()

    cards = load_cards_from_file(args.input, domain=args.domain, topic=args.topic)
    if not cards:
        raise ValueError("No cards loaded with given filters")
    store = FICStore(cards, embedding_model=args.embedding_model)
    store.save(args.output)
    print(f"FICStore built with {len(cards)} cards and saved to {args.output}")


if __name__ == "__main__":
    main()
