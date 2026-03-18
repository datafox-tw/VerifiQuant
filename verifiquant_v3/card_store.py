from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from sqlalchemy import Float, Integer, String, Text, UniqueConstraint, create_engine, select, text
    from sqlalchemy.exc import SQLAlchemyError
    from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
except ImportError:  # pragma: no cover - optional dependency guard
    DeclarativeBase = object  # type: ignore[assignment]
    Mapped = Any  # type: ignore[assignment]
    Session = None  # type: ignore[assignment]
    mapped_column = None  # type: ignore[assignment]
    create_engine = None  # type: ignore[assignment]
    select = None  # type: ignore[assignment]
    text = None  # type: ignore[assignment]
    SQLAlchemyError = Exception

from verifiquant_v3.preprocessing.validate_relations import validate_artifact_relations


def _tokenize(text_value: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", str(text_value or "").lower())


def _load_records(path: Path) -> List[Dict[str, Any]]:
    text_value = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text_value.splitlines() if line.strip()]
    payload = json.loads(text_value)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported JSON payload type in {path}")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_loads(value: str, *, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


if mapped_column is not None:

    class Base(DeclarativeBase):
        pass


    class CoreCard(Base):
        __tablename__ = "core_cards"

        fic_id: Mapped[str] = mapped_column(String(255), primary_key=True)
        name: Mapped[str] = mapped_column(String(512), nullable=False)
        short_description: Mapped[str] = mapped_column(Text, nullable=False)
        domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
        topic: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
        version: Mapped[str] = mapped_column(String(64), nullable=False, default="v3")
        source_meta_json: Mapped[str] = mapped_column(Text, nullable=False)
        inputs_json: Mapped[str] = mapped_column(Text, nullable=False)
        output_json: Mapped[str] = mapped_column(Text, nullable=False)
        execution_json: Mapped[str] = mapped_column(Text, nullable=False)
        diagnostic_checks_json: Mapped[str] = mapped_column(Text, nullable=False)


    class RetrievalCard(Base):
        __tablename__ = "retrieval_cards"

        fic_id: Mapped[str] = mapped_column(String(255), primary_key=True)
        domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
        topic: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
        title: Mapped[str] = mapped_column(String(512), nullable=False)
        summary: Mapped[str] = mapped_column(Text, nullable=False)
        applicable_when_json: Mapped[str] = mapped_column(Text, nullable=False)
        not_applicable_when_json: Mapped[str] = mapped_column(Text, nullable=False)
        common_confusions_json: Mapped[str] = mapped_column(Text, nullable=False)
        keywords_json: Mapped[str] = mapped_column(Text, nullable=False)
        embedding_text: Mapped[str] = mapped_column(Text, nullable=False)


    class RepairRule(Base):
        __tablename__ = "repair_rules"

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        fic_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
        rule_id: Mapped[str] = mapped_column(String(255), nullable=False)
        diagnostic_type: Mapped[str] = mapped_column(String(8), nullable=False)
        severity: Mapped[str] = mapped_column(String(16), nullable=False)
        title: Mapped[str] = mapped_column(String(512), nullable=False)
        user_message: Mapped[str] = mapped_column(Text, nullable=False)
        explanation: Mapped[str] = mapped_column(Text, nullable=False)
        ask_user_for_json: Mapped[str] = mapped_column(Text, nullable=False)
        repair_action_json: Mapped[str] = mapped_column(Text, nullable=False)
        allowed_next_steps_json: Mapped[str] = mapped_column(Text, nullable=False)

        __table_args__ = (UniqueConstraint("fic_id", "rule_id", name="uq_repair_fic_rule"),)


class SQLAlchemyArtifactStore:
    """Persistent v3 artifact store with separated core/retrieval/repair tables."""

    def __init__(self, db_url: str) -> None:
        if create_engine is None or Session is None or mapped_column is None:
            raise RuntimeError(
                "Missing dependency: SQLAlchemy is required. Install with `pip install sqlalchemy`."
            )
        self.db_url = db_url
        self.engine = create_engine(db_url, future=True)
        self.is_sqlite = db_url.startswith("sqlite")
        self.create_schema()

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)
        if self.is_sqlite:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        "CREATE VIRTUAL TABLE IF NOT EXISTS retrieval_fts "
                        "USING fts5(fic_id UNINDEXED, title, summary, keywords, embedding_text)"
                    )
                )

    def ingest_artifacts(
        self,
        *,
        core_cards: List[Dict[str, Any]],
        retrieval_cards: List[Dict[str, Any]],
        repair_rules: List[Dict[str, Any]],
        validate_relations: bool = True,
    ) -> Dict[str, Any]:
        if validate_relations:
            validate_artifact_relations(
                core_cards=core_cards,
                retrieval_cards=retrieval_cards,
                repair_rules=repair_rules,
            )

        try:
            with Session(self.engine) as session:
                self._upsert_core_cards(session, core_cards)
                self._upsert_retrieval_cards(session, retrieval_cards)
                self._upsert_repair_rules(session, repair_rules)
                session.commit()
        except SQLAlchemyError as err:
            raise RuntimeError(f"Failed to ingest artifacts: {err}") from err

        if self.is_sqlite:
            self._sync_retrieval_fts(retrieval_cards)

        return self.validate_integrity()

    def _upsert_core_cards(self, session: Session, cards: Iterable[Dict[str, Any]]) -> None:
        for card in cards:
            fic_id = str(card.get("fic_id", "")).strip()
            if not fic_id:
                continue
            row = session.get(CoreCard, fic_id)
            payload = {
                "fic_id": fic_id,
                "name": str(card.get("name", "")),
                "short_description": str(card.get("short_description", "")),
                "domain": str(card.get("domain", "")),
                "topic": str(card.get("topic", "")),
                "version": str(card.get("version", "v3")),
                "source_meta_json": _json_dumps(card.get("source_meta", {})),
                "inputs_json": _json_dumps(card.get("inputs", [])),
                "output_json": _json_dumps(card.get("output", {})),
                "execution_json": _json_dumps(card.get("execution", {})),
                "diagnostic_checks_json": _json_dumps(card.get("diagnostic_checks", [])),
            }
            if row is None:
                session.add(CoreCard(**payload))
            else:
                for k, v in payload.items():
                    setattr(row, k, v)

    def _upsert_retrieval_cards(self, session: Session, cards: Iterable[Dict[str, Any]]) -> None:
        for card in cards:
            fic_id = str(card.get("fic_id", "")).strip()
            if not fic_id:
                continue
            row = session.get(RetrievalCard, fic_id)
            payload = {
                "fic_id": fic_id,
                "domain": str(card.get("domain", "")),
                "topic": str(card.get("topic", "")),
                "title": str(card.get("title", "")),
                "summary": str(card.get("summary", "")),
                "applicable_when_json": _json_dumps(card.get("applicable_when", [])),
                "not_applicable_when_json": _json_dumps(card.get("not_applicable_when", [])),
                "common_confusions_json": _json_dumps(card.get("common_confusions", [])),
                "keywords_json": _json_dumps(card.get("keywords", [])),
                "embedding_text": str(card.get("embedding_text", "")),
            }
            if row is None:
                session.add(RetrievalCard(**payload))
            else:
                for k, v in payload.items():
                    setattr(row, k, v)

    def _upsert_repair_rules(self, session: Session, rows: Iterable[Dict[str, Any]]) -> None:
        for rule in rows:
            fic_id = str(rule.get("fic_id", "")).strip()
            rule_id = str(rule.get("rule_id", "")).strip()
            if not fic_id or not rule_id:
                continue
            existing = session.execute(
                select(RepairRule).where(
                    RepairRule.fic_id == fic_id,
                    RepairRule.rule_id == rule_id,
                )
            ).scalar_one_or_none()
            payload = {
                "fic_id": fic_id,
                "rule_id": rule_id,
                "diagnostic_type": str(rule.get("diagnostic_type", "")),
                "severity": str(rule.get("severity", "")),
                "title": str(rule.get("title", "")),
                "user_message": str(rule.get("user_message", "")),
                "explanation": str(rule.get("explanation", "")),
                "ask_user_for_json": _json_dumps(rule.get("ask_user_for", [])),
                "repair_action_json": _json_dumps(rule.get("repair_action", {})),
                "allowed_next_steps_json": _json_dumps(rule.get("allowed_next_steps", [])),
            }
            if existing is None:
                session.add(RepairRule(**payload))
            else:
                for k, v in payload.items():
                    setattr(existing, k, v)

    def _sync_retrieval_fts(self, retrieval_cards: Iterable[Dict[str, Any]]) -> None:
        with self.engine.begin() as conn:
            for card in retrieval_cards:
                fic_id = str(card.get("fic_id", "")).strip()
                if not fic_id:
                    continue
                conn.execute(text("DELETE FROM retrieval_fts WHERE fic_id = :fic_id"), {"fic_id": fic_id})
                conn.execute(
                    text(
                        "INSERT INTO retrieval_fts (fic_id, title, summary, keywords, embedding_text) "
                        "VALUES (:fic_id, :title, :summary, :keywords, :embedding_text)"
                    ),
                    {
                        "fic_id": fic_id,
                        "title": str(card.get("title", "")),
                        "summary": str(card.get("summary", "")),
                        "keywords": " ".join(str(x) for x in (card.get("keywords", []) or [])),
                        "embedding_text": str(card.get("embedding_text", "")),
                    },
                )

    def _core_row_to_dict(self, row: CoreCard) -> Dict[str, Any]:
        return {
            "fic_id": row.fic_id,
            "name": row.name,
            "short_description": row.short_description,
            "domain": row.domain,
            "topic": row.topic,
            "version": row.version,
            "source_meta": _json_loads(row.source_meta_json, default={}),
            "inputs": _json_loads(row.inputs_json, default=[]),
            "output": _json_loads(row.output_json, default={}),
            "execution": _json_loads(row.execution_json, default={}),
            "diagnostic_checks": _json_loads(row.diagnostic_checks_json, default=[]),
        }

    def _retrieval_row_to_dict(self, row: RetrievalCard) -> Dict[str, Any]:
        return {
            "fic_id": row.fic_id,
            "domain": row.domain,
            "topic": row.topic,
            "title": row.title,
            "summary": row.summary,
            "applicable_when": _json_loads(row.applicable_when_json, default=[]),
            "not_applicable_when": _json_loads(row.not_applicable_when_json, default=[]),
            "common_confusions": _json_loads(row.common_confusions_json, default=[]),
            "keywords": _json_loads(row.keywords_json, default=[]),
            "embedding_text": row.embedding_text,
        }

    def _repair_row_to_dict(self, row: RepairRule) -> Dict[str, Any]:
        return {
            "fic_id": row.fic_id,
            "rule_id": row.rule_id,
            "diagnostic_type": row.diagnostic_type,
            "severity": row.severity,
            "title": row.title,
            "user_message": row.user_message,
            "explanation": row.explanation,
            "ask_user_for": _json_loads(row.ask_user_for_json, default=[]),
            "repair_action": _json_loads(row.repair_action_json, default={}),
            "allowed_next_steps": _json_loads(row.allowed_next_steps_json, default=[]),
        }

    def load_core_by_id(self) -> Dict[str, Dict[str, Any]]:
        with Session(self.engine) as session:
            rows = session.execute(select(CoreCard)).scalars().all()
        return {row.fic_id: self._core_row_to_dict(row) for row in rows}

    def load_retrieval_cards(self) -> List[Dict[str, Any]]:
        with Session(self.engine) as session:
            rows = session.execute(select(RetrievalCard)).scalars().all()
        return [self._retrieval_row_to_dict(r) for r in rows]

    def load_repair_rules(self) -> List[Dict[str, Any]]:
        with Session(self.engine) as session:
            rows = session.execute(select(RepairRule)).scalars().all()
        return [self._repair_row_to_dict(r) for r in rows]

    def build_repair_index(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        out: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row in self.load_repair_rules():
            out[(str(row.get("fic_id", "")), str(row.get("rule_id", "")))] = row
        return out

    def validate_integrity(self) -> Dict[str, Any]:
        core_cards = list(self.load_core_by_id().values())
        retrieval_cards = self.load_retrieval_cards()
        repair_rules = self.load_repair_rules()
        return validate_artifact_relations(
            core_cards=core_cards,
            retrieval_cards=retrieval_cards,
            repair_rules=repair_rules,
        )

    def retrieve_candidates(
        self,
        *,
        query: str,
        top_k: int = 5,
        domain: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if self.is_sqlite:
            fts = self._retrieve_candidates_fts(query=query, top_k=top_k, domain=domain, topic=topic)
            if fts:
                return fts
        return self._retrieve_candidates_overlap(query=query, top_k=top_k, domain=domain, topic=topic)

    def _retrieve_candidates_fts(
        self,
        *,
        query: str,
        top_k: int,
        domain: Optional[str],
        topic: Optional[str],
    ) -> List[Dict[str, Any]]:
        tokens = _tokenize(query)
        if not tokens:
            return []
        match_q = " OR ".join(tokens[:24])

        sql = text(
            "SELECT r.fic_id, r.domain, r.topic, r.title, r.summary, "
            "r.applicable_when_json, r.not_applicable_when_json, r.common_confusions_json, "
            "r.keywords_json, r.embedding_text, bm25(retrieval_fts) AS rank "
            "FROM retrieval_fts JOIN retrieval_cards r ON r.fic_id = retrieval_fts.fic_id "
            "WHERE retrieval_fts MATCH :match_q "
            "AND (:domain IS NULL OR r.domain = :domain) "
            "AND (:topic IS NULL OR r.topic = :topic) "
            "ORDER BY rank ASC "
            "LIMIT :k"
        )
        with self.engine.connect() as conn:
            rows = conn.execute(
                sql,
                {
                    "match_q": match_q,
                    "domain": domain,
                    "topic": topic,
                    "k": max(top_k, 1),
                },
            ).mappings().all()

        out = []
        for idx, r in enumerate(rows):
            rank = float(r.get("rank") or 0.0)
            # Use rank order for confidence to avoid bm25 sign/magnitude quirks across SQLite builds.
            score = 1.0 / float(idx + 1)
            out.append(
                {
                    "fic_id": r["fic_id"],
                    "domain": r["domain"],
                    "topic": r["topic"],
                    "title": r["title"],
                    "summary": r["summary"],
                    "applicable_when": _json_loads(r["applicable_when_json"], default=[]),
                    "not_applicable_when": _json_loads(r["not_applicable_when_json"], default=[]),
                    "common_confusions": _json_loads(r["common_confusions_json"], default=[]),
                    "keywords": _json_loads(r["keywords_json"], default=[]),
                    "embedding_text": r["embedding_text"],
                    "score": score,
                    "rank": rank,
                }
            )
        return out

    def _retrieve_candidates_overlap(
        self,
        *,
        query: str,
        top_k: int,
        domain: Optional[str],
        topic: Optional[str],
    ) -> List[Dict[str, Any]]:
        q_tokens = set(_tokenize(query))
        all_rows = self.load_retrieval_cards()
        scored = []
        for row in all_rows:
            if domain and str(row.get("domain", "")).lower() != domain.lower():
                continue
            if topic and str(row.get("topic", "")).lower() != topic.lower():
                continue
            text_blob = " ".join(
                [
                    str(row.get("title", "")),
                    str(row.get("summary", "")),
                    str(row.get("embedding_text", "")),
                    " ".join(str(x) for x in row.get("keywords", [])),
                ]
            )
            c_tokens = set(_tokenize(text_blob))
            overlap = len(q_tokens & c_tokens)
            denom = max(len(q_tokens), 1)
            score = overlap / denom
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, row in scored[:top_k]:
            payload = dict(row)
            payload["score"] = float(score)
            out.append(payload)
        return out


def build_store_from_files(
    *,
    db_url: str,
    core_path: Path,
    retrieval_path: Path,
    repair_path: Path,
) -> Dict[str, Any]:
    core_cards = _load_records(core_path)
    retrieval_cards = _load_records(retrieval_path)
    repair_rules = _load_records(repair_path)
    store = SQLAlchemyArtifactStore(db_url)
    return store.ingest_artifacts(
        core_cards=core_cards,
        retrieval_cards=retrieval_cards,
        repair_rules=repair_rules,
        validate_relations=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="v3 SQLAlchemy artifact store for core/retrieval/repair cards."
    )
    parser.add_argument("--db-url", required=True, help="SQLAlchemy DB URL, e.g. sqlite:///verifiquant_v3/data/v3_cards.db")
    parser.add_argument("--core", type=Path, help="fic_core JSON/JSONL")
    parser.add_argument("--retrieval", type=Path, help="fic_retrieval JSON/JSONL")
    parser.add_argument("--repair", type=Path, help="repair_rule JSON/JSONL")
    parser.add_argument("--query", help="Optional retrieval query text")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    store = SQLAlchemyArtifactStore(args.db_url)

    if args.core and args.retrieval and args.repair:
        stats = build_store_from_files(
            db_url=args.db_url,
            core_path=args.core,
            retrieval_path=args.retrieval,
            repair_path=args.repair,
        )
        print(
            "Store updated. "
            f"core={stats['core_count']} retrieval={stats['retrieval_count']} "
            f"repair={stats['repair_count']} diagnostic_rules={stats['diagnostic_rule_count']}"
        )

    if args.query:
        rows = store.retrieve_candidates(query=args.query, top_k=args.top_k)
        print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
