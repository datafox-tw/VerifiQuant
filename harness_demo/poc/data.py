"""Paths and loaders for the 250-question PoC. Gold fields are read only by grading code."""

from __future__ import annotations

import ast
import json
import sqlite3
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "verifiquant/data/runs/paper_v2_250"
QUESTIONS = RUN / "questions_250.jsonl"
CARDS_DB = RUN / "fic/cards.db"
HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "sample_50.json"
AUTO_CARDS = HERE / "cards_auto.jsonl"
RUNS = HERE / "runs"

GOLD_FIELDS = ("ground_truth", "python_solution")


@lru_cache(maxsize=1)
def questions() -> dict[str, dict]:
    return {r["question_id"]: r for r in map(json.loads, QUESTIONS.read_text().splitlines())}


def text_of(q: dict) -> str:
    """What the harness sees: context then question, never the gold fields."""
    return f"{q['context'].strip()}\n\n{q['question'].strip()}"


def fic_id(q: dict) -> str:
    return "fic_" + q["function_id"].replace("-", "_")


def library_function(q: dict) -> tuple[str, ast.FunctionDef]:
    tree = ast.parse(q["function"])
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    return q["function"], fn


@lru_cache(maxsize=1)
def topics() -> dict[str, str]:
    con = sqlite3.connect(CARDS_DB)
    return {fid: topic for fid, topic in con.execute("select fic_id, topic from core_cards")}


def sample_ids() -> list[str]:
    return json.loads(SAMPLE.read_text())["question_ids"]
