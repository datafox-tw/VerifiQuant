from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - runtime dependency guard
    genai = None
    genai_types = None


@dataclass
class ConversionInput:
    source_meta: Dict[str, Any]
    function: str
    python_solution: str
    context: str
    question: str


def load_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported JSON payload type in {path}")


def dump_records(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = list(rows)
    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as fh:
            for row in payload:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def to_conversion_input(record: Dict[str, Any]) -> ConversionInput:
    return ConversionInput(
        source_meta={
            "function_id": record.get("function_id"),
            "article_title": record.get("article_title"),
            "source": record.get("source"),
            "question_id": record.get("question_id"),
            "difficulty": record.get("difficulty"),
            "level": record.get("level"),
        },
        function=str(record.get("function", "") or ""),
        python_solution=str(record.get("python_solution", "") or ""),
        context=str(record.get("context", "") or ""),
        question=str(record.get("question", "") or ""),
    )


def safe_id(raw: str, *, default_prefix: str = "fic") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", str(raw or "").strip()).strip("_").lower()
    if not cleaned:
        return f"{default_prefix}_generated"
    if not cleaned.startswith("fic_"):
        return f"fic_{cleaned}"
    return cleaned


def normalize_label(raw: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(raw or "").strip().lower()).strip("_")


def normalize_input_type(raw: str) -> str:
    t = normalize_label(raw)
    if not t:
        return "number"

    aliases = {
        "float": "number",
        "double": "number",
        "numeric": "number",
        "int": "integer",
        "list": "array[number]",
        "array": "array[number]",
        "list_number": "array[number]",
        "array_number": "array[number]",
        "array_float": "array[number]",
        "array_int": "array[number]",
        "bool": "boolean",
        "str": "string",
    }
    if t in aliases:
        return aliases[t]
    if t in {"number", "integer", "boolean", "string", "array[number]"}:
        return t
    if "array" in t or "list" in t:
        return "array[number]"
    return "number"


def require_compute_code(code: str) -> str:
    c = str(code or "").strip()
    if "def compute(inputs)" not in c:
        raise ValueError("execution.code must define compute(inputs)")
    return c


def require_gemini_client() -> Any:
    if genai is None:
        raise RuntimeError(
            "Missing dependency: google.genai import failed. Please install compatible google-genai package."
        )
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in environment.")
    return genai.Client(api_key=api_key)


def call_gemini_json(
    client: Any,
    *,
    model: str,
    prompt: str,
    schema: Any,
) -> Dict[str, Any]:
    if genai_types is None:
        raise RuntimeError("google.genai.types is unavailable in this environment.")
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    try:
        return json.loads(response.text)
    except json.JSONDecodeError as err:
        raise ValueError(f"Gemini returned invalid JSON: {response.text}") from err


def optional_str(value: Any) -> Optional[str]:
    s = str(value or "").strip()
    return s or None
