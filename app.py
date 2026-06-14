from __future__ import annotations

import glob
import json
import os
import time
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from verifiquant.card_store import SQLAlchemyArtifactStore
from verifiquant.pipeline.run_error_classification_pipeline import (
    ErrorClassificationAPI,
    create_genai_client_from_env,
)
from verifiquant.preprocessing.stage_transform import (
    _run_fic_code,
    apply_result_postprocess,
    get_transform_spec_for_choice,
    patch_spec_from_dict,
    verify_transform,
)


RATE_LIMIT_PER_HOUR = int(os.environ.get("VERIFIQUANT_RATE_LIMIT", "20"))


class _RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int = 3600):
        self._max = max_requests
        self._window = window_seconds
        self._hits: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            times = self._hits[key]
            times[:] = [t for t in times if now - t < self._window]
            if len(times) >= self._max:
                return False
            times.append(now)
            return True

    def remaining(self, key: str) -> int:
        now = time.monotonic()
        with self._lock:
            times = self._hits[key]
            times[:] = [t for t in times if now - t < self._window]
            return max(0, self._max - len(times))


_rate_limiter = _RateLimiter(RATE_LIMIT_PER_HOUR)


def _get_client_ip() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(ROOT_DIR, "verifiquant", "data", "runs")


# ── Demo run selection ────────────────────────────────────────────────────
# A "run" bundles a card store (cards.db) and a question bank from the SAME
# dataset. Deriving both from one run keeps repair cards coherent with the
# demo questions (every question maps to a card that carries repair rules).
#
# To swap datasets:
#   1. point VERIFIQUANT_DEMO_RUN at another dir under verifiquant/data/runs/
#      (the dir must contain a cards.db — at <run>/cards.db or <run>/fic/cards.db
#      — and a questions_*.jsonl, preferably questions_50.jsonl);
#   2. allowlist that run in .dockerignore so it ships into the container.
# Explicit VERIFIQUANT_DB_URL / VERIFIQUANT_DEMO_QUESTION_BANK still override.
DEFAULT_DEMO_RUN = os.environ.get("VERIFIQUANT_DEMO_RUN", "paper_v1")


def _resolve_run_db_path(run: str) -> str:
    """Locate a run's cards.db: <run>/cards.db, else <run>/fic/cards.db."""
    base = os.path.join(RUNS_DIR, run)
    candidates = (
        os.path.join(base, "cards.db"),
        os.path.join(base, "fic", "cards.db"),
    )
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    # Return the conventional location; a clear error surfaces at store init.
    return candidates[0]


def _resolve_run_question_bank(run: str) -> str:
    """Locate a run's question bank: questions_50.jsonl, else any questions_*.jsonl."""
    base = os.path.join(RUNS_DIR, run)
    preferred = os.path.join(base, "questions_50.jsonl")
    if os.path.exists(preferred):
        return preferred
    matches = sorted(glob.glob(os.path.join(base, "questions_*.jsonl")))
    return matches[0] if matches else preferred


def _resolve_run_showcase(run: str) -> str:
    """Locate a run's curated showcase set (demo_showcase.jsonl), if present."""
    return os.path.join(RUNS_DIR, run, "demo_showcase.jsonl")


DEFAULT_DB_PATH = _resolve_run_db_path(DEFAULT_DEMO_RUN)
DEFAULT_DB_URL = os.environ.get(
    "VERIFIQUANT_DB_URL",
    f"sqlite:///{DEFAULT_DB_PATH}",
)
DEFAULT_TOP_K = int(os.environ.get("VERIFIQUANT_TOP_K", "3"))
DEFAULT_SELECTOR_MODEL = os.environ.get("VERIFIQUANT_SELECTOR_MODEL", "gemini-2.5-flash")
DEFAULT_EXTRACTOR_MODEL = os.environ.get("VERIFIQUANT_EXTRACTOR_MODEL", "gemini-2.5-flash")
DEFAULT_JUDGE_MODEL = os.environ.get("VERIFIQUANT_JUDGE_MODEL", "gemini-2.5-flash")
DEFAULT_UPLOAD_DIR = os.path.join(ROOT_DIR, "data")
DEFAULT_OUTPUT_DIR = os.path.join(RUNS_DIR, DEFAULT_DEMO_RUN)
DEFAULT_DEMO_QUESTION_BANK = os.environ.get(
    "VERIFIQUANT_DEMO_QUESTION_BANK",
    _resolve_run_question_bank(DEFAULT_DEMO_RUN),
)
DEMO_MODE = os.environ.get("VERIFIQUANT_DEMO_MODE", "").lower() in ("1", "true", "yes")

# Control-group ("raw AI, no funnel") baseline models. Same brain as VerifiQuant's, but no
# FIC/funnel scaffolding — used purely for side-by-side comparison in the demo.
BASELINE_GEMINI_MODEL = os.environ.get("VERIFIQUANT_BASELINE_GEMINI", "gemini-2.5-flash")
BASELINE_GPT_MODEL = os.environ.get("GPT_FLASH", "gpt-5.2")
BASELINE_SYSTEM_PROMPT = (
    "You are a financial assistant. Using only the information in the question and context, "
    "compute the answer. Show brief working, then end with a single line 'Final answer: <number>'. "
    "Do not ask clarifying questions; make a reasonable assumption if needed."
)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    for line in open(path, "r", encoding="utf-8").read().splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _build_repaired_context(original_context: str, updates: Dict[str, Any]) -> str:
    clean_updates: Dict[str, Any] = {}
    for k, v in (updates or {}).items():
        kk = _clean_text(k)
        if kk:
            clean_updates[kk] = v

    original_text = original_context if original_context is not None else ""

    # Case 1: original context is a JSON object — merge updates into it (lossless).
    parsed_obj: Dict[str, Any] | None = None
    if _clean_text(original_text):
        try:
            obj = json.loads(original_text)
            if isinstance(obj, dict):
                parsed_obj = obj
        except Exception:
            parsed_obj = None

    if parsed_obj is not None:
        merged = dict(parsed_obj)
        merged.update(clean_updates)
        return json.dumps(merged, ensure_ascii=False, indent=2)

    # Case 2: original context is natural language (or non-dict JSON). NEVER discard it —
    # keep the original verbatim and append the clarified fields as a labeled JSON block so
    # the extractor sees both the original numbers and the user's answers.
    if not clean_updates:
        return original_text
    block = json.dumps(clean_updates, ensure_ascii=False, indent=2)
    if _clean_text(original_text):
        return f"{original_text.rstrip()}\n\n[clarified_fields]\n{block}"
    return block


def _resolve_local_path(path_value: str) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return (Path(ROOT_DIR) / p).resolve()


def _ensure_within_workspace(path: Path) -> None:
    root = Path(ROOT_DIR).resolve()
    target = path.resolve()
    if root not in target.parents and target != root:
        raise ValueError("path must be inside workspace")


def _load_records_file(path_value: str) -> List[Dict[str, Any]]:
    path = _resolve_local_path(path_value)
    _ensure_within_workspace(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported input format: {path}")


def _dump_records_file(path_value: str, rows: List[Dict[str, Any]]) -> str:
    path = _resolve_local_path(path_value)
    _ensure_within_workspace(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def create_app() -> Flask:
    _load_env_file()

    app = Flask(
        __name__,
        template_folder=os.path.join(ROOT_DIR, "templates"),
        static_folder=os.path.join(ROOT_DIR, "static"),
    )

    store = SQLAlchemyArtifactStore(DEFAULT_DB_URL)
    os.makedirs(DEFAULT_UPLOAD_DIR, exist_ok=True)
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

    try:
        client = create_genai_client_from_env()
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to initialize run_case API: {exc}. "
            "Please ensure GEMINI_API_KEY is available (in shell or .env)."
        ) from exc

    diag_api = ErrorClassificationAPI.from_db(
        db_url=DEFAULT_DB_URL,
        client=client,
        selector_model=DEFAULT_SELECTOR_MODEL,
        extractor_model=DEFAULT_EXTRACTOR_MODEL,
        judge_model=DEFAULT_JUDGE_MODEL,
        top_k=DEFAULT_TOP_K,
    )

    @app.get("/")
    def index() -> str:
        return render_template("home.html", active_page="home")

    @app.get("/demo")
    def demo() -> str:
        return render_template("demo.html", active_page="demo")

    @app.get("/about")
    def about() -> str:
        return render_template("about.html", active_page="about")

    @app.get("/api/cards/overview")
    def cards_overview() -> Any:
        rows = store.load_retrieval_cards()
        grouped: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            domain = _clean_text(row.get("domain")) or "unknown"
            topic = _clean_text(row.get("topic")) or "unknown"
            domain_entry = grouped.setdefault(domain, {"total": 0, "topics": {}})
            topic_entry = domain_entry["topics"].setdefault(topic, {"count": 0, "cards": []})
            topic_entry["count"] += 1
            domain_entry["total"] += 1
            topic_entry["cards"].append(
                {
                    "fic_id": row.get("fic_id"),
                    "title": row.get("title"),
                    "summary": row.get("summary"),
                    "keywords": row.get("keywords", []),
                }
            )

        return jsonify(
            {
                "status": "ok",
                "domain_count": len(grouped),
                "card_count": len(rows),
                "grouped": grouped,
            }
        )

    @app.get("/api/demo/questions")
    def demo_questions() -> Any:
        want_showcase = _clean_text(request.args.get("showcase")) in ("1", "true", "yes")
        default_path = _resolve_run_showcase(DEFAULT_DEMO_RUN) if want_showcase else DEFAULT_DEMO_QUESTION_BANK
        path_value = _clean_text(request.args.get("path")) or default_path
        try:
            rows = _load_records_file(path_value)
        except Exception as exc:
            return jsonify({"status": "error", "message": f"failed to load question bank: {exc}"}), 400

        questions: List[Dict[str, Any]] = []
        for idx, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                continue
            case_id = _clean_text(row.get("case_id") or row.get("question_id") or idx)
            question = _clean_text(row.get("question"))
            if not question:
                continue
            questions.append(
                {
                    "index": idx,
                    "case_id": case_id,
                    "question_id": row.get("question_id"),
                    "article_title": row.get("article_title"),
                    "function_id": row.get("function_id"),
                    "level": row.get("level"),
                    "question": question,
                    "context": row.get("context") or "",
                    "ground_truth": row.get("ground_truth"),
                    "demo_label": row.get("demo_label"),
                    "expected_class": row.get("expected_class"),
                    "demo_note": row.get("demo_note"),
                    "expected_diagnostic": row.get("expected_diagnostic"),
                    "expected_behavior": row.get("expected_behavior"),
                }
            )

        resolved = _resolve_local_path(path_value)
        return jsonify(
            {
                "status": "ok",
                "path": str(resolved),
                "run": DEFAULT_DEMO_RUN,
                "source": os.path.relpath(resolved, RUNS_DIR),
                "count": len(questions),
                "questions": questions,
            }
        )

    @app.post("/api/rag/retrieve")
    def rag_retrieve() -> Any:
        payload = request.get_json(silent=True) or {}
        question = _clean_text(payload.get("question"))
        context = str(payload.get("context") or "")
        domain = _clean_text(payload.get("domain")) or None
        topic = _clean_text(payload.get("topic")) or None
        top_k = int(payload.get("top_k") or DEFAULT_TOP_K)

        if not question:
            return jsonify({"status": "error", "message": "question is required"}), 400

        candidates = diag_api.retrieve(
            question=question,
            context=context,
            top_k=top_k,
            domain=domain,
            topic=topic,
        )
        return jsonify(
            {
                "status": "ok",
                "question": question,
                "context": context,
                "top_k": top_k,
                "domain": domain,
                "topic": topic,
                "candidates": candidates,
            }
        )

    @app.post("/api/files/upload")
    def upload_file() -> Any:
        if DEMO_MODE:
            return jsonify({"status": "error", "message": "File upload is disabled in demo mode."}), 403

        incoming = request.files.get("file")
        if incoming is None or not incoming.filename:
            return jsonify({"status": "error", "message": "file is required"}), 400

        filename = secure_filename(incoming.filename)
        if not filename:
            return jsonify({"status": "error", "message": "invalid filename"}), 400
        if not (filename.endswith(".jsonl") or filename.endswith(".json")):
            return jsonify({"status": "error", "message": "only .jsonl/.json is supported"}), 400

        stem, ext = os.path.splitext(filename)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_name = f"{stem}_{ts}{ext}"
        save_path = os.path.join(DEFAULT_UPLOAD_DIR, saved_name)
        incoming.save(save_path)

        return jsonify(
            {
                "status": "ok",
                "filename": saved_name,
                "saved_path": save_path,
                "relative_path": os.path.relpath(save_path, ROOT_DIR),
            }
        )

    @app.get("/api/files/history")
    def files_history() -> Any:
        directory = _clean_text(request.args.get("dir")) or os.path.relpath(DEFAULT_OUTPUT_DIR, ROOT_DIR)
        suffix = _clean_text(request.args.get("suffix")) or ".jsonl"
        try:
            target_dir = _resolve_local_path(directory)
            _ensure_within_workspace(target_dir)
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        if not target_dir.exists():
            return jsonify({"status": "ok", "directory": str(target_dir), "files": []})

        files: List[Dict[str, Any]] = []
        for p in target_dir.iterdir():
            if not p.is_file():
                continue
            if suffix and not p.name.endswith(suffix):
                continue
            stat = p.stat()
            files.append(
                {
                    "name": p.name,
                    "path": str(p),
                    "relative_path": os.path.relpath(str(p), ROOT_DIR),
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                }
            )
        files.sort(key=lambda x: x["modified_at"], reverse=True)
        return jsonify({"status": "ok", "directory": str(target_dir), "files": files})

    @app.get("/api/files/summary")
    def file_summary() -> Any:
        path_value = _clean_text(request.args.get("path"))
        if not path_value:
            return jsonify({"status": "error", "message": "path is required"}), 400
        try:
            rows = _load_records_file(path_value)
        except Exception as exc:
            return jsonify({"status": "error", "message": f"failed to load file: {exc}"}), 400

        status_counts: Dict[str, int] = {}
        diag_counts: Dict[str, int] = {}
        sample: List[Dict[str, Any]] = []
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            s = _clean_text(row.get("status")) or "unknown"
            d = _clean_text(row.get("diagnostic_type")) or "Unknown"
            status_counts[s] = status_counts.get(s, 0) + 1
            diag_counts[d] = diag_counts.get(d, 0) + 1
            if idx < 5:
                sample.append(
                    {
                        "case_id": row.get("case_id") or row.get("question_id"),
                        "status": row.get("status"),
                        "diagnostic_type": row.get("diagnostic_type"),
                        "fic_id": row.get("fic_id"),
                    }
                )

        return jsonify(
            {
                "status": "ok",
                "path": str(_resolve_local_path(path_value)),
                "records": len(rows),
                "status_counts": status_counts,
                "diagnostic_counts": diag_counts,
                "sample": sample,
            }
        )

    @app.get("/api/files/download")
    def file_download() -> Any:
        path_value = _clean_text(request.args.get("path"))
        if not path_value:
            return jsonify({"status": "error", "message": "path is required"}), 400
        try:
            p = _resolve_local_path(path_value)
            _ensure_within_workspace(p)
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        if not p.exists() or not p.is_file():
            return jsonify({"status": "error", "message": "file not found"}), 404
        return send_file(str(p), as_attachment=True, download_name=p.name)

    @app.post("/api/diagnose")
    def diagnose() -> Any:
        ip = _get_client_ip()
        if not _rate_limiter.is_allowed(ip):
            return jsonify({
                "status": "error",
                "message": f"Rate limit exceeded. Max {RATE_LIMIT_PER_HOUR} diagnoses per hour.",
                "remaining": 0,
            }), 429

        payload = request.get_json(silent=True) or {}
        question = _clean_text(payload.get("question"))
        context = str(payload.get("context") or "")
        case_id = _clean_text(payload.get("case_id") or payload.get("question_id"))
        domain = _clean_text(payload.get("domain")) or None
        topic = _clean_text(payload.get("topic")) or None
        top_k = int(payload.get("top_k") or diag_api.top_k)
        m_min_top_score = float(payload.get("m_min_top_score") or diag_api.m_min_top_score)
        debug_sanity = bool(payload.get("debug_sanity", True))

        if not question:
            return jsonify({"status": "error", "message": "question is required"}), 400

        row: Dict[str, Any] = {
            "case_id": case_id,
            "question": question,
            "context": context,
        }
        if domain:
            row["domain"] = domain
        if topic:
            row["topic"] = topic

        try:
            result = diag_api.diagnose_row(
                row,
                top_k=top_k,
                m_min_top_score=m_min_top_score,
                debug_sanity=debug_sanity,
            )
        except Exception as exc:
            return jsonify({"status": "error", "diagnostic_type": "Unknown", "message": str(exc)}), 500

        # Keep prompt echo for frontend memoryless repair UX.
        result["prompt"] = {"question": question, "context": context}
        return jsonify(result)

    @app.get("/api/rate-limit")
    def rate_limit_status() -> Any:
        ip = _get_client_ip()
        return jsonify({
            "status": "ok",
            "remaining": _rate_limiter.remaining(ip),
            "limit": RATE_LIMIT_PER_HOUR,
        })

    @app.post("/api/diagnose/batch")
    def diagnose_batch() -> Any:
        ip = _get_client_ip()
        if not _rate_limiter.is_allowed(ip):
            return jsonify({
                "status": "error",
                "message": f"Rate limit exceeded. Max {RATE_LIMIT_PER_HOUR} diagnoses per hour.",
                "remaining": 0,
            }), 429

        payload = request.get_json(silent=True) or {}
        input_path = _clean_text(payload.get("input_path"))
        output_path = _clean_text(payload.get("output_path"))
        max_records = int(payload.get("max_records") or 0)
        default_domain = _clean_text(payload.get("domain")) or None
        default_topic = _clean_text(payload.get("topic")) or None
        top_k = int(payload.get("top_k") or diag_api.top_k)
        m_min_top_score = float(payload.get("m_min_top_score") or diag_api.m_min_top_score)
        include_results = bool(payload.get("include_results", True))
        debug_sanity = bool(payload.get("debug_sanity", False))

        if not input_path:
            return jsonify({"status": "error", "message": "input_path is required"}), 400

        try:
            rows = _load_records_file(input_path)
        except Exception as exc:
            return jsonify({"status": "error", "message": f"Failed to load input: {exc}"}), 400

        if max_records > 0:
            rows = rows[:max_records]

        out: List[Dict[str, Any]] = []
        status_counts: Dict[str, int] = {}
        diag_counts: Dict[str, int] = {}

        for idx, src in enumerate(rows, 1):
            row = dict(src) if isinstance(src, dict) else {}
            question = _clean_text(row.get("question"))
            context = str(row.get("context") or "")
            case_id = _clean_text(row.get("case_id") or row.get("question_id") or idx)
            domain = _clean_text(row.get("domain")) or default_domain
            topic = _clean_text(row.get("topic")) or default_topic

            if not question:
                result = {
                    "case_id": case_id,
                    "status": "error",
                    "diagnostic_type": "Unknown",
                    "message": "question is required",
                    "prompt": {"question": question, "context": context},
                }
            else:
                run_row: Dict[str, Any] = {
                    "case_id": case_id,
                    "question": question,
                    "context": context,
                }
                if domain:
                    run_row["domain"] = domain
                if topic:
                    run_row["topic"] = topic
                try:
                    result = diag_api.diagnose_row(
                        run_row,
                        top_k=top_k,
                        m_min_top_score=m_min_top_score,
                        debug_sanity=debug_sanity,
                    )
                except Exception as exc:
                    result = {
                        "case_id": case_id,
                        "status": "error",
                        "diagnostic_type": "Unknown",
                        "message": str(exc),
                    }
                result["prompt"] = {"question": question, "context": context}

            status_key = _clean_text(result.get("status")) or "unknown"
            diag_key = _clean_text(result.get("diagnostic_type")) or "Unknown"
            status_counts[status_key] = status_counts.get(status_key, 0) + 1
            diag_counts[diag_key] = diag_counts.get(diag_key, 0) + 1
            out.append(result)

        written_to = None
        if output_path:
            try:
                written_to = _dump_records_file(output_path, out)
            except Exception as exc:
                return jsonify(
                    {
                        "status": "error",
                        "message": f"Failed to write output: {exc}",
                        "processed": len(out),
                    }
                ), 500

        response: Dict[str, Any] = {
            "status": "ok",
            "input_path": str(_resolve_local_path(input_path)),
            "output_path": written_to,
            "processed": len(out),
            "top_k": top_k,
            "m_min_top_score": m_min_top_score,
            "status_counts": status_counts,
            "diagnostic_counts": diag_counts,
        }
        if include_results:
            response["results"] = out
        return jsonify(response)

    @app.post("/api/repair/compose")
    def compose_repair() -> Any:
        payload = request.get_json(silent=True) or {}
        question = _clean_text(payload.get("question"))
        context = str(payload.get("context") or "")
        updates = payload.get("updates") or {}
        if not isinstance(updates, dict):
            return jsonify({"status": "error", "message": "updates must be an object"}), 400

        repaired_context = _build_repaired_context(context, updates)
        return jsonify(
            {
                "status": "ok",
                "repaired_question": question,
                "repaired_context": repaired_context,
            }
        )

    @app.post("/api/transform/apply")
    def transform_apply() -> Any:
        """Apply a Verifiable Atomic Transform for an I-class clarification choice.

        Given the chosen option's value, the FIC, and the inputs already extracted by a
        prior diagnosis, this loads the option's transform_spec, re-runs the FIC code to
        get result_old, verifies the transform (AST guard + numerical invariant), and
        returns result_old → result_new only if verification passes. No LLM call.
        """
        payload = request.get_json(silent=True) or {}
        fic_id = _clean_text(payload.get("fic_id"))
        chosen_value = _clean_text(payload.get("chosen_value"))
        inputs = payload.get("inputs") or {}
        if not fic_id or not chosen_value:
            return jsonify({"status": "error", "message": "fic_id and chosen_value are required"}), 400
        if not isinstance(inputs, dict):
            return jsonify({"status": "error", "message": "inputs must be an object"}), 400

        core = store.load_core_by_id().get(fic_id)
        if not core:
            return jsonify({"status": "error", "message": f"unknown fic_id: {fic_id}"}), 404

        # Resolve the transform spec: prefer the repair-rule transform_map (the documented
        # runtime contract), fall back to the core semantic_hints option.
        spec = None
        rule_id = None
        for rule in store.load_repair_rules():
            if str(rule.get("fic_id", "")) != fic_id:
                continue
            candidate = get_transform_spec_for_choice(rule, chosen_value)
            if candidate is not None:
                spec, rule_id = candidate, rule.get("rule_id")
                break
        if spec is None:
            for hint in core.get("semantic_hints", []) or []:
                for opt in hint.get("options", []) or []:
                    if isinstance(opt, dict) and str(opt.get("value", "")) == chosen_value and opt.get("transform_spec"):
                        spec = patch_spec_from_dict(opt["transform_spec"])
                        break
                if spec is not None:
                    break
        if spec is None:
            return jsonify({"status": "error", "message": f"no transform declared for choice '{chosen_value}'"}), 400
        if getattr(spec, "patch_type", "") != "result_postprocess":
            return jsonify({"status": "error", "message": "only result_postprocess transforms are supported at runtime"}), 400

        execution = core.get("execution", {}) or {}
        try:
            result_old = _run_fic_code(execution.get("code", ""), execution.get("entrypoint", ""), inputs)
            verify = verify_transform(spec, core, [inputs])
            if not verify.passed:
                return jsonify({
                    "status": "rejected",
                    "message": "Transform failed numerical verification; result NOT changed.",
                    "invariant": spec.invariant,
                    "invariant_check": verify.invariant_check,
                    "error": verify.error,
                }), 200
            result_new = apply_result_postprocess(result_old, spec, inputs)
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

        return jsonify({
            "status": "ok",
            "verified": True,
            "fic_id": fic_id,
            "fic_name": core.get("name") or core.get("article_title"),
            "rule_id": rule_id,
            "chosen_value": chosen_value,
            "result_old": result_old,
            "result_new": result_new,
            "result_expr": spec.result_expr,
            "invariant": spec.invariant,
            "invariant_check": verify.invariant_check,
            "numerical_samples": verify.numerical_samples,
            "python_code": execution.get("code", ""),
            "entrypoint": execution.get("entrypoint", ""),
            "inputs": inputs,
        })

    @app.post("/api/baseline")
    def baseline() -> Any:
        """Control group: answer with a raw LLM (no funnel, no FIC, generic prompt).

        Lets the demo show a vanilla assistant silently committing to an assumption while
        VerifiQuant intercepts. Provider is chosen in the UI (gemini | gpt).
        """
        ip = _get_client_ip()
        if not _rate_limiter.is_allowed(ip):
            return jsonify({"status": "error", "message": "Rate limit exceeded."}), 429

        payload = request.get_json(silent=True) or {}
        question = _clean_text(payload.get("question"))
        context = str(payload.get("context") or "")
        provider = (_clean_text(payload.get("provider")) or "gemini").lower()
        if not question:
            return jsonify({"status": "error", "message": "question is required"}), 400

        user_prompt = f"Question:\n{question}\n\nContext:\n{context or '(none)'}"
        try:
            if provider == "gpt":
                from openai import OpenAI

                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    return jsonify({"status": "error", "message": "OPENAI_API_KEY not set"}), 400
                oa = OpenAI(api_key=api_key)
                resp = oa.chat.completions.create(
                    model=BASELINE_GPT_MODEL,
                    messages=[
                        {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                answer = resp.choices[0].message.content
                model = BASELINE_GPT_MODEL
            else:
                # Gemini: fold the system prompt into the contents (no schema, free-form).
                prompt = f"{BASELINE_SYSTEM_PROMPT}\n\n{user_prompt}"
                resp = client.models.generate_content(model=BASELINE_GEMINI_MODEL, contents=prompt)
                answer = resp.text
                model = BASELINE_GEMINI_MODEL
                provider = "gemini"
        except Exception as exc:
            return jsonify({"status": "error", "provider": provider, "message": str(exc)}), 500

        return jsonify({
            "status": "ok",
            "provider": provider,
            "model": model,
            "answer": _clean_text(answer),
        })

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=int(os.environ.get("PORT", 6222)), debug=True)
