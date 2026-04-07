from __future__ import annotations

import json
import os
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


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_URL = os.environ.get(
    "VERIFIQUANT_DB_URL",
    "sqlite:///verifiquant/data/runs/demo_v2/cards.db",
)
DEFAULT_TOP_K = int(os.environ.get("VERIFIQUANT_TOP_K", "3"))
DEFAULT_SELECTOR_MODEL = os.environ.get("VERIFIQUANT_SELECTOR_MODEL", "gemini-2.5-flash")
DEFAULT_EXTRACTOR_MODEL = os.environ.get("VERIFIQUANT_EXTRACTOR_MODEL", "gemini-2.5-flash")
DEFAULT_JUDGE_MODEL = os.environ.get("VERIFIQUANT_JUDGE_MODEL", "gemini-2.5-flash")
DEFAULT_UPLOAD_DIR = os.path.join(ROOT_DIR, "data")
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "verifiquant", "data", "runs", "demo_v2")


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
    merged: Dict[str, Any] = {}
    if _clean_text(original_context):
        try:
            obj = json.loads(original_context)
            if isinstance(obj, dict):
                merged.update(obj)
        except Exception:
            pass
    for k, v in (updates or {}).items():
        kk = _clean_text(k)
        if kk:
            merged[kk] = v
    if not merged:
        return original_context
    return json.dumps(merged, ensure_ascii=False, indent=2)


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
        return render_template("index.html")

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
        payload = request.get_json(silent=True) or {}
        question = _clean_text(payload.get("question"))
        context = str(payload.get("context") or "")
        case_id = _clean_text(payload.get("case_id") or payload.get("question_id"))
        domain = _clean_text(payload.get("domain")) or None
        topic = _clean_text(payload.get("topic")) or None
        top_k = int(payload.get("top_k") or diag_api.top_k)
        m_min_top_score = float(payload.get("m_min_top_score") or diag_api.m_min_top_score)

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
            )
        except Exception as exc:
            return jsonify({"status": "error", "diagnostic_type": "Unknown", "message": str(exc)}), 500

        # Keep prompt echo for frontend memoryless repair UX.
        result["prompt"] = {"question": question, "context": context}
        return jsonify(result)

    @app.post("/api/diagnose/batch")
    def diagnose_batch() -> Any:
        payload = request.get_json(silent=True) or {}
        input_path = _clean_text(payload.get("input_path"))
        output_path = _clean_text(payload.get("output_path"))
        max_records = int(payload.get("max_records") or 0)
        default_domain = _clean_text(payload.get("domain")) or None
        default_topic = _clean_text(payload.get("topic")) or None
        top_k = int(payload.get("top_k") or diag_api.top_k)
        m_min_top_score = float(payload.get("m_min_top_score") or diag_api.m_min_top_score)
        include_results = bool(payload.get("include_results", True))

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

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=int(os.environ.get("PORT", 6222)), debug=True)
