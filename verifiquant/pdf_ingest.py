from __future__ import annotations

import io
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

from werkzeug.utils import secure_filename


class PdfIngestError(ValueError):
    """Raised when a PDF cannot be accepted or converted into usable text."""


@dataclass(frozen=True)
class PdfIngestConfig:
    max_bytes: int
    max_pages: int
    max_chars: int
    gcs_bucket: str = ""
    local_tmp_dir: str = ""
    keep_local_copy: bool = False
    local_ttl_seconds: int = 3600


@dataclass(frozen=True)
class PdfExtractionResult:
    filename: str
    text: str
    context_block: str
    size_bytes: int
    pages_total: int
    pages_read: int
    text_chars: int
    truncated: bool
    storage_uri: str | None = None
    storage_backend: str = "none"


def config_from_env(default_tmp_root: str | None = None) -> PdfIngestConfig:
    tmp_root = default_tmp_root or os.path.join(tempfile.gettempdir(), "verifiquant-pdf")
    return PdfIngestConfig(
        max_bytes=int(os.environ.get("VERIFIQUANT_MAX_PDF_BYTES", str(10 * 1024 * 1024))),
        max_pages=int(os.environ.get("VERIFIQUANT_MAX_PDF_PAGES", "20")),
        max_chars=int(os.environ.get("VERIFIQUANT_MAX_PDF_CHARS", "24000")),
        gcs_bucket=os.environ.get("VERIFIQUANT_GCS_BUCKET", "").strip(),
        local_tmp_dir=os.environ.get("VERIFIQUANT_PDF_TMP_DIR", tmp_root).strip(),
        keep_local_copy=os.environ.get("VERIFIQUANT_KEEP_LOCAL_PDF", "").lower() in ("1", "true", "yes"),
        local_ttl_seconds=int(os.environ.get("VERIFIQUANT_LOCAL_PDF_TTL_SECONDS", "3600")),
    )


def read_limited_pdf(stream: BinaryIO, filename: str, max_bytes: int) -> bytes:
    safe_name = secure_filename(filename or "")
    if not safe_name or not safe_name.lower().endswith(".pdf"):
        raise PdfIngestError("Only .pdf files are supported.")

    data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        mb = max_bytes / (1024 * 1024)
        raise PdfIngestError(f"PDF is too large. Limit is {mb:.1f} MiB.")
    if not data:
        raise PdfIngestError("PDF is empty.")
    if not data.startswith(b"%PDF-"):
        raise PdfIngestError("Uploaded file does not look like a PDF.")
    return data


def extract_pdf_text(data: bytes, filename: str, *, max_pages: int, max_chars: int) -> PdfExtractionResult:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise PdfIngestError("PDF support is not installed. Install pypdf to enable extraction.") from exc

    safe_name = secure_filename(filename or "document.pdf") or "document.pdf"
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise PdfIngestError(f"Failed to read PDF: {exc}") from exc

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise PdfIngestError("Encrypted PDFs are not supported.") from exc

    pages_total = len(reader.pages)
    if pages_total <= 0:
        raise PdfIngestError("PDF has no pages.")

    page_limit = max(1, min(max_pages, pages_total))
    chunks: list[str] = []
    truncated = pages_total > page_limit

    for index in range(page_limit):
        try:
            page_text = reader.pages[index].extract_text() or ""
        except Exception:
            page_text = ""
        page_text = _normalize_text(page_text)
        if page_text:
            chunks.append(f"[page {index + 1}]\n{page_text}")

    text = "\n\n".join(chunks).strip()
    if not text:
        raise PdfIngestError("No selectable text was found in the PDF.")

    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
        truncated = True

    note = []
    if truncated:
        note.append(f"truncated to {page_limit}/{pages_total} pages and {len(text)} chars")
    else:
        note.append(f"{page_limit}/{pages_total} pages")

    context_block = (
        f"[uploaded_pdf]\n"
        f"filename: {safe_name}\n"
        f"size_bytes: {len(data)}\n"
        f"extraction: {', '.join(note)}\n\n"
        f"{text}"
    )

    return PdfExtractionResult(
        filename=safe_name,
        text=text,
        context_block=context_block,
        size_bytes=len(data),
        pages_total=pages_total,
        pages_read=page_limit,
        text_chars=len(text),
        truncated=truncated,
    )


def persist_pdf(data: bytes, filename: str, config: PdfIngestConfig) -> tuple[str | None, str]:
    if config.gcs_bucket:
        return _upload_to_gcs(data, filename, config.gcs_bucket), "gcs"
    if config.keep_local_copy:
        return _save_local_tmp(data, filename, config), "local"
    return None, "none"


def cleanup_local_tmp(config: PdfIngestConfig) -> None:
    if not config.local_tmp_dir or config.local_ttl_seconds <= 0:
        return
    root = Path(config.local_tmp_dir)
    if not root.exists():
        return
    cutoff = datetime.now(timezone.utc).timestamp() - config.local_ttl_seconds
    for path in root.glob("*.pdf"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _object_name(filename: str) -> str:
    safe_name = secure_filename(filename or "document.pdf") or "document.pdf"
    today = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    return f"pdf-uploads/{today}/{uuid.uuid4().hex}-{safe_name}"


def _upload_to_gcs(data: bytes, filename: str, bucket_name: str) -> str:
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise PdfIngestError(
            "Cloud Storage support is not installed. Install google-cloud-storage or unset VERIFIQUANT_GCS_BUCKET."
        ) from exc

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(_object_name(filename))
    blob.upload_from_string(data, content_type="application/pdf")
    return f"gs://{bucket_name}/{blob.name}"


def _save_local_tmp(data: bytes, filename: str, config: PdfIngestConfig) -> str:
    root = Path(config.local_tmp_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / Path(_object_name(filename)).name
    path.write_bytes(data)
    return str(path)
