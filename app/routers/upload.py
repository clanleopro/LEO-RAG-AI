# app/routers/upload.py
"""
Secure PDF upload endpoint.

Security controls:
- Requires documents:write authorization.
- Never trusts UploadFile.filename.
- Streams uploads to disk (no full-file memory load).
- Validates extension, MIME type, and PDF magic bytes.
- Rejects encrypted, empty, oversized, or malformed PDFs.
- Enforces max files per request, max bytes per file, max total bytes.
- Writes to a temp file then atomically moves it.
- Removes partial files on failure.
- Enqueues background ingestion job (returns job ID immediately).
- Optional pluggable malware scan interface.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List

import fitz
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.auth import require_documents_write
from app.models.schemas import UploadResponse, UploadedFile
from app.services import config, ingest_service
from app.services.job_store import get_job_store

router = APIRouter(prefix="/api", tags=["upload"])
log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────
_PDF_MAGIC = b"%PDF-"
_SAFE_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/octet-stream",  # browsers sometimes send this for PDFs
}


# ─── Malware scan interface ───────────────────────────────────

def _scan_file(path: Path) -> None:
    """
    Optional malware scanning hook.

    To enable ClamAV scanning:
      1. Install ClamAV and start the clamd daemon.
      2. Install pyclamd: pip install pyclamd
      3. Set CLAMAV_SOCKET in .env to the clamd socket path or HTTP URL.

    Example pyclamd integration:
        import pyclamd
        cd = pyclamd.ClamdUnixSocket(config.ENV.CLAMAV_SOCKET)
        result = cd.scan_file(str(path))
        if result:
            raise HTTPException(422, f"Malware detected: {result}")
    """
    socket = config.ENV.CLAMAV_SOCKET
    if not socket:
        return  # Scanning not configured

    try:
        import pyclamd  # type: ignore[import]
        if socket.startswith("http"):
            cd = pyclamd.ClamdNetworkSocket(socket)
        else:
            cd = pyclamd.ClamdUnixSocket(socket)
        result = cd.scan_file(str(path))
        if result:
            log.warning("Malware detected in upload: %s", path.name)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="The uploaded file failed malware scanning and was rejected.",
            )
    except ImportError:
        log.warning("CLAMAV_SOCKET is set but pyclamd is not installed. Skipping scan.")
    except HTTPException:
        raise
    except Exception as exc:
        log.error("Malware scan failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Malware scanning service is unavailable. Upload rejected.",
        )


# ─── Validation helpers ────────────────────────────────────────

def _resolve_safe_dest(filename: str) -> Path:
    """
    Return a safe destination path inside SOURCE_PDFS.
    Prevents path traversal on both Linux and Windows.
    """
    dest = (config.SOURCE_PDFS / filename).resolve()
    if not str(dest).startswith(str(config.SOURCE_PDFS.resolve())):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename.",
        )
    return dest


def _validate_pdf(tmp_path: Path, original_name: str) -> int:
    """
    Validate the file is a non-empty, non-encrypted, well-formed PDF
    within page-count limits.

    Returns the page count on success.
    Raises HTTPException on failure.
    Raises ingest_service.IngestionError on controlled failures.
    """
    # Check magic bytes
    with open(tmp_path, "rb") as fh:
        header = fh.read(8)

    if not header.startswith(_PDF_MAGIC):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"'{original_name}' does not appear to be a valid PDF file.",
        )

    # Open with PyMuPDF
    try:
        doc = fitz.open(str(tmp_path))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"'{original_name}' is malformed or cannot be opened.",
        )

    with doc:
        if doc.is_encrypted:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"'{original_name}' is password-protected. "
                    "Please remove the password before uploading."
                ),
            )
        if doc.page_count == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"'{original_name}' has no pages.",
            )
        max_pages = config.ENV.MAX_PDF_PAGES
        if doc.page_count > max_pages:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"'{original_name}' has {doc.page_count} pages, which exceeds "
                    f"the maximum of {max_pages}. Please split the document."
                ),
            )
        return doc.page_count


# ─── Upload endpoint ──────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_files(
    files: List[UploadFile] = File(...),
    _role: str = Depends(require_documents_write),
) -> UploadResponse:
    """
    Upload one or more PDF files and queue them for background ingestion.

    Returns immediately with a job_id for each file to track ingestion progress.
    Requires the `documents:write` scope.
    """
    cfg = config.ENV
    max_files = cfg.MAX_UPLOAD_FILES
    max_file_bytes = cfg.MAX_FILE_BYTES
    max_total_bytes = cfg.MAX_TOTAL_REQUEST_BYTES

    if len(files) > max_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many files. Maximum {max_files} files per request.",
        )

    uploaded_files: List[UploadedFile] = []
    total_bytes_read = 0
    tmp_files: List[Path] = []

    try:
        for upload in files:
            # ── Extension check ──────────────────────────────
            original_name = (upload.filename or "unknown.pdf").strip()
            ext = Path(original_name).suffix.lower()
            if ext != ".pdf":
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Only PDF files are accepted. Got '{ext or 'no extension'}' for '{original_name}'.",
                )

            # ── Content-type check ───────────────────────────
            content_type = (upload.content_type or "").lower().split(";")[0].strip()
            if content_type and content_type not in _SAFE_CONTENT_TYPES:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid content type: '{content_type}'. Expected application/pdf.",
                )

            # ── Stream to temp file ──────────────────────────
            tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=".pdf", prefix="leo_upload_")
            tmp_path = Path(tmp_path_str)
            tmp_files.append(tmp_path)

            try:
                file_bytes = 0
                hasher = hashlib.sha256()

                with os.fdopen(tmp_fd, "wb") as tmp_fh:
                    while True:
                        chunk = await upload.read(65536)  # 64 KB chunks
                        if not chunk:
                            break
                        file_bytes += len(chunk)
                        total_bytes_read += len(chunk)

                        if file_bytes > max_file_bytes:
                            raise HTTPException(
                                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                detail=(
                                    f"File '{original_name}' exceeds the maximum "
                                    f"size of {max_file_bytes // (1024*1024)} MB."
                                ),
                            )
                        if total_bytes_read > max_total_bytes:
                            raise HTTPException(
                                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                detail=(
                                    f"Total upload size exceeds the maximum "
                                    f"of {max_total_bytes // (1024*1024)} MB."
                                ),
                            )

                        tmp_fh.write(chunk)
                        hasher.update(chunk)

                if file_bytes == 0:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"'{original_name}' is empty.",
                    )

                # ── Validate PDF ─────────────────────────────
                page_count = _validate_pdf(tmp_path, original_name)

                # ── Malware scan ─────────────────────────────
                _scan_file(tmp_path)

                # ── Generate safe storage name ────────────────
                doc_id = str(uuid.uuid4())
                # Sanitize display name: keep only alphanumeric, dash, underscore, dot
                safe_stem = re.sub(r"[^\w\-.]", "_", Path(original_name).stem)[:64]
                storage_name = f"{doc_id[:8]}_{safe_stem}.pdf"

                # ── Atomic move to destination ────────────────
                dest = _resolve_safe_dest(storage_name)
                if dest.exists():
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=f"A file with storage name '{storage_name}' already exists.",
                    )
                shutil.move(str(tmp_path), str(dest))
                tmp_files.remove(tmp_path)  # moved; no longer needs cleanup

                # ── Queue background ingestion ────────────────
                store = get_job_store()
                job_id = store.submit(
                    ingest_service.ingest_pdf,
                    dest,
                    idempotency_key=f"ingest:{doc_id}",
                    timeout_seconds=600.0,
                    doc_id=doc_id,
                )

                log.info(
                    "UPLOAD ACCEPTED: original='%s' storage='%s' doc_id=%s pages=%d bytes=%d job_id=%s",
                    original_name,
                    storage_name,
                    doc_id,
                    page_count,
                    file_bytes,
                    job_id,
                )

                uploaded_files.append(UploadedFile(
                    doc_id=doc_id,
                    original_filename=original_name,
                    size_bytes=file_bytes,
                    page_count=page_count,
                    job_id=job_id,
                ))

            except HTTPException:
                raise
            except Exception as exc:
                log.error("Upload processing failed for '%s': %s", original_name, exc)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="An error occurred while processing the upload.",
                )

    finally:
        # Clean up any remaining temp files (partial uploads / failures)
        for tmp in tmp_files:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    return UploadResponse(uploaded=uploaded_files)


# ─── Import re for sanitization ──────────────────────────────
import re  # noqa: E402 (placed here to keep top of file clean)
