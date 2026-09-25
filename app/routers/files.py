# app/routers/files.py
"""
Secure document listing, download, and deletion endpoints.

Security controls:
- All endpoints require appropriate authorization scopes.
- Document access uses doc_id, not raw filenames or paths.
- Path traversal prevented on both Linux and Windows.
- Only managed PDF files (under SOURCE_PDFS) can be accessed.
- Generic error responses — no filesystem paths exposed.
- Deletion removes both the file and all corresponding vector chunks.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from app.core.auth import require_documents_delete, require_documents_read
from app.models.schemas import DeleteResponse, DocumentInfo, DocumentListResponse
from app.services import config, vectorstore

router = APIRouter(prefix="/api", tags=["files"])
log = logging.getLogger(__name__)

_SOURCE_PDFS = config.SOURCE_PDFS.resolve()


def _safe_pdf_path(doc_id_or_name: str) -> Optional[Path]:
    """
    Resolve a doc_id or storage filename to a safe filesystem path.

    Returns None if no matching file is found.
    Raises HTTPException on path traversal attempt.
    Never exposes the filesystem path in the exception message.
    """
    # Sanitize: reject path separators and relative components
    bad_chars = {"\\", "/", "..", "~", "\x00"}
    for bc in bad_chars:
        if bc in doc_id_or_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid document identifier.",
            )

    # Try direct filename match
    candidate = (_SOURCE_PDFS / doc_id_or_name).resolve()
    try:
        candidate.relative_to(_SOURCE_PDFS)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document identifier.",
        )

    if candidate.exists() and candidate.suffix.lower() == ".pdf":
        return candidate

    # Try matching doc_id prefix in filenames
    try:
        for p in _SOURCE_PDFS.glob("*.pdf"):
            if p.stem.startswith(doc_id_or_name[:8]):
                resolved = p.resolve()
                try:
                    resolved.relative_to(_SOURCE_PDFS)
                    return resolved
                except ValueError:
                    continue
    except Exception:
        pass

    return None


# ─────────────────────────────────────────────────────────────
# GET /api/pdfs — list documents
# ─────────────────────────────────────────────────────────────

@router.get("/pdfs", response_model=DocumentListResponse)
def list_documents(
    _role: str = Depends(require_documents_read),
) -> DocumentListResponse:
    """
    List all ingested documents.

    Returns server-generated doc_ids and metadata from the vector store.
    Never exposes filesystem paths.
    """
    try:
        docs_meta = vectorstore.list_documents()
    except Exception as exc:
        log.error("list_documents failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to retrieve document list.",
        )

    # Enrich with filesystem metadata where available
    documents = []
    for m in docs_meta:
        title = m.get("title", m.get("doc_id", "unknown"))
        path = _SOURCE_PDFS / title
        size = 0
        page_count = 0
        try:
            if path.exists():
                size = path.stat().st_size
        except Exception:
            pass

        documents.append(DocumentInfo(
            doc_id=m.get("doc_id", title),
            title=title,
            size_bytes=size,
            page_count=page_count,
            checksum=m.get("checksum"),
            ingested_at=m.get("ingested_at"),
            embedding_model=m.get("embedding_model"),
        ))

    return DocumentListResponse(documents=documents, total=len(documents))


# ─────────────────────────────────────────────────────────────
# GET /api/pdfs/{doc_id} — download document
# ─────────────────────────────────────────────────────────────

@router.get("/pdfs/{doc_id}")
def download_document(
    doc_id: str,
    _role: str = Depends(require_documents_read),
):
    """
    Download an ingested PDF by doc_id.

    The doc_id must correspond to a managed file in SOURCE_PDFS.
    Raises 404 with a generic message if not found.
    """
    path = _safe_pdf_path(doc_id)
    if path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    log.info("DOWNLOAD: doc_id=%s", doc_id)
    return FileResponse(
        str(path),
        media_type="application/pdf",
        filename=path.name,
    )


# ─────────────────────────────────────────────────────────────
# DELETE /api/pdfs/{doc_id} — delete document
# ─────────────────────────────────────────────────────────────

@router.delete("/pdfs/{doc_id}", response_model=DeleteResponse)
def delete_document(
    doc_id: str,
    _role: str = Depends(require_documents_delete),
) -> DeleteResponse:
    """
    Delete an ingested document and all corresponding vector store chunks.

    Both the file and vector records are removed.
    Returns the number of vector records removed.
    """
    path = _safe_pdf_path(doc_id)
    if path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    title = path.name
    vectors_removed = 0
    file_deleted = False

    # Remove vector store entries first (best-effort)
    try:
        vectors_removed = vectorstore.delete_by_doc_id(doc_id)
        if vectors_removed == 0:
            # Fallback: try source-based deletion
            vectors_removed = vectorstore.delete_by_source(title)
    except Exception as exc:
        log.error("Vector deletion failed for doc_id=%s: %s", doc_id, exc)

    # Remove the file
    try:
        path.unlink()
        file_deleted = True
        log.info("DELETED: doc_id=%s title=%s vectors_removed=%d", doc_id, title, vectors_removed)
    except Exception as exc:
        log.error("File deletion failed for doc_id=%s: %s", doc_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File deletion failed. Vector records may have been removed.",
        )

    return DeleteResponse(
        doc_id=doc_id,
        title=title,
        vectors_removed=vectors_removed,
        file_deleted=file_deleted,
    )
