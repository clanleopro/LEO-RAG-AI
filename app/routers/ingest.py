# app/routers/ingest.py
"""
Ingest trigger endpoints.

Security controls:
- Requires documents:write authorization.
- Re-ingestion removes stale chunks before upserting.
- Background job system is used; no blocking during HTTP request.
- No raw filesystem paths in SSE events.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.auth import require_documents_write
from app.services import ingest_service, config
from app.services.job_store import get_job_store

router = APIRouter(prefix="/api", tags=["ingest"])
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# POST /api/ingest — trigger background ingestion of all PDFs
# ─────────────────────────────────────────────────────────────

@router.post("/ingest")
def ingest_all(
    _role: str = Depends(require_documents_write),
):
    """
    Trigger ingestion of all PDFs in the source directory.

    Returns a job_id for tracking. Poll GET /api/jobs/{job_id} for status.
    Re-ingestion automatically removes stale chunks for each document.
    """
    store = get_job_store()
    job_id = store.submit(
        _ingest_all_sync,
        idempotency_key="ingest:all",
        timeout_seconds=3600.0,
    )
    return JSONResponse(
        {"job_id": job_id, "status": "pending", "message": "Ingestion job queued."},
        status_code=202,
    )


def _ingest_all_sync() -> dict:
    """Synchronous wrapper for ingest_all_pdfs, suitable for thread pool."""
    started = time.time()
    results = ingest_service.ingest_all_pdfs()
    total = sum(cnt for _, cnt, _ in results)
    return {
        "ingested": [
            {"filename": name, "chunks_upserted": cnt, "pages_processed": pages}
            for (name, cnt, pages) in results
        ],
        "total_chunks": total,
        "file_count": len(results),
        "took_seconds": round(time.time() - started, 3),
    }


# ─────────────────────────────────────────────────────────────
# GET /api/ingest/stream — SSE streaming ingest
# ─────────────────────────────────────────────────────────────

@router.get("/ingest/stream")
def ingest_stream(
    _role: str = Depends(require_documents_write),
) -> StreamingResponse:
    """
    Server-Sent Events stream for real-time ingestion progress.

    Events do not include filesystem paths.
    """
    def gen() -> Iterator[str]:
        files = sorted(config.SOURCE_PDFS.glob("*.pdf"))
        if not files:
            yield "event: status\n"
            yield f"data: {json.dumps({'type': 'empty', 'message': 'No PDFs found.'})}\n\n"
            return

        yield "event: status\n"
        yield f"data: {json.dumps({'type': 'start', 'count': len(files)})}\n\n"

        total_chunks = 0
        for idx, p in enumerate(files, start=1):
            t0 = time.time()
            # Emit start event with filename only (not full path)
            yield "event: file\n"
            yield f"data: {json.dumps({'type': 'file_start', 'index': idx, 'filename': p.name})}\n\n"

            try:
                name, count, pages = ingest_service.ingest_pdf(p)
                total_chunks += count
                dt = round(time.time() - t0, 3)
                yield "event: file\n"
                yield f"data: {json.dumps({'type': 'file_done', 'index': idx, 'filename': name, 'chunks_upserted': count, 'pages': pages, 'seconds': dt})}\n\n"
            except ingest_service.IngestionError as exc:
                dt = round(time.time() - t0, 3)
                yield "event: file\n"
                yield f"data: {json.dumps({'type': 'file_error', 'index': idx, 'filename': p.name, 'error': str(exc), 'seconds': dt})}\n\n"
            except Exception as exc:
                log.exception("Ingest failed for %s", p.name)
                dt = round(time.time() - t0, 3)
                yield "event: file\n"
                yield f"data: {json.dumps({'type': 'file_error', 'index': idx, 'filename': p.name, 'error': 'Internal error during ingestion.', 'seconds': dt})}\n\n"

        yield "event: status\n"
        yield f"data: {json.dumps({'type': 'done', 'files': len(files), 'total_chunks': total_chunks})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
