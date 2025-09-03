# app/routers/ingest.py
from __future__ import annotations
import json
import logging
import time
from typing import Iterator, List, Tuple
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from ..services import ingest_service, config

router = APIRouter(prefix="/api", tags=["ingest"])
log = logging.getLogger(__name__)

@router.post("/ingest")
def ingest_all():
    started = time.time()
    results: List[Tuple[str, int]] = ingest_service.ingest_all_pdfs()
    total = sum(cnt for _, cnt in results)
    payload = [{"filename": name, "chunks_upserted": cnt} for (name, cnt) in results]
    return JSONResponse({
        "success": True,
        "ingested": payload,
        "total_chunks": total,
        "file_count": len(payload),
        "took_seconds": round(time.time() - started, 3),
    })

@router.get("/ingest/stream")
def ingest_stream() -> StreamingResponse:
    def gen() -> Iterator[str]:
        files = sorted(config.SOURCE_PDFS.glob("*.pdf"))
        if not files:
            yield "event: status\n"
            yield f"data: {json.dumps({'type':'empty','message':'No PDFs found','dir': str(config.SOURCE_PDFS)})}\n\n"
            return
        yield "event: status\n"
        yield f"data: {json.dumps({'type':'start','count':len(files)})}\n\n"

        total_chunks = 0
        for idx, p in enumerate(files, start=1):
            t0 = time.time()
            yield "event: file\n"
            yield f"data: {json.dumps({'type':'file_start','index':idx,'filename':p.name})}\n\n"
            try:
                name, count = ingest_service.ingest_pdf(p)
                total_chunks += count
                dt = round(time.time() - t0, 3)
                yield "event: file\n"
                yield f"data: {json.dumps({'type':'file_done','index':idx,'filename':name,'chunks_upserted':count,'seconds':dt})}\n\n"
            except Exception as e:
                log.exception("Ingest failed for %s", p.name)
                dt = round(time.time() - t0, 3)
                yield "event: file\n"
                yield f"data: {json.dumps({'type':'file_error','index':idx,'filename':p.name,'error':str(e),'seconds':dt})}\n\n"
        yield "event: status\n"
        yield f"data: {json.dumps({'type':'done','files':len(files),'total_chunks':total_chunks})}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
