# app/routers/upload.py
from __future__ import annotations
from typing import List
from pathlib import Path
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from ..services import config, ingest_service

router = APIRouter(prefix="/api", tags=["upload"])

@router.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    saved = []
    for f in files:
        dest = Path(config.SOURCE_PDFS) / f.filename
        data = await f.read()
        dest.write_bytes(data)
        saved.append(f.filename)
    # auto-ingest the newly uploaded files
    counts = ingest_service.ingest_specific_files([Path(config.SOURCE_PDFS) / s for s in saved])
    return JSONResponse({"uploaded": saved, "ingested": [{"filename": n, "chunks_upserted": c} for (n, c) in counts]})
