# app/routers/files.py
from __future__ import annotations
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from ..services import config, vectorstore

router = APIRouter(prefix="/api", tags=["files"])

@router.get("/pdfs")
def list_pdfs():
    files = []
    for p in sorted(config.SOURCE_PDFS.glob("*.pdf")):
        stat = p.stat()
        files.append({
            "name": p.name,
            "size": stat.st_size,
            "modified": int(stat.st_mtime),
        })
    return JSONResponse({"dir": str(config.SOURCE_PDFS), "files": files})

@router.get("/pdfs/{filename}")
def download_pdf(filename: str):
    path = Path(config.SOURCE_PDFS) / filename
    if not path.exists():
        raise HTTPException(404, f"{filename} not found")
    return FileResponse(str(path), media_type="application/pdf", filename=filename)

@router.delete("/pdfs/{filename}")
def delete_pdf(filename: str):
    path = Path(config.SOURCE_PDFS) / filename
    if not path.exists():
        raise HTTPException(404, f"{filename} not found")
    path.unlink()
    removed = vectorstore.delete_by_source(filename)
    return JSONResponse({"deleted": filename, "vectors_removed": removed})
