# app/main.py
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# Load .env from repo root explicitly
try:
    from dotenv import load_dotenv
    ROOT = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=ROOT / ".env", override=False)
except Exception:
    pass

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
)
log = logging.getLogger("app.main")

# Routers & services
from app.routers import ingest, query, upload, files, search
from app.services import vectorstore, config

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        vectorstore.rebuild_bm25_index()
        log.info("[startup] BM25 index built.")
        log.info(
            "[startup] Chroma dir=%s | collection=%s | embed_model=%s | llm_provider=%s | openai_model=%s",
            str(config.VECTOR_DIR),
            getattr(config.ENV, "CHROMA_COLLECTION", "leo_rigging_ai"),
            config.ENV.EMBED_MODEL,
            (config.ENV.LLM_PROVIDER or "openai"),
            getattr(config.ENV, "OPENAI_MODEL", "gpt-4o-mini"),
        )
    except Exception as e:
        log.exception("[startup] init failed: %s", e)
    yield
    log.info("[shutdown] Bye.")

app = FastAPI(
    title="LEO Rigging AI",
    version="0.2.2",
    description="RAG service for lifting/cranes/rigging with hybrid retrieval, OCR, and citations.",
    lifespan=lifespan,
)

# CORS
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*")
allow_origins = [o.strip() for o in CORS_ALLOW_ORIGINS.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
start_time = time.time()
# Health & info
@app.get("/health")
def health():
    uptime = round(time.time() - start_time, 2)  # seconds
    return {
        "service": "LEO Rigging AI",
        "status": "healthy",
        "uptime_seconds": uptime,
        "version": "0.2.2"
    }

@app.get("/")
def root():
    return {
        "name": "LEO Rigging AI",
        "version": "0.2.2",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "redoc": "/redoc",
            "upload": "/api/upload",
            "ingest": "/api/ingest",
            "ingest_stream": "/api/ingest/stream",
            "query": "/api/query",
            "search": "/api/search",
            "files_list": "/api/pdfs",
            "files_download": "/api/pdfs/{filename}",
            "files_delete": "/api/pdfs/{filename}",
            "info": "/info",
            "routes": "/routes",
        },
    }

@app.get("/info")
def info():
    try:
        sources = vectorstore.list_sources()
    except Exception:
        sources = []
    return {
        "vectorstore": {
            "dir": str(config.VECTOR_DIR),
            "collection": getattr(config.ENV, "CHROMA_COLLECTION", "leo_rigging_ai"),
            "sample": sources,
        },
        "embedding_model": config.ENV.EMBED_MODEL,
        "llm": {"provider": (config.ENV.LLM_PROVIDER or "openai"),
                "model": getattr(config.ENV, "OPENAI_MODEL", "gpt-4o-mini")},
        "chunking": {"tokens": config.ENV.CHUNK_TOKENS, "overlap": config.ENV.CHUNK_OVERLAP},
    }

@app.get("/routes")
def list_routes():
    return [{"path": r.path, "name": r.name, "methods": list(r.methods or [])} for r in app.router.routes]

# Routers
app.include_router(ingest.router)                 # has internal prefix="/api"
app.include_router(query.router)                  # prefix="/api"
app.include_router(upload.router)                 # prefix="/api"
app.include_router(files.router)                  # prefix="/api"
app.include_router(search.router, prefix="/api")  # search had no internal prefix



# run  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload