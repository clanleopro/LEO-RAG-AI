# app/main.py
"""
LEO Rigging AI — FastAPI application entry point.

Security controls applied:
- CORS: no wildcard + credentials; requires explicit origin allowlist in production.
- /docs, /redoc, /openapi.json disabled in production.
- /info and /routes protected by admin auth.
- Separate liveness and readiness health endpoints.
- Security headers middleware.
- Trusted-host middleware.
- Structured logging with request-id correlation.
- Global safe exception handler (no stack traces to clients).
- Startup security warning banner for exposed API key.
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException, Request, status
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.middleware.trustedhost import TrustedHostMiddleware
# pyrefly: ignore [missing-import]
from fastapi.responses import JSONResponse
# pyrefly: ignore [missing-import]
from starlette.middleware.base import BaseHTTPMiddleware

# ─── Load config first (also loads .env) ─────────────────────
from app.services import config as app_config  # noqa: E402

# ─── Logging ─────────────────────────────────────────────────
_LOG_LEVEL = app_config.ENV.LOG_LEVEL
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | req=%(request_id)s | %(message)s"
    if app_config.ENV.LOG_JSON
    else "%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
)
log = logging.getLogger("app.main")

# ─── Import routers ───────────────────────────────────────────
from app.routers import ingest, query, upload, files, search  # noqa: E402
from app.routers import jobs  # noqa: E402
from app.services import vectorstore, embeddings  # noqa: E402
from app.core.auth import require_admin  # noqa: E402
from app.models.schemas import (  # noqa: E402
    ErrorResponse,
    LivenessResponse,
    ReadinessResponse,
)
# pyrefly: ignore [missing-import]
from fastapi import Depends  # noqa: E402

_ENV = app_config.ENV


# ─────────────────────────────────────────────────────────────
# Security banner at startup
# ─────────────────────────────────────────────────────────────

def _print_security_banner() -> None:
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║  ⚠️  SECURITY WARNING — ACTION REQUIRED BY REPOSITORY OWNER           ║
║                                                                      ║
║  A real OpenAI API key was previously committed to Git history.       ║
║                                                                      ║
║  You MUST:                                                           ║
║  1. REVOKE the exposed key at: https://platform.openai.com/api-keys  ║
║  2. Review billing for unauthorized charges.                         ║
║  3. Create a NEW replacement key.                                    ║
║  4. Store it ONLY in a secret manager or untracked .env file.        ║
║  5. Purge the old secret from Git history (git filter-repo / BFG).   ║
║                                                                      ║
║  See SECURITY.md for the full checklist.                             ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    log.warning(banner)


# ─────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _print_security_banner()

    try:
        embeddings.check_model_consistency()
    except Exception as exc:
        log.warning("[startup] Embedding model consistency check error: %s", exc)

    try:
        vectorstore.rebuild_bm25_index()
        log.info("[startup] BM25 index built.")
        log.info(
            "[startup] collection=%s embed_model=%s llm=%s/%s",
            _ENV.CHROMA_COLLECTION,
            _ENV.EMBED_MODEL,
            _ENV.LLM_PROVIDER,
            _ENV.OPENAI_MODEL,
        )
    except Exception as exc:
        log.exception("[startup] Initialization failed: %s", exc)

    yield

    # Graceful shutdown
    from app.services.job_store import get_job_store  # noqa: PLC0415
    try:
        get_job_store().shutdown(wait=False)
    except Exception:
        pass
    log.info("[shutdown] LEO Rigging AI stopped.")


# ─────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────

def _make_app() -> FastAPI:
    # Disable /docs, /redoc, /openapi.json in production
    docs_url = None if _ENV.is_production else "/docs"
    redoc_url = None if _ENV.is_production else "/redoc"
    openapi_url = None if _ENV.is_production else "/openapi.json"

    application = FastAPI(
        title="LEO Rigging AI",
        version="1.0.0",
        description=(
            "Educational RAG assistant for the rigging, lifting, crane, and hoisting industry. "
            "NOT a rigging calculator, lift-plan approval system, or engineering authority."
        ),
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
        lifespan=lifespan,
    )
    return application


app = _make_app()


# ─────────────────────────────────────────────────────────────
# Middleware
# ─────────────────────────────────────────────────────────────

# ── Trusted Host ─────────────────────────────────────────────
if _ENV.is_production:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=_ENV.allowed_hosts_list,
    )

# ── CORS ─────────────────────────────────────────────────────
_cors_origins = _ENV.cors_origins
_allow_credentials = "*" not in _cors_origins  # Never combine wildcard with credentials

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
)


# ── Request ID middleware ─────────────────────────────────────
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(RequestIDMiddleware)


# ── Security headers middleware ───────────────────────────────
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"] = "no-store"
        if _ENV.is_production:
            response.headers["Strict-Transport-Security"] = (
                "max-age=63072000; includeSubDomains; preload"
            )
        return response


app.add_middleware(SecurityHeadersMiddleware)


# ─────────────────────────────────────────────────────────────
# Global exception handler
# ─────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    log.error("Unhandled exception req=%s: %s", request_id, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "An unexpected error occurred.",
            "request_id": request_id,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "request_id": request_id,
        },
        headers=getattr(exc, "headers", None),
    )


# ─────────────────────────────────────────────────────────────
# Health endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/health/live", response_model=LivenessResponse, tags=["health"])
def liveness():
    """Liveness probe — returns 200 if the process is running."""
    return LivenessResponse(status="ok")


@app.get("/health/ready", response_model=ReadinessResponse, tags=["health"])
def readiness():
    """
    Readiness probe — returns 200 only when all required services are available.
    Returns 503 if any required check fails.
    """
    checks: Dict[str, str] = {}
    overall = "ready"

    # Check vector store
    try:
        sources = vectorstore.list_sources()
        checks["vectorstore"] = "ok"
    except Exception as exc:
        checks["vectorstore"] = f"error: {type(exc).__name__}"
        overall = "not_ready"

    # Check embedding model (lazy load; just verify config)
    if _ENV.EMBED_MODEL:
        checks["embedding_model"] = "configured"
    else:
        checks["embedding_model"] = "not configured"
        overall = "not_ready"

    # Check LLM API key presence
    key = _ENV.OPENAI_API_KEY or ""
    if key and "REPLACE" not in key.upper() and len(key) > 20:
        checks["llm_api_key"] = "present"
    else:
        checks["llm_api_key"] = "missing or placeholder"
        if overall == "ready":
            overall = "degraded"

    status_code = 200 if overall == "ready" else 503
    return JSONResponse(
        status_code=status_code,
        content=ReadinessResponse(status=overall, checks=checks).model_dump(),
    )


# ─────────────────────────────────────────────────────────────
# Root endpoint
# ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "LEO Rigging AI",
        "version": "1.0.0",
        "status": "running",
        "documentation": "See README.md and docs/ for usage.",
        "health": {
            "liveness": "/health/live",
            "readiness": "/health/ready",
        },
    }


# ─────────────────────────────────────────────────────────────
# Admin-only diagnostic endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/info", tags=["admin"])
def info(_role: str = Depends(require_admin)):
    """
    System info — requires admin scope.
    Does NOT expose filesystem paths or credentials.
    """
    try:
        sources = vectorstore.list_sources()
    except Exception:
        sources = []
    return {
        "vectorstore": {
            "collection": _ENV.CHROMA_COLLECTION,
            "sources": sources,
        },
        "embedding_model": _ENV.EMBED_MODEL,
        "llm": {
            "provider": _ENV.LLM_PROVIDER,
            "model": _ENV.OPENAI_MODEL,
        },
        "chunking": {
            "tokens": _ENV.CHUNK_TOKENS,
            "overlap": _ENV.CHUNK_OVERLAP,
        },
        "retrieval": {
            "topk_dense": _ENV.TOPK_DENSE,
            "topk_bm25": _ENV.TOPK_BM25,
            "min_relevance_score": _ENV.MIN_RELEVANCE_SCORE,
        },
    }


@app.get("/routes", tags=["admin"])
def list_routes(_role: str = Depends(require_admin)):
    """Route listing — requires admin scope."""
    return [
        {
            "path": r.path,
            "name": r.name,
            "methods": sorted(r.methods or []),
        }
        for r in app.router.routes
        if hasattr(r, "methods")
    ]


# ─────────────────────────────────────────────────────────────
# Include routers
# ─────────────────────────────────────────────────────────────

app.include_router(upload.router)
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(files.router)
app.include_router(search.router, prefix="/api")
app.include_router(jobs.router)

# ─────────────────────────────────────────────────────────────
# Development entrypoint
# Run: uvicorn app.main:app --host 0.0.0.0 --port 8000
# DO NOT use --reload in production.
# ─────────────────────────────────────────────────────────────