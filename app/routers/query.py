# app/routers/query.py
"""
Query endpoint — ask questions against ingested documents.

HTTP status code mapping for LLM errors:
  LLMRateLimitError    → 429
  LLMCreditError       → 503
  LLMSpendLimitError   → 503
  LLMAuthError         → 503
  LLMTimeoutError      → 504
  LLMUnavailableError  → 503
  LLMBadRequestError   → 500

grounded is NEVER True when generation fails.
Citations are NEVER returned without a successful answer.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from app.core.auth import require_query
from app.models.schemas import Citation, QueryRequest, QueryResponse
from app.services import rag_service
from app.services.llm import (
    LLMAuthError,
    LLMBadRequestError,
    LLMCreditError,
    LLMError,
    LLMRateLimitError,
    LLMSpendLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)

router = APIRouter(prefix="/api", tags=["query"])
log = logging.getLogger(__name__)


def _llm_error_response(exc: LLMError, request_id: str = "unknown") -> JSONResponse:
    """
    Map a LLMError to a JSONResponse with grounded:false and correct HTTP status.
    Never returns HTTP 200 on generation failure.
    Never exposes provider response bodies.
    """
    if isinstance(exc, LLMRateLimitError):
        http_status = status.HTTP_429_TOO_MANY_REQUESTS
        headers = {"Retry-After": "10"}
    elif isinstance(exc, LLMTimeoutError):
        http_status = status.HTTP_504_GATEWAY_TIMEOUT
        headers = {}
    elif isinstance(exc, LLMBadRequestError):
        http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
        headers = {}
    else:
        # LLMCreditError, LLMSpendLimitError, LLMAuthError, LLMUnavailableError
        http_status = status.HTTP_503_SERVICE_UNAVAILABLE
        headers = {}

    return JSONResponse(
        status_code=http_status,
        content={
            "error": exc.safe_message,
            "grounded": False,
            "request_id": request_id,
        },
        headers=headers,
    )


@router.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    _role: str = Depends(require_query),
) -> QueryResponse | JSONResponse:
    """
    Ask an educational question about rigging, lifting, cranes, or hoisting.

    Returns a grounded answer based on ingested documents.
    If retrieved context scores are below the minimum relevance threshold,
    a grounded refusal is returned instead of a hallucinated answer.
    """
    try:
        result = rag_service.answer(
            query=req.question,  # already validated and stripped
            top_k=req.top_k,
            max_context_chars=req.max_context_chars,
            filter_doc=req.filter_doc,
        )
    except LLMError as exc:
        # LLM-specific failures — return correct HTTP code with grounded:false
        log.warning(
            "Query LLM error [type=%s]: returning HTTP error, grounded=false",
            type(exc).__name__,
        )
        return _llm_error_response(exc)
    except Exception as exc:
        log.error("Query pipeline error: %s", type(exc).__name__, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the query.",
        )

    # Map raw citation dicts to Citation schema objects
    citations = [
        Citation(
            n=c["n"],
            doc_id=c["doc_id"],
            title=c["title"],
            page=c["page"],
            revision=c.get("revision"),
            snippet=c["snippet"],
            score_vec=c["score_vec"],
            score_bm25=c["score_bm25"],
            score_blended=c["score_blended"],
        )
        for c in result.get("citations", [])
    ]

    return QueryResponse(
        answer=result["answer"],
        citations=citations,
        low_confidence=result.get("low_confidence", False),
        grounded=result.get("grounded", False),   # safe default: False
        used_provider=result.get("used_provider", "openai"),
        meta=result.get("meta", {}),
    )
