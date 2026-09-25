# app/routers/search.py
"""
Raw hybrid search endpoint (no LLM generation).
"""
from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.auth import require_query
from app.models.schemas import SearchHitResponse, SearchRequest, SearchResponse
from app.services import config, vectorstore

router = APIRouter(tags=["search"])
log = logging.getLogger(__name__)


@router.post("/search", response_model=SearchResponse)
def search(
    req: SearchRequest,
    _role: str = Depends(require_query),
) -> SearchResponse:
    """
    Hybrid semantic + BM25 search over ingested documents.

    Returns raw retrieval results without LLM generation.
    Useful for debugging retrieval quality.
    """
    try:
        hits = vectorstore.hybrid_search(
            req.query,
            topk_dense=config.ENV.TOPK_DENSE,
            topk_bm25=config.ENV.TOPK_BM25,
        )
        hits = vectorstore.mmr_diverse(
            hits,
            top_k=req.top_k,
            lambda_mult=config.ENV.MMR_LAMBDA,
        )
    except Exception as exc:
        log.error("Search failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search operation failed.",
        )

    results: List[SearchHitResponse] = []
    for h in hits:
        snippet = (h.text[:400] + "...") if len(h.text) > 400 else h.text
        results.append(SearchHitResponse(
            doc_id=h.doc_id or h.source,
            title=h.source,
            page=h.page,
            score_vec=round(h.score_vec, 4),
            score_bm25=round(h.score_bm25, 4),
            score_blended=round(h.score_blended, 4),
            snippet=snippet,
        ))

    return SearchResponse(results=results)
