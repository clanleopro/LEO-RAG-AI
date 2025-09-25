# app/routers/search.py
from __future__ import annotations
from typing import List
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from ..services import vectorstore, config

router = APIRouter(tags=["search"])

@router.post("/search")
def search(req: dict):
    q = req.get("query", "")
    top_k = int(req.get("top_k", 10))
    hits = vectorstore.hybrid_search(q, topk_dense=config.ENV.TOPK_DENSE, topk_bm25=config.ENV.TOPK_BM25)
    hits = vectorstore.mmr_diverse(hits, top_k=top_k, lambda_mult=config.ENV.MMR_LAMBDA)
    out: List[dict] = []
    for h in hits:
        out.append({
            "source": h.source,
            "page": h.page,
            "score_vec": h.score_vec,
            "score_bm25": h.score_bm25,
            "snippet": (h.text[:400] + "...") if len(h.text) > 400 else h.text,
        })
    return JSONResponse({"results": out})

