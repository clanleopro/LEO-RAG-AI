# app/routers/query.py
from __future__ import annotations
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from ..services import rag_service

router = APIRouter(prefix="/api", tags=["query"])

@router.post("/query")
def query(req: dict):
    q = req.get("query") or req.get("question") or ""
    top_k = int(req.get("top_k", 10))
    max_context_chars = int(req.get("max_context_chars", 6000))
    filter_doc = req.get("filter_doc")
    result = rag_service.answer(
        query=q,
        top_k=top_k,
        max_context_chars=max_context_chars,
        filter_doc=filter_doc,
    )
    return JSONResponse(result)
