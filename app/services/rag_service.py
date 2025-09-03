# app/services/rag_service.py
from __future__ import annotations
from typing import Dict, List, Tuple
import textwrap

from . import config, vectorstore, expand, embeddings, llm

def retrieve(query: str, top_k: int = 10, filter_doc: str | None = None):
    # expand acronyms for recall
    queries = expand.expanded_queries(query)
    all_hits = []
    for q in queries:
        hv = vectorstore.hybrid_search(q, topk_dense=config.ENV.TOPK_DENSE, topk_bm25=config.ENV.TOPK_BM25)
        all_hits.extend(hv)
    # de-dup by (source,page)
    uniq = {}
    def blended(h): return config.ENV.HYBRID_WEIGHT_DENSE*h.score_vec + config.ENV.HYBRID_WEIGHT_BM25*h.score_bm25
    for h in all_hits:
        if filter_doc and h.source != filter_doc:
            continue
        key = (h.source, h.page)
        if key not in uniq or blended(h) > blended(uniq[key]):
            uniq[key] = h
    ranked = sorted(uniq.values(), key=blended, reverse=True)
    ranked = vectorstore.mmr_diverse(ranked, top_k=min(top_k, config.ENV.TOPK_AFTER_MMR), lambda_mult=config.ENV.MMR_LAMBDA)
    return ranked

def build_context(hits) -> Tuple[str, List[Dict]]:
    # format as numbered snippets with citations
    pieces = []
    cits: List[Dict] = []
    for i, h in enumerate(hits, start=1):
        txt = h.text.strip().replace("\n", " ").strip()
        pieces.append(f"[{i}] Source: {h.source} (p.{h.page})\n{txt}")
        cits.append({"n": i, "source": h.source, "page": h.page})
    context = "\n\n".join(pieces)
    return context, cits

def answer(query: str, top_k: int = 10, max_context_chars: int = 6000, filter_doc: str | None = None) -> Dict:
    hits = retrieve(query, top_k=top_k, filter_doc=filter_doc)
    context, cits = build_context(hits)
    if not context.strip():
        return {
            "answer": "I couldn't find enough context in the ingested documents. Please upload or specify the standard/document.",
            "citations": [],
            "used_provider": "openai",
            "meta": {"hits": []},
        }
    # Trim context to size
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n...\n"

    system = config.SYSTEM_PROMPT
    text = llm.generate(system=system, context=context, user_query=query)

    return {
        "answer": text,
        "citations": cits,
        "used_provider": "openai",
        "meta": {
            "hit_count": len(hits),
            "top_sources": list({(h.source) for h in hits}),
        },
    }
