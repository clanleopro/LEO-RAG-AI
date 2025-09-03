# app/services/vectorstore.py
from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import chromadb
import numpy as np
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from . import config, embeddings

log = logging.getLogger(__name__)

# ---------- Data types ----------
@dataclass
class Chunk:
    id: str | None
    text: str
    source: str
    page: int
    headings: List[str] | None = None
    language: str | None = None
    standard_code: str | None = None
    embedding: Optional[np.ndarray] = None

@dataclass
class SearchHit:
    id: str
    text: str
    source: str
    page: int
    score_vec: float
    score_bm25: float

# ---------- Helpers ----------
def _meta_primitive(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple, set)):
        try:
            return ", ".join(str(x) for x in v)
        except Exception:
            return str(list(v))
    return str(v)

def _tokenize(s: str) -> List[str]:
    if not s:
        return []
    return [t for t in s.lower().split() if t.strip()]

def _deterministic_id(source: str, page: int, text: str) -> str:
    h = hashlib.sha256()
    h.update((source or "").encode("utf-8"))
    h.update(str(page).encode("utf-8"))
    h.update((text or "")[:1024].encode("utf-8"))
    return h.hexdigest()[:32]

def _parse_max_batch_from_msg(msg: str) -> int | None:
    m = re.search(r"max batch size of (\d+)", msg)
    return int(m.group(1)) if m else None

# ---------- Chroma setup ----------
_client = chromadb.PersistentClient(
    path=str(config.VECTOR_DIR),
    settings=Settings(anonymized_telemetry=False),
)
_COLLECTION_NAME = config.ENV.CHROMA_COLLECTION or "leo_rigging_ai"

def _get_collection():
    try:
        return _client.get_collection(_COLLECTION_NAME)
    except Exception:
        return _client.create_collection(
            _COLLECTION_NAME,
            metadata={"embedding_model": config.ENV.EMBED_MODEL, "created": time.time()},
        )

# ---------- BM25 index ----------
_BM25: Optional[BM25Okapi] = None
_BM25_IDS: List[str] = []
_BM25_TEXTS: List[str] = []

def rebuild_bm25_index() -> None:
    global _BM25, _BM25_IDS, _BM25_TEXTS
    coll = _get_collection()
    _BM25_IDS, _BM25_TEXTS = [], []

    offset = 0
    page_size = 1000
    while True:
        res = coll.get(limit=page_size, offset=offset, include=["documents"])
        ids = res.get("ids", [])
        docs = res.get("documents", [])
        if not ids:
            break
        _BM25_IDS.extend(ids)
        _BM25_TEXTS.extend(docs)
        offset += len(ids)

    corpus = [_tokenize(t) for t in _BM25_TEXTS]
    _BM25 = BM25Okapi(corpus) if corpus else None

# ---------- Public API ----------
def upsert_chunks(chunks: List[Chunk]) -> int:
    if not chunks:
        return 0
    coll = _get_collection()

    # De-dupe (source,page,text) & ids
    seen_triplet = set()
    uniq_chunks: List[Chunk] = []
    for c in chunks:
        key = (c.source, int(c.page), (c.text or "").strip())
        if key in seen_triplet:
            continue
        seen_triplet.add(key)
        uniq_chunks.append(c)

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict] = []
    vecs: List[Optional[np.ndarray]] = []
    missing_texts: List[str] = []
    seen_ids = set()

    for c in uniq_chunks:
        cid = (c.id or _deterministic_id(c.source, c.page, c.text or "")).strip()
        if cid in seen_ids:
            continue
        seen_ids.add(cid)

        ids.append(cid)
        docs.append(c.text or "")
        meta_raw = {
            "source": c.source,
            "page": int(c.page),
            "headings": c.headings,
            "language": c.language,
            "standard_code": c.standard_code,
        }
        metas.append({k: _meta_primitive(v) for k, v in meta_raw.items()})

        if c.embedding is None:
            missing_texts.append(c.text or "")
            vecs.append(None)
        else:
            vecs.append(c.embedding)

    if missing_texts:
        new_vecs = embeddings.embed(missing_texts)
        j = 0
        for i in range(len(vecs)):
            if vecs[i] is None:
                vecs[i] = new_vecs[j]
                j += 1

    if not ids:
        return 0

    batch_size = max(1, config.ENV.CHROMA_BATCH_SIZE)
    start = 0
    while start < len(ids):
        end = min(start + batch_size, len(ids))
        try:
            coll.upsert(
                ids=ids[start:end],
                documents=docs[start:end],
                metadatas=metas[start:end],
                embeddings=[(v.tolist() if hasattr(v, "tolist") else v) for v in vecs[start:end]],
            )
            start = end
        except Exception as e:
            msg = str(e)
            hinted = _parse_max_batch_from_msg(msg)
            if hinted is not None and hinted < batch_size:
                log.warning("Chroma hinted max batch=%d (was %d). Reducing.", hinted, batch_size)
                batch_size = max(1, hinted)
            elif batch_size > 1:
                batch_size = max(1, batch_size // 2)
                log.warning("Upsert failed (%s). Retrying with smaller batch_size=%d", msg, batch_size)
            else:
                raise

    rebuild_bm25_index()
    return len(ids)

def delete_by_source(source_filename: str) -> int:
    coll = _get_collection()
    res = coll.get(where={"source": source_filename})
    ids = res.get("ids", []) if isinstance(res, dict) else []
    if ids:
        coll.delete(ids=ids)
    rebuild_bm25_index()
    return len(ids)

def wipe() -> None:
    try:
        _client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass
    rebuild_bm25_index()

def list_sources() -> List[Dict]:
    coll = _get_collection()
    res = coll.get(include=[])
    ids = res.get("ids", []) if isinstance(res, dict) else []
    metas = coll.get(ids=ids[:100], include=["metadatas"]).get("metadatas", [])
    return [{
        "collection": _COLLECTION_NAME,
        "count_sample": len(ids),
        "sample_sources": list({m.get("source") for m in metas}) if metas else [],
    }]

def hybrid_search(query: str, topk_dense: int, topk_bm25: int) -> List[SearchHit]:
    coll = _get_collection()
    q_vec = embeddings.embed_one(query).tolist()
    dres = coll.query(
        query_embeddings=[q_vec],
        n_results=topk_dense,
        include=["documents", "metadatas", "distances"],  # no "ids" here
    )

    ids_d   = dres.get("ids", [[]])[0]
    docs_d  = dres.get("documents", [[]])[0]
    metas_d = dres.get("metadatas", [[]])[0]
    dists   = dres.get("distances", [[]])[0]

    dense_sims: Dict[str, float] = {}
    dense_payload: Dict[str, Tuple[str, Dict]] = {}
    for i, _id in enumerate(ids_d):
        sim = 1.0 - float(dists[i]) if i < len(dists) else 0.0
        dense_sims[_id] = sim
        t = docs_d[i] if i < len(docs_d) else ""
        m = metas_d[i] if i < len(metas_d) else {}
        dense_payload[_id] = (t, m)

    bm25_scores: Dict[str, float] = {}
    if _BM25 is not None and _BM25_IDS:
        scores = _BM25.get_scores(_tokenize(query))
        top_idx = np.argsort(scores)[::-1][:topk_bm25]
        for idx in top_idx:
            bm25_scores[_BM25_IDS[idx]] = float(scores[idx])

    keys: List[str] = []
    merged: Dict[str, Dict[str, float]] = {}
    for _id, sim in dense_sims.items():
        if _id not in merged:
            keys.append(_id); merged[_id] = {}
        merged[_id]["score_vec"] = sim
    for _id, score in bm25_scores.items():
        if _id not in merged:
            keys.append(_id); merged[_id] = {}
        merged[_id]["score_bm25"] = score

    dense_vals = [merged[k].get("score_vec", 0.0) for k in keys]
    bm25_vals  = [merged[k].get("score_bm25", 0.0) for k in keys]
    def _norm(vals: List[float]) -> List[float]:
        if not vals:
            return []
        arr = np.array(vals, dtype=np.float32)
        if float(arr.std()) < 1e-6:
            m = arr / (abs(arr).max() + 1e-6)
        else:
            z = (arr - arr.mean()) / (arr.std() + 1e-6)
            m = (z - z.min()) / (z.max() - z.min() + 1e-6)
        return [float(v) for v in m]
    ndense = _norm(dense_vals); nbm25 = _norm(bm25_vals)

    hits: List[SearchHit] = []
    bm25_only_ids = [k for k in keys if k not in dense_payload]
    bm25_payload: Dict[str, Tuple[str, Dict]] = {}
    if bm25_only_ids:
        got = coll.get(ids=bm25_only_ids, include=["documents", "metadatas"])
        for _id, text, meta in zip(got.get("ids", []), got.get("documents", []), got.get("metadatas", [])):
            bm25_payload[_id] = (text, meta)

    for i, _id in enumerate(keys):
        dv = ndense[i] if i < len(ndense) else 0.0
        bv = nbm25[i] if i < len(nbm25) else 0.0
        if _id in dense_payload:
            text, meta = dense_payload[_id]
        else:
            text, meta = bm25_payload.get(_id, ("", {}))
        source = meta.get("source", "") if isinstance(meta, dict) else ""
        page = int(meta.get("page", 0)) if isinstance(meta, dict) else 0
        hits.append(SearchHit(id=_id, text=text or "", source=source, page=page, score_vec=dv, score_bm25=bv))
    return hits

def mmr_diverse(hits: List[SearchHit], top_k: int, lambda_mult: float = 0.6) -> List[SearchHit]:
    def blended(h: SearchHit) -> float:
        return config.ENV.HYBRID_WEIGHT_DENSE * h.score_vec + config.ENV.HYBRID_WEIGHT_BM25 * h.score_bm25
    selected: List[SearchHit] = []; used = set()
    pool = sorted(hits, key=blended, reverse=True)
    while pool and len(selected) < top_k:
        best = None; best_val = -1e9
        for h in pool:
            sim_penalty = (1.0 - lambda_mult) * 0.4 if (h.source, h.page) in used else 0.0
            val = lambda_mult * blended(h) - sim_penalty
            if val > best_val:
                best, best_val = h, val
        if not best: break
        selected.append(best); used.add((best.source, best.page)); pool.remove(best)
    return selected

def query(q_or_emb, top_k: int = 10, filter_doc: str | None = None):
    coll = _get_collection()
    if isinstance(q_or_emb, str):
        hits = hybrid_search(q_or_emb, topk_dense=config.ENV.TOPK_DENSE, topk_bm25=config.ENV.TOPK_BM25)
        hits = mmr_diverse(hits, top_k=top_k, lambda_mult=config.ENV.MMR_LAMBDA)
        if filter_doc:
            hits = [h for h in hits if h.source == filter_doc]
        def blended(h: SearchHit) -> float:
            return config.ENV.HYBRID_WEIGHT_DENSE * h.score_vec + config.ENV.HYBRID_WEIGHT_BM25 * h.score_bm25
        scores = np.array([blended(h) for h in hits], dtype=np.float32)
        if len(scores) > 0:
            smin, smax = float(scores.min()), float(scores.max())
            norm = (scores - smin) / (smax - smin + 1e-6)
        else:
            norm = np.array([], dtype=np.float32)
        documents = [[h.text for h in hits]]
        metadatas = [[{"source": h.source, "page": h.page} for h in hits]]
        distances = [[float(1.0 - s) for s in norm]]
        return {"documents": documents, "metadatas": metadatas, "distances": distances}
    q_vec = q_or_emb.tolist() if isinstance(q_or_emb, np.ndarray) else q_or_emb
    return coll.query(
        query_embeddings=[q_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
        where={"source": filter_doc} if filter_doc else None,
    )
