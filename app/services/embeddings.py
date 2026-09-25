# app/services/embeddings.py
"""
Sentence-transformer embedding service.

Applies model-specific query/document prefixes (required by multilingual-e5
family models) and detects embedding-model mismatches at startup.
"""
from __future__ import annotations

import logging
import threading
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from . import config

log = logging.getLogger(__name__)

_model_lock = threading.Lock()
_model: Optional[SentenceTransformer] = None
_model_name: Optional[str] = None

# ─── E5-family prefix requirements ───────────────────────────
# Models in the intfloat/multilingual-e5-* and intfloat/e5-* families
# require "query: " prefixed to queries and "passage: " to documents.
_E5_FAMILIES = (
    "intfloat/multilingual-e5",
    "intfloat/e5-",
    "intfloat/e5_",
)


def _needs_e5_prefix(model_name: str) -> bool:
    lower = model_name.lower()
    return any(lower.startswith(f.lower()) for f in _E5_FAMILIES)


def _prefix_query(text: str) -> str:
    if _needs_e5_prefix(config.ENV.EMBED_MODEL):
        return f"query: {text}"
    return text


def _prefix_passage(text: str) -> str:
    if _needs_e5_prefix(config.ENV.EMBED_MODEL):
        return f"passage: {text}"
    return text


# ─── Model loader ─────────────────────────────────────────────

def _load_model() -> SentenceTransformer:
    global _model, _model_name
    if _model is None:
        with _model_lock:
            if _model is None:
                name = config.ENV.EMBED_MODEL
                log.info("Loading embedding model: %s", name)
                _model = SentenceTransformer(name, device="cpu")
                _model_name = name
                log.info("Embedding model loaded: %s", name)
    return _model


# ─── Mismatch detection ───────────────────────────────────────

def check_model_consistency() -> bool:
    """
    Check that the configured embedding model matches the model used
    to build the Chroma collection.

    Returns True if consistent, False if mismatched.
    Logs a critical warning on mismatch — caller should re-index.
    """
    try:
        from . import vectorstore  # noqa: PLC0415
        coll_model = vectorstore.get_collection_embedding_model()
        if coll_model and coll_model != config.ENV.EMBED_MODEL:
            log.critical(
                "⚠️  EMBEDDING MODEL MISMATCH: collection was built with '%s', "
                "but EMBED_MODEL is now '%s'. "
                "Retrieval results will be INCORRECT. "
                "You must wipe and re-index the vector store: "
                "DELETE /api/pdfs for each document, then re-ingest.",
                coll_model,
                config.ENV.EMBED_MODEL,
            )
            return False
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not verify embedding model consistency: %s", exc)
    return True


# ─── Public API ───────────────────────────────────────────────

def embed(texts: List[str], is_query: bool = False) -> List[np.ndarray]:
    """
    Embed a list of texts.

    Args:
        texts:    List of text strings to embed.
        is_query: If True, apply query prefix (for retrieval queries).
                  If False, apply passage prefix (for document chunks).
    """
    if not texts:
        return []
    model = _load_model()
    batch_size = max(1, config.ENV.EMBED_BATCH)

    # Apply model-specific prefixes
    if is_query:
        prefixed = [_prefix_query(t) for t in texts]
    else:
        prefixed = [_prefix_passage(t) for t in texts]

    out: List[np.ndarray] = []
    for i in range(0, len(prefixed), batch_size):
        batch = prefixed[i : i + batch_size]
        vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        if isinstance(vecs, np.ndarray):
            out.extend([np.array(v, dtype=np.float32) for v in vecs])
        else:
            out.extend([np.array(v, dtype=np.float32) for v in vecs])
    return out


def embed_one(text: str, is_query: bool = True) -> np.ndarray:
    """Embed a single text. Defaults to query mode."""
    vecs = embed([text], is_query=is_query)
    if vecs:
        return vecs[0]
    # Return zero vector as fallback (will score 0 in cosine sim)
    return np.zeros((384,), dtype=np.float32)


def embed_dim() -> int:
    """Return the embedding dimension of the loaded model."""
    model = _load_model()
    return model.get_sentence_embedding_dimension() or 384


__all__ = [
    "embed",
    "embed_one",
    "embed_dim",
    "check_model_consistency",
]
