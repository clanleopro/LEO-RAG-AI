# app/services/embeddings.py
from __future__ import annotations
from typing import List
import threading
import numpy as np
from sentence_transformers import SentenceTransformer
from . import config

_model_lock = threading.Lock()
_model: SentenceTransformer | None = None

def _load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(config.ENV.EMBED_MODEL, device="cpu")
    return _model

def embed(texts: List[str]) -> List[np.ndarray]:
    if not texts:
        return []
    model = _load_model()
    batch = max(1, config.ENV.EMBED_BATCH)
    out: List[np.ndarray] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        vecs = model.encode(chunk, normalize_embeddings=True, show_progress_bar=False)
        if isinstance(vecs, np.ndarray):
            out.extend([np.array(v, dtype=np.float32) for v in vecs])
        else:
            out.extend([np.array(v, dtype=np.float32) for v in vecs])
    return out

def embed_one(text: str) -> np.ndarray:
    vecs = embed([text])
    return vecs[0] if vecs else np.zeros((384,), dtype=np.float32)
