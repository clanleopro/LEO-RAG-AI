# app/services/chunking.py
from __future__ import annotations
from typing import List

def _approx_tokens(s: str) -> int:
    # quick & dirty token estimate ~ 4 chars / token
    return max(1, len(s) // 4)

def smart_chunk(text: str, max_tokens: int = 600, overlap_tokens: int = 120) -> List[str]:
    if not text:
        return []
    # split by double newline, then window with overlap
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        paras = [text.strip()]
    chunks: List[str] = []
    cur: List[str] = []
    cur_toks = 0
    for p in paras:
        t = _approx_tokens(p)
        if cur and cur_toks + t > max_tokens:
            chunks.append("\n\n".join(cur))
            # overlap: keep last overlap_tokens tokens (~ truncate by chars)
            keep = chunks[-1]
            keep_chars = overlap_tokens * 4
            cur = [keep[-keep_chars:]] if len(keep) > keep_chars else [keep]
            cur_toks = _approx_tokens("\n\n".join(cur))
        cur.append(p)
        cur_toks += t
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks
