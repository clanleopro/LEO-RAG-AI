# app/services/chunking.py
"""
Token-aware text chunking using tiktoken for accurate token counts.

Chunks on sentence/paragraph boundaries to avoid splitting mid-sentence,
mid-table, or mid-formula. Preserves standards codes, model numbers,
units, and acronyms.
"""
from __future__ import annotations

import re
import logging
from typing import List, Optional

log = logging.getLogger(__name__)

# ─── Tokenizer ────────────────────────────────────────────────

def _get_tokenizer():
    """Load tiktoken encoder. Falls back to char-based estimate if unavailable."""
    try:
        import tiktoken
        # cl100k_base is used by gpt-4, gpt-3.5-turbo, and text-embedding-ada-002
        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        log.warning("tiktoken unavailable (%s); using char/4 approximation.", exc)
        return None


_enc = _get_tokenizer()


def _count_tokens(text: str) -> int:
    if _enc is not None:
        return len(_enc.encode(text, disallowed_special=()))
    return max(1, len(text) // 4)


# ─── Sentence splitting ───────────────────────────────────────

# Split on sentence endings, but be conservative: require capital after period.
# Preserves abbreviations like "e.g.", standards codes "ASME B30.9", "SWL", "WLL".
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\d])"
)


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences conservatively.
    Avoids splitting inside standards codes, abbreviations, and units.
    """
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_paragraphs(text: str) -> List[str]:
    """Split on double newlines (paragraphs), then fall back to sentences."""
    paragraphs: List[str] = []
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if para:
            paragraphs.append(para)
    return paragraphs if paragraphs else [text.strip()]


# ─── Main chunking function ───────────────────────────────────

def smart_chunk(
    text: str,
    max_tokens: int = 600,
    overlap_tokens: int = 120,
) -> List[str]:
    """
    Split text into overlapping chunks, respecting token limits.

    Strategy:
    1. Split on paragraph boundaries.
    2. If a paragraph exceeds max_tokens, split on sentence boundaries.
    3. Pack paragraphs/sentences into chunks until the limit is reached.
    4. Add overlap by appending the last N tokens of the previous chunk.

    The function never blindly truncates mid-sentence, mid-table, or
    mid-formula. Each chunk is a complete unit of paragraphs/sentences.
    """
    if not text or not text.strip():
        return []

    # Step 1: paragraph segments
    paragraphs = _split_paragraphs(text)

    # Step 2: further split oversized paragraphs on sentence boundaries
    segments: List[str] = []
    for para in paragraphs:
        if _count_tokens(para) > max_tokens:
            sentences = _split_sentences(para)
            segments.extend(sentences)
        else:
            segments.append(para)

    if not segments:
        return [text.strip()[:max_tokens * 4]]  # last-resort fallback

    # Step 3: pack segments into chunks
    chunks: List[str] = []
    current_parts: List[str] = []
    current_tokens = 0
    overlap_text: Optional[str] = None

    for seg in segments:
        seg_tokens = _count_tokens(seg)

        # A single segment larger than max_tokens → emit as its own chunk
        if seg_tokens > max_tokens:
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                overlap_text = _tail(current_parts, overlap_tokens)
                current_parts = []
                current_tokens = 0

            # Include overlap prefix if available
            if overlap_text:
                chunk = overlap_text + "\n\n" + seg
            else:
                chunk = seg
            chunks.append(chunk)
            overlap_text = _tail([seg], overlap_tokens)
            continue

        # Would this segment push us over the limit?
        if current_parts and current_tokens + seg_tokens > max_tokens:
            chunks.append("\n\n".join(current_parts))
            overlap_text = _tail(current_parts, overlap_tokens)
            # Start new chunk with overlap
            if overlap_text:
                current_parts = [overlap_text]
                current_tokens = _count_tokens(overlap_text)
            else:
                current_parts = []
                current_tokens = 0

        current_parts.append(seg)
        current_tokens += seg_tokens

    # Flush remaining
    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return [c for c in chunks if c.strip()]


def _tail(parts: List[str], overlap_tokens: int) -> str:
    """
    Return the last `overlap_tokens` tokens worth of text from the
    given list of parts, preserving complete sentences where possible.
    """
    if not parts:
        return ""
    joined = "\n\n".join(parts)
    if _count_tokens(joined) <= overlap_tokens:
        return joined

    # Work backwards through sentences
    sentences = _split_sentences(joined)
    tail_parts: List[str] = []
    tail_tokens = 0
    for sent in reversed(sentences):
        t = _count_tokens(sent)
        if tail_tokens + t > overlap_tokens:
            break
        tail_parts.insert(0, sent)
        tail_tokens += t

    return " ".join(tail_parts) if tail_parts else ""


__all__ = ["smart_chunk"]
