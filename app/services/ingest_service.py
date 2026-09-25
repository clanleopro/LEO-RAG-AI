# app/services/ingest_service.py
"""
PDF ingestion pipeline.

Extracts text (native or OCR), chunks, embeds, and upserts into the
vector store. Supports re-ingestion (stale chunks are removed first).

Security controls:
- Encrypted PDFs are rejected.
- Page count and pixel limits are enforced.
- Checksum, ingestion timestamp, embedding model, and page count are
  stored in chunk metadata.
"""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
from langdetect import DetectorFactory, detect as lang_detect

DetectorFactory.seed = 0

from . import config, chunking, vectorstore

log = logging.getLogger(__name__)


class IngestionError(Exception):
    """Raised for controlled ingestion failures."""


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _extract_page_text(doc: fitz.Document, page_index: int) -> str:
    page = doc.load_page(page_index)
    try:
        text = page.get_text("text")
    except Exception:
        text = page.get_text()
    return (text or "").strip()


def _ocr_page(doc: fitz.Document, page_index: int) -> str:
    """OCR a page, enforcing pixel limit."""
    try:
        import io
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise IngestionError(
            "pytesseract or Pillow is not installed. OCR is unavailable."
        ) from exc

    page = doc.load_page(page_index)
    dpi_scale = config.ENV.OCR_DPI_SCALE
    mat = fitz.Matrix(dpi_scale, dpi_scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    pixel_count = pix.width * pix.height
    if pixel_count > config.ENV.OCR_MAX_PIXELS:
        log.warning(
            "OCR page %d pixel count %d exceeds limit %d — skipping OCR for this page.",
            page_index + 1,
            pixel_count,
            config.ENV.OCR_MAX_PIXELS,
        )
        return ""

    with Image.open(io.BytesIO(pix.tobytes("png"))) as img:
        text = pytesseract.image_to_string(img, lang=config.ENV.TESSERACT_LANGS or "eng")
    return (text or "").strip()


def _detect_lang(text: str) -> Optional[str]:
    try:
        return lang_detect(text[:4000])
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Main ingest function
# ─────────────────────────────────────────────────────────────

def ingest_pdf(
    path: Path | str,
    job_id: Optional[str] = None,
    doc_id: Optional[str] = None,
    version: Optional[str] = None,
) -> Tuple[str, int, int]:
    """
    Ingest a single PDF into the vector store.

    1. Validates the file (not encrypted, page count within limits).
    2. Deletes any existing chunks for this source (re-ingestion cleanup).
    3. Extracts text per page (native → OCR fallback).
    4. Chunks, embeds, and upserts to Chroma.
    5. Stores checksum, timestamp, and embedding model in metadata.

    Returns (source_name, chunks_upserted, pages_processed).
    Raises IngestionError on controlled failures.
    """
    path = Path(path)
    if not path.exists():
        raise IngestionError(f"File not found: {path.name}")

    t0 = time.time()
    source_name = path.name
    log.info("INGEST START: %s (doc_id=%s)", source_name, doc_id)

    checksum = _sha256_file(path)
    ingested_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    embed_model = config.ENV.EMBED_MODEL

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise IngestionError(f"Cannot open PDF: {exc}") from exc

    with doc:
        # ── Encryption check ─────────────────────────────────
        if doc.is_encrypted:
            raise IngestionError(
                f"'{source_name}' is encrypted. "
                "Please decrypt or remove the password before ingestion."
            )

        page_total = doc.page_count

        # ── Page count check ─────────────────────────────────
        max_pages = config.ENV.MAX_PDF_PAGES
        if page_total > max_pages:
            raise IngestionError(
                f"'{source_name}' has {page_total} pages, exceeding the "
                f"maximum of {max_pages}. Split the document before ingesting."
            )

        # ── Remove stale chunks (re-ingestion) ───────────────
        removed = vectorstore.delete_by_source(source_name)
        if removed:
            log.info("Re-ingestion: removed %d stale chunks for '%s'.", removed, source_name)

        min_text_len = max(1, config.ENV.MIN_EXTRACTED_TEXT)
        ocr_pages = 0
        pages_processed = 0
        all_chunks: List[vectorstore.Chunk] = []

        for i in range(page_total):
            raw = _extract_page_text(doc, i)

            if len(raw) < min_text_len:
                # Attempt OCR
                try:
                    ocr_text = _ocr_page(doc, i)
                    if ocr_text:
                        raw = ocr_text
                        ocr_pages += 1
                except IngestionError as exc:
                    log.warning("OCR unavailable for %s p.%d: %s", source_name, i + 1, exc)
                except Exception as exc:
                    log.warning("OCR failed on %s p.%d: %s", source_name, i + 1, exc)

            if not raw:
                continue

            pages_processed += 1
            lang = _detect_lang(raw)

            for chunk_text in chunking.smart_chunk(
                raw,
                max_tokens=config.ENV.CHUNK_TOKENS,
                overlap_tokens=config.ENV.CHUNK_OVERLAP,
            ):
                if not chunk_text.strip():
                    continue
                all_chunks.append(
                    vectorstore.Chunk(
                        id=None,
                        text=chunk_text,
                        source=source_name,
                        doc_id=doc_id or source_name,
                        page=i + 1,
                        headings=None,
                        language=lang,
                        standard_code=None,
                        embedding=None,
                        checksum=checksum,
                        ingested_at=ingested_at,
                        embedding_model=embed_model,
                        version=version,
                    )
                )

    # Upsert outside the with-doc block
    count = vectorstore.upsert_chunks(all_chunks)
    took = time.time() - t0
    log.info(
        "INGEST DONE: %s | pages=%d, ocr_pages=%d, chunks_upserted=%d, took=%.3fs",
        source_name,
        page_total,
        ocr_pages,
        count,
        took,
    )
    return (source_name, count, pages_processed)


def ingest_all_pdfs() -> List[Tuple[str, int, int]]:
    """Ingest all PDFs in SOURCE_PDFS directory."""
    paths = sorted(config.SOURCE_PDFS.glob("*.pdf"))
    results: List[Tuple[str, int, int]] = []
    for p in paths:
        try:
            results.append(ingest_pdf(p))
        except IngestionError as exc:
            log.error("Ingestion skipped for %s: %s", p.name, exc)
    return results


def ingest_specific_files(
    files: List[Path],
    doc_ids: Optional[List[str]] = None,
) -> List[Tuple[str, int, int]]:
    """Ingest a specific list of PDF paths."""
    results: List[Tuple[str, int, int]] = []
    for idx, p in enumerate(files):
        did = doc_ids[idx] if doc_ids and idx < len(doc_ids) else None
        try:
            results.append(ingest_pdf(p, doc_id=did))
        except IngestionError as exc:
            log.error("Ingestion failed for %s: %s", p.name, exc)
    return results
