# app/services/ingest_service.py
from __future__ import annotations
from typing import List, Tuple
from pathlib import Path
import io
import logging
import time

import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from langdetect import detect as lang_detect, DetectorFactory
DetectorFactory.seed = 0

from . import config, chunking, vectorstore

log = logging.getLogger(__name__)
_MIN_TEXT_LEN = max(1, int(getattr(config.ENV, "MIN_EXTRACTED_TEXT", 25)))

def _extract_page_text(doc: fitz.Document, page_index: int) -> str:
    page = doc.load_page(page_index)
    try:
        text = page.get_text("text")
    except Exception:
        text = page.get_text()
    return (text or "").strip()

def _ocr_page(doc: fitz.Document, page_index: int) -> str:
    page = doc.load_page(page_index)
    dpi_scale = float(getattr(config.ENV, "OCR_DPI_SCALE", 2.0))
    mat = fitz.Matrix(dpi_scale, dpi_scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    with Image.open(io.BytesIO(pix.tobytes("png"))) as img:
        text = pytesseract.image_to_string(img, lang=config.ENV.TESSERACT_LANGS or "eng")
    return (text or "").strip()

def _split_into_chunks(text: str) -> List[str]:
    try:
        return chunking.smart_chunk(
            text,
            max_tokens=config.ENV.CHUNK_TOKENS,
            overlap_tokens=config.ENV.CHUNK_OVERLAP,
        )
    except Exception:
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not parts:
            parts = [text]
        max_chars = max(1000, config.ENV.CHUNK_TOKENS * 4)
        out: List[str] = []
        for p in parts:
            if len(p) <= max_chars:
                out.append(p)
            else:
                for i in range(0, len(p), max_chars):
                    out.append(p[i:i+max_chars])
        return out

def ingest_pdf(path: Path | str) -> Tuple[str, int]:
    path = Path(path)
    t0 = time.time()
    log.info("INGEST START: %s", path.name)

    with fitz.open(str(path)) as doc:
        page_total = doc.page_count
        all_chunks: List[vectorstore.Chunk] = []
        ocr_pages = 0

        for i in range(page_total):
            raw = _extract_page_text(doc, i)
            if len(raw) < _MIN_TEXT_LEN:
                try:
                    ocr_text = _ocr_page(doc, i)
                    if ocr_text:
                        raw = ocr_text
                        ocr_pages += 1
                except Exception as e:
                    log.warning("OCR failed on %s p.%d: %s", path.name, i + 1, e)
            if not raw:
                continue

            try:
                lang = lang_detect(raw[:4000])
            except Exception:
                lang = None

            for chunk_text in _split_into_chunks(raw):
                all_chunks.append(
                    vectorstore.Chunk(
                        id=None,
                        text=chunk_text,
                        source=path.name,
                        page=i + 1,
                        headings=None,
                        language=lang,
                        standard_code=None,
                        embedding=None,
                    )
                )

        count = vectorstore.upsert_chunks(all_chunks)
        took = time.time() - t0
        log.info(
            "INGEST DONE: %s | pages=%d, ocr_pages=%d, chunks_upserted=%d, took=%.3fs",
            path.name, page_total, ocr_pages, count, took
        )
    return (path.name, count)

def ingest_all_pdfs() -> List[Tuple[str, int]]:
    paths = sorted([p for p in config.SOURCE_PDFS.glob("*.pdf")])
    out: List[Tuple[str, int]] = []
    for p in paths:
        out.append(ingest_pdf(p))
    return out

def ingest_specific_files(files: List[Path]) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for p in files:
        out.append(ingest_pdf(p))
    return out
