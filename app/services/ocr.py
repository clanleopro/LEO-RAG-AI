from typing import Optional
import pytesseract
from PIL import Image
import fitz
from app.services import config

def ocr_page(page: "fitz.Page") -> str:
    dpi = float(getattr(config.ENV, "OCR_DPI_SCALE", 2.0))
    mat = fitz.Matrix(dpi, dpi)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    lang = getattr(config.ENV, "TESSERACT_LANGS", "eng") or "eng"
    text = pytesseract.image_to_string(img, lang=lang)
    return text or ""
