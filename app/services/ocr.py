from typing import Optional
import pytesseract
from PIL import Image
import fitz
from app.services.config import TESSERACT_LANGS, OCR_DPI_SCALE

def ocr_page(page: "fitz.Page") -> str:
    mat = fitz.Matrix(OCR_DPI_SCALE, OCR_DPI_SCALE)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    text = pytesseract.image_to_string(img, lang=TESSERACT_LANGS)
    return text or ""
