# Dockerfile
FROM python:3.11-slim

# ---- System packages needed by Tesseract & PyMuPDF ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    # add extra languages if you’ll OCR Arabic/Hindi/etc.
    # tesseract-ocr-ara tesseract-ocr-hin \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy and install Python deps first (leverages Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source
COPY . .

# Run as non-root (optional but good practice)
# RUN useradd -m appuser && chown -R appuser:appuser /app
# USER appuser

# The FastAPI app listens on 8000
EXPOSE 8000

# Start the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
