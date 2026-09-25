# Dockerfile for LEO Rigging AI
# Multi-stage build for production.

# ─── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.11-slim as builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ─── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.11-slim

# Install system dependencies (Tesseract for OCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /install /usr/local

# Create non-root user
RUN useradd -m -u 1000 leo

# Create required directories with correct permissions
RUN mkdir -p /app/app/data/source_pdfs \
             /app/app/data/processed \
             /app/app/data/vectorstore \
    && chown -R leo:leo /app/app/data

# Copy application code
COPY --chown=leo:leo . /app

# Switch to non-root user
USER leo

# Default environment variables
ENV PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health/live || exit 1

EXPOSE ${PORT}

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
