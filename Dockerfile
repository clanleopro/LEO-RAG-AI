# Use a slightly larger base (bullseye) for easier builds
FROM python:3.11-bullseye

# --- System dependencies (Tesseract, PyMuPDF, langdetect, chromadb) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    build-essential \
    libgl1 \
    pkg-config \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working dir
WORKDIR /app

# --- Dependency installation ---
# Copy only requirements first (for Docker cache)
COPY requirements.txt .

# Upgrade pip/setuptools/wheel to latest
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy source code ---
COPY . .

# --- Runtime user (security best practice) ---
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose FastAPI port
EXPOSE 8000

# --- Entrypoint ---
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
