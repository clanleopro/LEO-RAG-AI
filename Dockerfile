# =====================================================================
# Stage 1: The "Builder" Stage
#
# This stage installs all dependencies, including heavy build tools,
# and compiles the Python packages.
# =====================================================================
FROM python:3.11-bullseye AS builder

# Install build-time system dependencies
# These are needed to install Python packages but not to run the app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
# This leverages Docker's cache. The packages will be compiled here.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code
COPY . .

# =====================================================================
# Stage 2: The "Final" Stage
#
# This is the lean, production-ready image. We start from a fresh
# Python base and copy only the necessary artifacts from the builder.
# =====================================================================
FROM python:3.11-bullseye

# Install RUNTIME system dependencies ONLY
# Notice 'build-essential' and 'git' are gone.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Copy Artifacts from Builder Stage ---
# Copy the installed Python packages from the 'builder' stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
# Copy the application code from the 'builder' stage
COPY --from=builder /app /app

# --- Runtime user (security best practice) ---
# Create a non-root user to run the application
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose FastAPI port
EXPOSE 8000

# --- Entrypoint ---
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]