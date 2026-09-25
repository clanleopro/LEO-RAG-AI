# Deployment Guide

LEO Rigging AI is designed to be deployed as a Docker container.

## Requirements

- Docker and Docker Compose
- Minimum 2GB RAM (4GB recommended for OCR/Embeddings)
- Persistent volume for `/app/app/data`

## Production Configuration

Create a `.env` file based on `.env.example`.

**Critical Settings:**
- `ENVIRONMENT=production`: Disables OpenAPI docs, enables strict security headers.
- `AUTH_ENABLED=true`: Enforces API key checks.
- `API_KEYS`: Define your access keys (e.g., `my_secret_key:admin`).
- `OPENAI_API_KEY`: Your valid OpenAI API key.
- `CORS_ALLOW_ORIGINS`: Explicit list of frontend origins (e.g., `https://my-frontend.com`). DO NOT USE `*`.

## Docker Compose Example

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - leo_data:/app/app/data
    restart: always

volumes:
  leo_data:
```

## Scaling Limitations

The current BM25 lexical index and the background job store (`app/services/job_store.py`) are **in-memory and process-local**.

**Do not run multiple workers (e.g., `gunicorn --workers 4`) or multiple container replicas** without replacing these components, as state will become inconsistent across workers. For production, run a single worker/container, or migrate to a Redis-backed queue (Celery/ARQ) and a distributed lexical index (Elasticsearch).
