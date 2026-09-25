# LEO Rigging AI

LEO Rigging AI is a retrieval-augmented generation (RAG) backend application for the rigging, lifting, crane, and hoisting industry.

> **⚠️ SECURITY NOTICE:**
> This repository previously contained an exposed OpenAI API key in its Git history. If you have cloned or forked this repository, **you must ensure the key has been revoked** before deploying or sharing the code further. See [SECURITY.md](SECURITY.md) for immediate actions.

## Overview

This service provides an API for:
- **Secure PDF Ingestion**: Upload, validate, OCR (via Tesseract), and chunk documents.
- **Hybrid Retrieval**: Semantic search (SentenceTransformers) + Lexical search (BM25) with cross-model Maximum Marginal Relevance (MMR) for diversity.
- **Grounded Q&A**: Domain-specific constraint enforcement. The assistant only answers using retrieved context. It is strictly instructed not to use external knowledge for factual claims and enforces a minimum relevance threshold.

## Documentation

- [SECURITY.md](SECURITY.md): Vulnerability reporting and security controls.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): System architecture and data flow.
- [docs/API.md](docs/API.md): Endpoint documentation and authentication.
- [docs/INGESTION.md](docs/INGESTION.md): Details on the OCR and chunking pipeline.
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md): Production deployment guide.
- [docs/INCIDENT_RESPONSE.md](docs/INCIDENT_RESPONSE.md): Guidelines for addressing unsafe AI outputs.

## Quick Start (Local Development)

1. Ensure Python 3.11+ is installed.
2. Install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
3. Configure environment:
   ```bash
   cp .env.example .env
   ```
   *Edit `.env` to add your `OPENAI_API_KEY` and an `API_KEYS` mapping.*
4. Run the server:
   ```bash
   uvicorn app.main:app --reload
   ```

## Testing

Run the test suite offline (no real API calls are made):
```bash
pytest tests/ -v
```

## Disclaimer

This application is strictly for **educational purposes**. It does not replace competent person inspections, manufacturer instructions, engineered lift plans, or adherence to applicable regulations (e.g., OSHA, ASME).
