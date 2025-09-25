# Copilot Instructions for LEO-Rigging-AI_MOD

## Project Overview
- This project is a modular Python application for document ingestion, processing, and retrieval-augmented generation (RAG) focused on rigging and lifting standards.
- The main entry point is `app/main.py`.
- Core logic is organized into `routers/` (API endpoints), `services/` (business logic, RAG, embeddings, vectorstore, OCR, etc.), and `models/` (schemas).
- Data and source documents are under `app/data/`.

## Key Architectural Patterns
- **Routers**: Each file in `routers/` exposes FastAPI endpoints for a specific domain (e.g., `query.py`, `ingest.py`, `search.py`).
- **Services**: Encapsulate business logic and integrations (e.g., `rag_service.py` for RAG, `vectorstore.py` for vector DB, `ocr.py` for OCR, `embeddings.py` for LLM embeddings).
- **Data Flow**: Documents are ingested (via `ingest.py`/`ingest_service.py`), chunked, embedded, and stored in a vector DB (`vectorstore/`). Queries are processed via RAG pipelines.
- **Extensibility**: New document types or LLM providers can be added by extending the relevant service modules.

## Developer Workflows
- **Run the app**: Use `run.bat` or run `main.py` directly (typically with `uvicorn app.main:app --reload`).
- **Ingest documents**: Use scripts in `scripts/` (e.g., `ingest_local.py`) or API endpoints in `routers/ingest.py`.
- **Add new endpoints**: Create a new file in `routers/`, register the router in `main.py`.
- **Testing**: No explicit test suite found; recommend using `pytest` and placing tests in a `tests/` directory if added.

## Project-Specific Conventions
- **File Structure**: Keep business logic in `services/`, API logic in `routers/`, and data models in `models/`.
- **Data Storage**: Vector DB files are in `app/data/vectorstore/` (e.g., `chroma.sqlite3`).
- **PDFs**: Source documents are in `app/data/source_pdfs/`.
- **Aliases**: Rigging aliases are managed in `app/data/rigging_aliases.json`.

## Integration Points
- **LLM/Embeddings**: See `services/llm.py` and `services/embeddings.py` for integration with language models and embedding providers.
- **OCR**: Handled in `services/ocr.py` for extracting text from PDFs.
- **Vector DB**: Managed in `services/vectorstore.py` using ChromaDB (see `chroma.sqlite3`).

## Examples
- To add a new ingestion pipeline, extend `services/ingest_service.py` and expose via `routers/ingest.py`.
- To add a new RAG workflow, modify `services/rag_service.py` and register endpoints in `routers/query.py`.

## See Also
- `README.md` (minimal, see code for details)
- `requirements.txt` for dependencies

---
For more details, review the code in `app/` and `scripts/`.
