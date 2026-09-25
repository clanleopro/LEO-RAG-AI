# System Architecture

LEO Rigging AI is a modular FastAPI application built for retrieval-augmented generation (RAG) over PDFs.

## Components

1.  **API Layer (`app/routers/`)**: FastAPI endpoints for upload, ingestion, querying, and file management. Secured by role-based bearer token authentication.
2.  **Core (`app/core/`)**: Authentication and authorization middleware.
3.  **Services (`app/services/`)**:
    *   `config.py`: Environment variable validation using `pydantic-settings`.
    *   `job_store.py`: In-memory thread-safe background job queue.
    *   `upload.py`: Secure file handling (validation, sanitization, atomic moves).
    *   `ingest_service.py`: Pipeline for PDF extraction, OCR fallback, and chunking.
    *   `chunking.py`: Semantic text splitting respecting sentence boundaries.
    *   `embeddings.py`: Sentence-transformer wrappers (e.g., multilingual-e5).
    *   `vectorstore.py`: Hybrid retrieval system (ChromaDB for dense vectors, BM25 for lexical). Implements real cosine-similarity MMR.
    *   `rag_service.py`: Orchestrates retrieval, builds grounded context, and enforces safety boundaries.
    *   `llm.py`: Resilient OpenAI client wrapper with retry logic and error sanitization.
4.  **Models (`app/models/`)**: Strict Pydantic schemas for request/response validation.

## Data Flow (Query)
1.  User submits a question.
2.  `rag_service.retrieve` expands the query and performs hybrid search (dense + BM25).
3.  Candidate chunks are de-duplicated and re-ranked using MMR (diversity).
4.  Chunks below `MIN_RELEVANCE_SCORE` are discarded.
5.  If chunks remain, they are formatted into a context block with prompt-injection delimiters.
6.  The LLM generates an answer grounded *strictly* in the provided context.
7.  A response with verified citations is returned.

## Storage
-   **Source PDFs**: `/app/data/source_pdfs/`
-   **Vector Database**: `/app/data/vectorstore/` (ChromaDB SQLite and parquet files)
