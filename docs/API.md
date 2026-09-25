# API Documentation

All `/api/*` endpoints require authentication via the `Authorization: Bearer <key>` or `X-API-Key: <key>` header.

## Scopes

-   `query`: Can ask questions and perform searches.
-   `documents:read`: Can list and download documents.
-   `documents:write`: Can upload and ingest documents (implies `documents:read`).
-   `documents:delete`: Can delete documents (implies `documents:read`).
-   `admin`: Full access, plus system info.

## Endpoints

### Query
-   `POST /api/query`: Ask a question. Returns answer + citations. (Scope: `query`)
-   `POST /api/search`: Raw hybrid search. Returns chunks. (Scope: `query`)

### Documents
-   `POST /api/upload`: Upload PDFs. Returns job_id. (Scope: `documents:write`)
-   `GET /api/pdfs`: List ingested documents. (Scope: `documents:read`)
-   `GET /api/pdfs/{doc_id}`: Download a PDF. (Scope: `documents:read`)
-   `DELETE /api/pdfs/{doc_id}`: Delete a PDF and its vectors. (Scope: `documents:delete`)

### Ingestion & Jobs
-   `POST /api/ingest`: Trigger bulk ingestion. Returns job_id. (Scope: `documents:write`)
-   `GET /api/ingest/stream`: Real-time SSE progress for bulk ingestion. (Scope: `documents:write`)
-   `GET /api/jobs/{job_id}`: Check background job status. (Scope: `admin`)

### Health & Admin
-   `GET /health/live`: Liveness probe. (Public)
-   `GET /health/ready`: Readiness probe. (Public)
-   `GET /info`: System configuration. (Scope: `admin`)
-   `GET /routes`: List all registered routes. (Scope: `admin`)
