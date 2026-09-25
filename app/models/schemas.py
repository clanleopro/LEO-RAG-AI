# app/models/schemas.py
"""
Validated Pydantic request/response models for LEO Rigging AI.

All request models enforce strict bounds to prevent abuse.
Response models never expose filesystem paths or internal details.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────
# Query
# ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """POST /api/query — ask a question against ingested documents."""

    question: Optional[str] = Field(None, description="User question")
    query: Optional[str] = Field(None, description="Alias for 'question'")

    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of retrieved chunks to include in context",
    )
    max_context_chars: int = Field(
        default=12000,
        ge=500,
        le=32000,
        description="Maximum characters of context to send to the LLM",
    )
    filter_doc: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Limit retrieval to a specific document ID",
    )
    stream: Optional[bool] = Field(
        default=False,
        description="Whether to stream the response via SSE",
    )

    @model_validator(mode="after")
    def _normalize_question(self) -> "QueryRequest":
        """Accept either 'question' or 'query'; require at least one."""
        q = self.question or self.query or ""
        q = q.strip()
        if not q:
            raise ValueError(
                "Field 'question' (or 'query') is required and must not be empty or whitespace."
            )
        if len(q) > 2000:
            raise ValueError(
                f"Question is too long ({len(q)} chars). Maximum is 2000 characters."
            )
        self.question = q
        self.query = q
        return self

    @field_validator("filter_doc")
    @classmethod
    def _sanitize_filter_doc(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        # Reject path separators or relative components
        bad = {"\\", "/", "..", "~"}
        for char in bad:
            if char in v:
                raise ValueError(
                    "'filter_doc' must be a plain document ID, not a path."
                )
        return v or None


class Citation(BaseModel):
    """A citation corresponding to a chunk actually sent to the model."""

    n: int = Field(..., description="Citation number used in the answer text, e.g. [1]")
    doc_id: str = Field(..., description="Server-assigned document identifier")
    title: str = Field(..., description="Display filename / document title")
    page: int = Field(..., description="1-based page number")
    revision: Optional[str] = Field(None, description="Document revision/version if stored")
    snippet: str = Field(..., description="Short excerpt from the retrieved chunk")
    score_vec: float = Field(..., description="Normalized dense retrieval score")
    score_bm25: float = Field(..., description="Normalized BM25 score")
    score_blended: float = Field(..., description="Final blended relevance score")


class QueryResponse(BaseModel):
    """POST /api/query — response."""

    answer: str
    citations: List[Citation]
    low_confidence: bool = Field(
        default=False,
        description="True if retrieval scores were below the configured threshold "
                    "and a grounded refusal was returned.",
    )
    grounded: bool = Field(
        default=True,
        description="True if the answer is supported by retrieved context.",
    )
    used_provider: str
    meta: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    """POST /api/search — semantic/hybrid search without LLM generation."""

    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def _require_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("'query' must not be empty or whitespace.")
        return v.strip()


class SearchHitResponse(BaseModel):
    doc_id: str
    title: str
    page: int
    score_vec: float
    score_bm25: float
    score_blended: float
    snippet: str


class SearchResponse(BaseModel):
    results: List[SearchHitResponse]


# ─────────────────────────────────────────────────────────────
# Upload / Ingest
# ─────────────────────────────────────────────────────────────

class UploadedFile(BaseModel):
    doc_id: str = Field(..., description="Server-generated document identifier")
    original_filename: str = Field(..., description="Sanitized display name (not a path)")
    size_bytes: int
    page_count: int
    job_id: str = Field(..., description="Background ingestion job ID")


class UploadResponse(BaseModel):
    uploaded: List[UploadedFile]
    message: str = "Upload accepted. Ingestion running in background."


class IngestResult(BaseModel):
    doc_id: str
    title: str
    chunks_upserted: int
    pages_processed: int
    took_seconds: float


class IngestResponse(BaseModel):
    ingested: List[IngestResult]
    total_chunks: int
    file_count: int
    took_seconds: float


# ─────────────────────────────────────────────────────────────
# Background Jobs
# ─────────────────────────────────────────────────────────────

class JobStatusResponse(BaseModel):
    job_id: str
    status: str   # pending | running | done | failed
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# File management
# ─────────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    doc_id: str
    title: str
    size_bytes: int
    page_count: int
    checksum: Optional[str] = None
    ingested_at: Optional[str] = None
    embedding_model: Optional[str] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int


class DeleteResponse(BaseModel):
    doc_id: str
    title: str
    vectors_removed: int
    file_deleted: bool


# ─────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────

class LivenessResponse(BaseModel):
    status: str = "ok"


class ReadinessResponse(BaseModel):
    status: str          # "ready" | "degraded" | "not_ready"
    checks: Dict[str, str]  # component → status string


# ─────────────────────────────────────────────────────────────
# Error
# ─────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Generic safe error response — never exposes paths or stack traces."""
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None


__all__ = [
    "QueryRequest",
    "QueryResponse",
    "Citation",
    "SearchRequest",
    "SearchResponse",
    "SearchHitResponse",
    "UploadResponse",
    "UploadedFile",
    "IngestResponse",
    "IngestResult",
    "JobStatusResponse",
    "DocumentInfo",
    "DocumentListResponse",
    "DeleteResponse",
    "LivenessResponse",
    "ReadinessResponse",
    "ErrorResponse",
]
