# app/models/schemas.py
from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Dict, Any

class IngestResponse(BaseModel):
    files_processed: int
    pages_processed: int
    chunks_added: int

class QueryRequest(BaseModel):
    # Accept BOTH keys; we'll normalize so `question` is always set
    question: Optional[str] = Field(None, description="User question")
    query: Optional[str] = Field(None, description="Alias for 'question'")

    # Optional knobs
    stream: Optional[bool] = Field(None, description="SSE streaming")
    top_k: int = Field(5, ge=1, le=20)
    max_context_chars: int = Field(6000, ge=500, le=20000)
    filter_doc: Optional[str] = Field(None, description="Filter by source filename (optional)")

    @model_validator(mode="after")
    def _normalize(self):
        if (not self.question or not self.question.strip()) and self.query:
            self.question = self.query
        if not self.question or not self.question.strip():
            raise ValueError("Field 'question' (or 'query') is required")
        return self

class Citation(BaseModel):
    source: str
    page: int
    score: float
    snippet: str

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    used_provider: str
    meta: Dict[str, Any] = {}
