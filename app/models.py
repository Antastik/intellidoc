"""Pydantic request/response schemas — the public contract of the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """One retrieved chunk that supported the answer."""
    doc_id: str
    source: str
    page: int | None = None
    chunk_index: int
    score: float = Field(..., description="Similarity score (higher = more relevant).")
    snippet: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(None, ge=1, le=20)
    doc_ids: list[str] | None = Field(
        None, description="Restrict retrieval to these document ids. None = all docs."
    )


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    model: str


class DocumentInfo(BaseModel):
    doc_id: str
    source: str
    pages: int
    chunks: int


class UploadResponse(BaseModel):
    documents: list[DocumentInfo]


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    embedding_provider: str
    documents: int
    chunks: int
