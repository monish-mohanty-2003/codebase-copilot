"""API request and response schemas."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    repo_path: str = Field(..., description="Absolute path to the repository.")
    repo_name: str | None = Field(None, description="Name to use in the index.")
    full: bool = Field(False, description="Force full re-index even if unchanged.")


class IndexResponse(BaseModel):
    repo: str
    files_scanned: int
    files_indexed: int
    files_skipped_unchanged: int
    chunks_embedded: int
    errors: list[str]


class SearchRequest(BaseModel):
    query: str
    k: int = 8
    language: str | None = None
    path_prefix: str | None = None


class SearchHitPayload(BaseModel):
    path: str
    symbol: str
    kind: str
    language: str
    start_line: int
    end_line: int
    preview: str
    score: float


class SearchResponse(BaseModel):
    hits: list[SearchHitPayload]


class ChatRequest(BaseModel):
    query: str
    stream: bool = False


class ChatResponse(BaseModel):
    answer: str
    steps: list[dict[str, Any]]
    elapsed_seconds: float
    model: str


class ToolCallRequest(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallResponse(BaseModel):
    name: str
    output: Any = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    repo_attached: str | None
    chunk_count: int
    model: str