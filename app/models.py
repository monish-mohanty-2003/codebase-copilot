"""Domain models. Single source of truth for shapes that cross module boundaries."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChunkKind(str, Enum):
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    MODULE = "module"
    OVERSIZED_PART = "oversized_part"


class CodeChunk(BaseModel):
    """A semantically meaningful slice of source code, ready to embed."""

    id: str
    repo: str
    path: str
    language: str
    kind: ChunkKind
    symbol: str
    parent_symbol: str | None = None
    start_line: int
    end_line: int
    content: str
    content_hash: str
    imports: list[str] = Field(default_factory=list)
    calls: list[str] = Field(default_factory=list)
    git_commit: str | None = None
    indexed_at: int = 0

    def to_metadata(self) -> dict[str, Any]:
        """Flat metadata dict for vector store filtering. Chroma forbids nested values."""
        return {
            "repo": self.repo,
            "path": self.path,
            "language": self.language,
            "kind": self.kind.value,
            "symbol": self.symbol,
            "parent_symbol": self.parent_symbol or "",
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content_hash": self.content_hash,
            "imports": ",".join(self.imports),
            "calls": ",".join(self.calls),
            "git_commit": self.git_commit or "",
            "indexed_at": self.indexed_at,
        }

    def embed_text(self) -> str:
        """The text we actually feed the embedder.

        We blend symbol+kind+language with the code so the symbol name
        carries semantic weight in the vector — pure code under-weights identifiers.
        """
        header = f"{self.language} {self.kind.value} {self.symbol}"
        if self.parent_symbol:
            header = f"{header} (in {self.parent_symbol})"
        return f"{header}\n{self.content}"


class SearchHit(BaseModel):
    """A retrieval result returned to the agent or end user."""

    chunk: CodeChunk
    score: float
    source: Literal["vector", "bm25", "fused"] = "fused"


class ToolCall(BaseModel):
    """An agent's structured request to invoke a tool."""

    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    """Output of a tool execution. `error` set on failure."""

    name: str
    output: Any = None
    error: str | None = None


class AgentStep(BaseModel):
    """One iteration of the ReAct loop. Useful for tracing and UI display."""

    step: int
    thought: str | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    final_answer: str | None = None


class AgentTrace(BaseModel):
    """Complete record of an agent run, including final answer."""

    query: str
    steps: list[AgentStep] = Field(default_factory=list)
    answer: str = ""
    elapsed_seconds: float = 0.0
    model: str = ""