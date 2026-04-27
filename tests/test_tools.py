"""Tool tests using a stub vector store and retriever where needed."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.indexing.vector_store import VectorStore
from app.models import ChunkKind, CodeChunk, ToolCall
from app.tools.base import ToolRegistry
from app.tools.dependency_graph import DependencyGraphTool
from app.tools.open_file import OpenFileTool


class StubStore(VectorStore):
    """In-memory VectorStore for tests, with a Chroma-shaped duck-typed `collection`."""

    def __init__(self, chunks: list[CodeChunk]):
        self._chunks = chunks
        # Mimic ChromaStore.collection.get() behavior
        self.collection = self  # type: ignore[assignment]

    def add(self, chunks, embeddings):  # noqa: ANN001
        self._chunks.extend(chunks)

    def query(self, query_embedding, k, where=None):  # noqa: ANN001
        return [(c, 0.5) for c in self._chunks[:k]]

    def delete_by_path(self, repo, path):  # noqa: ANN001
        self._chunks = [c for c in self._chunks if not (c.repo == repo and c.path == path)]

    def get_all_paths(self, repo):  # noqa: ANN001
        return {c.path: c.content_hash for c in self._chunks if c.repo == repo}

    def count(self, repo=None):  # noqa: ANN001
        if repo is None:
            return len(self._chunks)
        return sum(1 for c in self._chunks if c.repo == repo)

    # Methods to satisfy DependencyGraphTool's duck-typed access:
    def get(self, where: dict[str, Any], include: list[str]):
        chunks = [c for c in self._chunks if c.repo == where["repo"]]
        return {
            "ids": [c.id for c in chunks],
            "documents": [c.content for c in chunks],
            "metadatas": [c.to_metadata() for c in chunks],
        }

    @staticmethod
    def _hydrate(chunk_id, content, metadata):  # noqa: ANN001
        from app.indexing.vector_store import ChromaStore
        return ChromaStore._hydrate(chunk_id, content, metadata)


def _make_chunk(symbol: str, content: str, path: str = "x.py", kind: str = "function") -> CodeChunk:
    return CodeChunk(
        id=f"id-{symbol}",
        repo="test_repo",
        path=path,
        language="python",
        kind=ChunkKind(kind),
        symbol=symbol,
        start_line=1,
        end_line=10,
        content=content,
        content_hash="h",
    )


def test_open_file_returns_slice(tmp_repo: Path) -> None:
    tool = OpenFileTool(tmp_repo)
    out = tool.run(path="mod.py", start_line=1, end_line=3)
    assert out["path"] == "mod.py"
    assert out["start_line"] == 1
    assert out["content"].count("\n") <= 3


def test_open_file_rejects_path_traversal(tmp_repo: Path) -> None:
    tool = OpenFileTool(tmp_repo)
    with pytest.raises(ValueError):
        tool.run(path="../../../etc/passwd")


def test_dependency_graph_finds_dead_code() -> None:
    chunks = [
        _make_chunk("alive", "alive()\nused()\n"),
        _make_chunk("used", "def used(): return 1\n", kind="function"),
        _make_chunk("orphan", "def orphan(): return 2\n", kind="function", path="other.py"),
    ]
    store = StubStore(chunks)
    dep = DependencyGraphTool(store, repo="test_repo")
    result = dep.run()
    dead_symbols = {d["symbol"] for d in result["dead_code_candidates"]}
    assert "orphan" in dead_symbols
    assert "used" not in dead_symbols


def test_tool_registry_dispatches_unknown_gracefully() -> None:
    registry = ToolRegistry()
    import asyncio
    result = asyncio.run(registry.dispatch(ToolCall(name="nope", arguments={})))
    assert result.error is not None and "Unknown" in result.error