"""Tests for tree-sitter parsing and chunking. No network or Ollama needed."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.indexing.chunker import Chunker, count_tokens
from app.indexing.parser import TreeSitterParser
from app.models import ChunkKind


def test_parser_extracts_python_definitions(tmp_repo: Path) -> None:
    parser = TreeSitterParser()
    language, defs, imports = parser.parse_file(tmp_repo / "mod.py")
    assert language == "python"
    symbols = {d.symbol for d in defs}
    assert "add" in symbols
    assert "Greeter" in symbols
    assert "Greeter.hello" in symbols
    assert any(d.kind == "class" for d in defs)
    assert any(d.kind == "method" for d in defs)


def test_chunker_produces_module_and_definition_chunks(tmp_repo: Path) -> None:
    parser = TreeSitterParser()
    chunker = Chunker(parser, max_tokens=500, overlap_tokens=50)
    chunks = chunker.chunk_file("test_repo", tmp_repo, tmp_repo / "mod.py")
    kinds = [c.kind for c in chunks]
    assert ChunkKind.MODULE in kinds
    assert ChunkKind.FUNCTION in kinds or ChunkKind.METHOD in kinds
    # Every chunk must have non-empty content and stable id
    assert all(c.content.strip() for c in chunks)
    ids = [c.id for c in chunks]
    assert len(set(ids)) == len(ids)


def test_chunker_splits_oversized_function(tmp_path: Path) -> None:
    big_func = "def big():\n    " + "x = 1\n    " * 800
    file_path = tmp_path / "big.py"
    file_path.write_text(big_func)

    parser = TreeSitterParser()
    chunker = Chunker(parser, max_tokens=200, overlap_tokens=20)
    chunks = chunker.chunk_file("test_repo", tmp_path, file_path)
    parts = [c for c in chunks if c.kind == ChunkKind.OVERSIZED_PART]
    assert len(parts) > 1
    # Every part should fit the budget
    for c in parts:
        assert count_tokens(c.content) <= 250  # small slack for signature line


def test_unsupported_language_returns_no_chunks(tmp_path: Path) -> None:
    parser = TreeSitterParser()
    chunker = Chunker(parser)
    file_path = tmp_path / "readme.md"
    file_path.write_text("# hello")
    assert chunker.chunk_file("r", tmp_path, file_path) == []


def test_chunk_metadata_roundtrip(tmp_repo: Path) -> None:
    """to_metadata() output must be flat scalars (Chroma constraint)."""
    parser = TreeSitterParser()
    chunker = Chunker(parser)
    chunks = chunker.chunk_file("test_repo", tmp_repo, tmp_repo / "mod.py")
    for c in chunks:
        meta = c.to_metadata()
        for v in meta.values():
            assert isinstance(v, (str, int, float, bool))


@pytest.mark.parametrize("text,expected_min", [("hello world", 1), ("def f():\n    pass", 4)])
def test_count_tokens(text: str, expected_min: int) -> None:
    assert count_tokens(text) >= expected_min
 