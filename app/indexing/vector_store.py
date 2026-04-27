"""Vector store abstraction.

Why an abstraction: we want to swap Chroma → Qdrant → FAISS without touching
indexing or retrieval code. The interface is deliberately narrow.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.models import CodeChunk

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[CodeChunk], embeddings: list[list[float]]) -> None: ...

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[CodeChunk, float]]:
        """Return list of (chunk, distance). Lower distance = more similar."""

    @abstractmethod
    def delete_by_path(self, repo: str, path: str) -> None: ...

    @abstractmethod
    def get_all_paths(self, repo: str) -> dict[str, str]:
        """Return {path: latest content_hash} for incremental indexing decisions."""

    @abstractmethod
    def count(self, repo: str | None = None) -> int: ...


class ChromaStore(VectorStore):
    """Chroma persistent client. One collection per deployment, repo as filter."""

    COLLECTION = "code_chunks"

    def __init__(self, persist_dir: Path):
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
        )
        # We supply embeddings ourselves, so disable Chroma's default ef
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[CodeChunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        ids = [c.id for c in chunks]
        # Chroma upserts on id; safe to call repeatedly with same content
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=[c.content for c in chunks],
            metadatas=[c.to_metadata() for c in chunks],
        )

    def query(
        self,
        query_embedding: list[float],
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[CodeChunk, float]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
        )
        hits: list[tuple[CodeChunk, float]] = []
        if not results["ids"] or not results["ids"][0]:
            return hits

        for i, chunk_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            content = results["documents"][0][i]
            distance = results["distances"][0][i]
            chunk = self._hydrate(chunk_id, content, metadata)
            hits.append((chunk, float(distance)))
        return hits

    def delete_by_path(self, repo: str, path: str) -> None:
        self.collection.delete(where={"$and": [{"repo": repo}, {"path": path}]})

    def get_all_paths(self, repo: str) -> dict[str, str]:
        # Chroma doesn't expose distinct queries — fetch metadata only.
        # For very large repos this becomes slow; acceptable up to ~100k chunks.
        result = self.collection.get(where={"repo": repo}, include=["metadatas"])
        path_to_hash: dict[str, str] = {}
        for metadata in result.get("metadatas") or []:
            path = metadata.get("path", "")
            content_hash = metadata.get("content_hash", "")
            # Last write wins — fine, we only need *any* hash to detect change
            path_to_hash[path] = content_hash
        return path_to_hash

    def count(self, repo: str | None = None) -> int:
        if repo is None:
            return self.collection.count()
        return len(self.collection.get(where={"repo": repo}, include=[]).get("ids") or [])

    @staticmethod
    def _hydrate(chunk_id: str, content: str, metadata: dict[str, Any]) -> CodeChunk:
        from app.models import ChunkKind
        return CodeChunk(
            id=chunk_id,
            repo=metadata.get("repo", ""),
            path=metadata.get("path", ""),
            language=metadata.get("language", ""),
            kind=ChunkKind(metadata.get("kind", "function")),
            symbol=metadata.get("symbol", ""),
            parent_symbol=metadata.get("parent_symbol") or None,
            start_line=int(metadata.get("start_line", 1)),
            end_line=int(metadata.get("end_line", 1)),
            content=content,
            content_hash=metadata.get("content_hash", ""),
            imports=[s for s in (metadata.get("imports") or "").split(",") if s],
            calls=[s for s in (metadata.get("calls") or "").split(",") if s],
            git_commit=metadata.get("git_commit") or None,
            indexed_at=int(metadata.get("indexed_at", 0)),
        )