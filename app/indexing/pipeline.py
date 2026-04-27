"""Indexing pipeline.

Scans a repo, parses each supported file, chunks definitions, embeds in batches,
and stores in the vector DB. Supports incremental reindex: only changed files
(by content hash) are re-embedded.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings
from app.indexing.chunker import Chunker
from app.indexing.embedder import OllamaEmbedder
from app.indexing.parser import EXTENSION_TO_LANGUAGE, TreeSitterParser
from app.indexing.vector_store import VectorStore
from app.models import CodeChunk

logger = logging.getLogger(__name__)

# Skip these directories outright — never useful, often huge.
IGNORED_DIRS: set[str] = {
    ".git", "node_modules", ".venv", "venv", "env", "__pycache__",
    "dist", "build", "target", ".next", ".nuxt", ".cache", ".copilot",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
}

# Hard cap to avoid embedding huge generated files.
MAX_FILE_BYTES = 500_000


@dataclass
class IndexStats:
    files_scanned: int = 0
    files_indexed: int = 0
    files_skipped_unchanged: int = 0
    chunks_created: int = 0
    chunks_embedded: int = 0
    errors: list[str] = field(default_factory=list)


class IndexingPipeline:
    def __init__(
        self,
        settings: Settings,
        parser: TreeSitterParser,
        chunker: Chunker,
        embedder: OllamaEmbedder,
        store: VectorStore,
    ):
        self.settings = settings
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.store = store

    async def index_repo(self, repo_path: Path, repo_name: str | None = None, *, full: bool = False) -> IndexStats:
        """Index (or re-index) a repo. Set `full=True` to force re-embed everything."""
        repo_path = repo_path.resolve()
        if not repo_path.is_dir():
            raise ValueError(f"Not a directory: {repo_path}")

        repo = repo_name or repo_path.name
        git_commit = self._read_git_commit(repo_path)
        stats = IndexStats()

        existing = {} if full else self.store.get_all_paths(repo)

        # Group chunks by file so we can delete-old-then-insert-new atomically.
        per_file_chunks: list[tuple[str, list[CodeChunk]]] = []

        for path in self._walk(repo_path):
            stats.files_scanned += 1
            try:
                rel_path = str(path.relative_to(repo_path))
                file_hash = self._file_hash(path)
                # Skip unchanged files: if any chunk we already have for this path
                # has a content_hash equal to this file's hash, we can skip.
                # (Simple heuristic — collisions would just cause a re-embed.)
                if not full and existing.get(rel_path) == file_hash:
                    stats.files_skipped_unchanged += 1
                    continue

                chunks = self.chunker.chunk_file(repo, repo_path, path, git_commit=git_commit)
                if not chunks:
                    continue
                # Tag every chunk's content_hash off the file hash too,
                # so the skip check above works.
                for c in chunks:
                    c.content_hash = file_hash
                per_file_chunks.append((rel_path, chunks))
                stats.chunks_created += len(chunks)
            except Exception as e:  # noqa: BLE001 — record and continue
                stats.errors.append(f"{path}: {e}")
                logger.exception("Error indexing %s", path)

        # Embed in batches across files (better GPU utilization)
        all_chunks = [c for _, chunks in per_file_chunks for c in chunks]
        if not all_chunks:
            logger.info("No chunks to embed; index is up to date.")
            return stats

        logger.info("Embedding %d chunks…", len(all_chunks))
        embeddings = await self.embedder.embed([c.embed_text() for c in all_chunks])

        # Delete old chunks for changed files, then upsert new ones
        for rel_path, _ in per_file_chunks:
            self.store.delete_by_path(repo, rel_path)

        self.store.add(all_chunks, embeddings)
        stats.files_indexed = len(per_file_chunks)
        stats.chunks_embedded = len(all_chunks)
        logger.info(
            "Indexed repo=%s files=%d chunks=%d skipped_unchanged=%d",
            repo, stats.files_indexed, stats.chunks_embedded, stats.files_skipped_unchanged,
        )
        return stats

    def _walk(self, root: Path) -> Iterator[Path]:
        """Yield all source files under root, respecting ignore list & size cap."""
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            # Skip if any parent dir is in IGNORED_DIRS
            if any(part in IGNORED_DIRS for part in path.relative_to(root).parts):
                continue
            if path.suffix.lower() not in EXTENSION_TO_LANGUAGE:
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield path

    @staticmethod
    def _file_hash(path: Path) -> str:
        h = hashlib.sha1()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    @staticmethod
    def _read_git_commit(repo_path: Path) -> str | None:
        head = repo_path / ".git" / "HEAD"
        if not head.exists():
            return None
        try:
            text = head.read_text().strip()
            if text.startswith("ref:"):
                ref_path = repo_path / ".git" / text.split(" ", 1)[1]
                if ref_path.exists():
                    return ref_path.read_text().strip()[:12]
            return text[:12]
        except OSError:
            return None


async def _main() -> None:  # pragma: no cover — manual smoke test
    """Smoke test: `python -m app.indexing.pipeline /path/to/repo`."""
    import sys
    from app.config import get_settings

    if len(sys.argv) < 2:
        print("usage: python -m app.indexing.pipeline <repo_path>")
        return

    settings = get_settings()
    parser = TreeSitterParser()
    chunker = Chunker(parser, settings.max_chunk_tokens, settings.chunk_overlap_tokens)
    embedder = OllamaEmbedder(settings.ollama_host, settings.embed_model, settings.embed_batch_size)
    from app.indexing.vector_store import ChromaStore
    store = ChromaStore(settings.chroma_dir)
    pipeline = IndexingPipeline(settings, parser, chunker, embedder, store)
    stats = await pipeline.index_repo(Path(sys.argv[1]))
    print(stats)


if __name__ == "__main__":
    asyncio.run(_main())