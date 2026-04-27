"""Chunking strategy.

- Each definition becomes one chunk if it fits the token budget.
- Oversized functions are split with overlap.
- We also emit a 'module' chunk per file containing imports + module docstring,
  so file-level questions ("what does this file do?") have a target.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import tiktoken

from app.indexing.parser import ParsedDefinition, TreeSitterParser
from app.models import ChunkKind, CodeChunk

# Use cl100k_base — close enough to most code-tokenizer behaviors for budgeting.
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text, disallowed_special=()))


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]


class Chunker:
    """Convert parsed source into CodeChunk objects."""

    def __init__(self, parser: TreeSitterParser, max_tokens: int = 1500, overlap_tokens: int = 100):
        self.parser = parser
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_file(self, repo: str, root: Path, path: Path, git_commit: str | None = None) -> list[CodeChunk]:
        """Return all chunks for a single file. Empty if unsupported language."""
        language, defs, imports = self.parser.parse_file(path)
        if not language:
            return []

        rel_path = str(path.relative_to(root))
        chunks: list[CodeChunk] = []
        now = int(time.time())

        # Module-level chunk: imports + first ~30 lines.
        # Useful for file-level questions even if the file has no top-level defs.
        try:
            head = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        head_lines = head.splitlines()[:30]
        module_text = "\n".join(head_lines)
        if module_text.strip():
            chunks.append(CodeChunk(
                id=_hash(f"{repo}:{rel_path}:module"),
                repo=repo,
                path=rel_path,
                language=language,
                kind=ChunkKind.MODULE,
                symbol=Path(rel_path).stem,
                parent_symbol=None,
                start_line=1,
                end_line=min(len(head_lines), 30),
                content=module_text,
                content_hash=_hash(module_text),
                imports=imports,
                git_commit=git_commit,
                indexed_at=now,
            ))

        for d in defs:
            chunks.extend(
                self._chunks_for_definition(repo, rel_path, language, d, imports, git_commit, now)
            )

        return chunks

    def _chunks_for_definition(
        self,
        repo: str,
        rel_path: str,
        language: str,
        d: ParsedDefinition,
        imports: list[str],
        git_commit: str | None,
        now: int,
    ) -> list[CodeChunk]:
        token_count = count_tokens(d.content)
        kind = self._kind_from_str(d.kind)

        if token_count <= self.max_tokens:
            return [CodeChunk(
                id=_hash(f"{repo}:{rel_path}:{d.symbol}:{d.start_line}"),
                repo=repo,
                path=rel_path,
                language=language,
                kind=kind,
                symbol=d.symbol,
                parent_symbol=d.parent_symbol,
                start_line=d.start_line,
                end_line=d.end_line,
                content=d.content,
                content_hash=_hash(d.content),
                imports=imports,
                git_commit=git_commit,
                indexed_at=now,
            )]

        # Oversized — split by lines with token-aware boundaries
        return self._split_oversized(repo, rel_path, language, d, imports, git_commit, now)

    def _split_oversized(
        self,
        repo: str,
        rel_path: str,
        language: str,
        d: ParsedDefinition,
        imports: list[str],
        git_commit: str | None,
        now: int,
    ) -> list[CodeChunk]:
        """Split a too-big definition into overlapping line windows.

        We carry the signature line into every part so each chunk has context.
        """
        lines = d.content.splitlines()
        signature = lines[0] if lines else ""

        parts: list[CodeChunk] = []
        buffer: list[str] = []
        buffer_tokens = 0
        part_index = 0
        part_start_line = d.start_line

        for offset, line in enumerate(lines):
            line_tokens = count_tokens(line) + 1  # +1 for newline
            if buffer_tokens + line_tokens > self.max_tokens and buffer:
                content = signature + "\n" + "\n".join(buffer) if part_index > 0 else "\n".join(buffer)
                parts.append(CodeChunk(
                    id=_hash(f"{repo}:{rel_path}:{d.symbol}:part{part_index}"),
                    repo=repo,
                    path=rel_path,
                    language=language,
                    kind=ChunkKind.OVERSIZED_PART,
                    symbol=f"{d.symbol}#part{part_index}",
                    parent_symbol=d.symbol,
                    start_line=part_start_line,
                    end_line=d.start_line + offset - 1,
                    content=content,
                    content_hash=_hash(content),
                    imports=imports,
                    git_commit=git_commit,
                    indexed_at=now,
                ))
                # Keep a tail as overlap
                overlap_lines: list[str] = []
                overlap_tokens = 0
                for tail_line in reversed(buffer):
                    t = count_tokens(tail_line) + 1
                    if overlap_tokens + t > self.overlap_tokens:
                        break
                    overlap_lines.insert(0, tail_line)
                    overlap_tokens += t
                buffer = overlap_lines
                buffer_tokens = overlap_tokens
                part_start_line = d.start_line + offset - len(overlap_lines)
                part_index += 1

            buffer.append(line)
            buffer_tokens += line_tokens

        if buffer:
            content = signature + "\n" + "\n".join(buffer) if part_index > 0 else "\n".join(buffer)
            parts.append(CodeChunk(
                id=_hash(f"{repo}:{rel_path}:{d.symbol}:part{part_index}"),
                repo=repo,
                path=rel_path,
                language=language,
                kind=ChunkKind.OVERSIZED_PART,
                symbol=f"{d.symbol}#part{part_index}",
                parent_symbol=d.symbol,
                start_line=part_start_line,
                end_line=d.end_line,
                content=content,
                content_hash=_hash(content),
                imports=imports,
                git_commit=git_commit,
                indexed_at=now,
            ))

        return parts

    @staticmethod
    def _kind_from_str(s: str) -> ChunkKind:
        return {
            "function": ChunkKind.FUNCTION,
            "method": ChunkKind.METHOD,
            "class": ChunkKind.CLASS,
        }.get(s, ChunkKind.FUNCTION)