"""search_code: the agent's primary tool for finding relevant code."""
from __future__ import annotations

from typing import Any

from app.retrieval.retriever import HybridRetriever
from app.tools.base import Tool


class SearchCodeTool(Tool):
    name = "search_code"
    description = (
        "Semantic + lexical search across the indexed repository. "
        "Use this whenever you need to find code related to a concept, "
        "feature, identifier, or behavior. Returns top-k chunks with "
        "file path and line numbers. Prefer specific multi-word queries."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language or identifier query.",
            },
            "k": {
                "type": "integer",
                "description": "Number of results to return (1-20).",
                "default": 8,
            },
            "language": {
                "type": "string",
                "description": "Optional language filter (python, typescript, go, ...).",
            },
            "path_prefix": {
                "type": "string",
                "description": "Optional path prefix filter, e.g. 'src/auth/'.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, retriever: HybridRetriever, default_repo: str | None = None):
        self.retriever = retriever
        self.default_repo = default_repo

    async def run(
        self,
        query: str,
        k: int = 8,
        language: str | None = None,
        path_prefix: str | None = None,
    ) -> list[dict[str, Any]]:
        k = max(1, min(int(k), 20))
        hits = await self.retriever.search(
            query=query, k=k, repo=self.default_repo,
            language=language, path_prefix=path_prefix,
        )
        # Path-prefix filter applied here (Chroma can't do prefix).
        if path_prefix:
            hits = [h for h in hits if h.chunk.path.startswith(path_prefix)]

        return [
            {
                "path": h.chunk.path,
                "symbol": h.chunk.symbol,
                "kind": h.chunk.kind.value,
                "language": h.chunk.language,
                "start_line": h.chunk.start_line,
                "end_line": h.chunk.end_line,
                # Truncate to keep tokens manageable in the agent loop
                "preview": h.chunk.content[:600],
                "score": round(h.score, 4),
            }
            for h in hits
        ]