"""Higher-level tools composed from primitives."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.tools.base import Tool
from app.tools.dependency_graph import DependencyGraphTool
from app.tools.open_file import OpenFileTool
from app.tools.search_code import SearchCodeTool


class FindDeadCodeTool(Tool):
    name = "find_dead_code"
    description = (
        "Heuristically find unreferenced functions/classes in the repo. "
        "Combines dependency graph analysis with usage search."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "default": 15},
        },
    }

    def __init__(self, dep: DependencyGraphTool):
        self.dep = dep

    def run(self, limit: int = 15) -> dict[str, Any]:
        result = self.dep.run(max_dead_code=limit)
        # Re-shape for clarity
        return {
            "candidates": result["dead_code_candidates"],
            "graph_size": {
                "nodes": result["node_count"],
                "edges": result["edge_count"],
            },
            "note": (
                "Heuristic: definitions with zero static references in the graph. "
                "May produce false positives for dynamic dispatch, plugins, or "
                "framework-registered handlers. Verify before deletion."
            ),
        }


class SuggestTestsTool(Tool):
    name = "suggest_tests"
    description = (
        "Given a symbol or file, retrieve its source and adjacent context, "
        "then suggest unit-test cases covering happy path, edge cases, and "
        "error paths. Returns the gathered context — the LLM uses it to write tests."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Function or class name to test.",
            },
            "path": {
                "type": "string",
                "description": "Optional path hint to disambiguate.",
            },
        },
        "required": ["symbol"],
    }

    def __init__(self, search: SearchCodeTool, open_file: OpenFileTool):
        self.search = search
        self.open_file = open_file

    async def run(self, symbol: str, path: str | None = None) -> dict[str, Any]:
        # Find the symbol via search
        hits = await self.search.run(query=symbol, k=5)
        candidates = [h for h in hits if h["symbol"].endswith(symbol)] or hits
        if not candidates:
            return {"error": f"Symbol {symbol!r} not found"}
        target = next((c for c in candidates if path and c["path"] == path), candidates[0])

        full = self.open_file.run(
            path=target["path"],
            start_line=max(1, target["start_line"] - 5),
            end_line=target["end_line"] + 5,
        )

        return {
            "symbol": target["symbol"],
            "path": target["path"],
            "language": target["language"],
            "source": full["content"],
            "test_seeds": [
                "Happy path: typical valid input → expected output",
                "Edge case: empty/zero/single-element input",
                "Edge case: maximum or boundary input",
                "Error path: invalid type / null / out-of-range",
                "Error path: external dependency failure (mock & assert)",
            ],
        }


class ProposePatchTool(Tool):
    name = "propose_patch"
    description = (
        "Read a file, gather context, and return a suggested unified diff. "
        "Does NOT write to disk — the user reviews and applies the patch."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
            "intent": {
                "type": "string",
                "description": "What to change, in plain English.",
            },
        },
        "required": ["path", "intent"],
    }

    def __init__(self, repo_root: Path, open_file: OpenFileTool):
        self.repo_root = repo_root.resolve()
        self.open_file = open_file

    def run(
        self,
        path: str,
        intent: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        ctx = self.open_file.run(path=path, start_line=start_line, end_line=end_line)
        return {
            "path": path,
            "intent": intent,
            "context": ctx,
            "instructions": (
                "Return a unified diff (--- a/{path} +++ b/{path}) that implements "
                "the intent. Preserve indentation. Don't include unrelated changes."
            ),
        }