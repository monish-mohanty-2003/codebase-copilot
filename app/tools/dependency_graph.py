"""get_dependency_graph: build a coarse module/symbol dependency graph.

Strategy: re-walk the indexed chunks, extract identifier references, link them
to definitions by symbol name. Imperfect (no scope analysis) but useful enough
for "what calls X" and dead-code detection.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import networkx as nx

from app.indexing.vector_store import VectorStore
from app.models import CodeChunk
from app.tools.base import Tool

_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
# Words too common to count as references — would create false edges.
_STOPWORDS = {
    "if", "else", "for", "while", "return", "import", "from", "def", "class",
    "True", "False", "None", "self", "this", "let", "const", "var", "function",
    "async", "await", "try", "except", "raise", "throw", "in", "is", "not",
    "and", "or", "as", "with", "yield", "pass", "break", "continue", "lambda",
    "fn", "mut", "pub", "use", "match", "type", "impl", "trait", "struct",
}


class DependencyGraphTool(Tool):
    name = "get_dependency_graph"
    description = (
        "Build a symbol-level dependency graph for the repo and return summary "
        "metrics: top-referenced symbols, candidate dead code (unreferenced), "
        "and optional callers/callees of a specific symbol."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Optional: get callers and callees of this symbol.",
            },
            "max_dead_code": {
                "type": "integer",
                "description": "Max dead-code candidates to return (default 20).",
                "default": 20,
            },
        },
    }

    def __init__(self, store: VectorStore, repo: str):
        self.store = store
        self.repo = repo

    def run(self, symbol: str | None = None, max_dead_code: int = 20) -> dict[str, Any]:
        chunks = self._all_chunks()
        graph = self._build_graph(chunks)

        # Top in-degree (most referenced)
        in_deg = sorted(graph.in_degree, key=lambda x: x[1], reverse=True)[:10]

        # Dead code candidates: defs with zero incoming edges,
        # excluding obvious entry points (main, __init__, test_*, lib roots).
        dead: list[dict[str, Any]] = []
        for node, in_d in graph.in_degree():
            if in_d > 0:
                continue
            attrs = graph.nodes[node]
            if not attrs.get("is_definition"):
                continue
            short = attrs["symbol"].split(".")[-1]
            if short.startswith("_") or short in {"main", "lambda_handler"}:
                continue
            if "test_" in attrs["path"] or attrs["path"].startswith("tests/"):
                continue
            dead.append({
                "symbol": attrs["symbol"],
                "path": attrs["path"],
                "start_line": attrs["start_line"],
                "kind": attrs["kind"],
            })
            if len(dead) >= max_dead_code:
                break

        result: dict[str, Any] = {
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "top_referenced": [
                {"symbol": graph.nodes[n]["symbol"], "in_degree": d}
                for n, d in in_deg
            ],
            "dead_code_candidates": dead,
        }

        if symbol:
            result["symbol_neighborhood"] = self._neighborhood(graph, symbol)
        return result

    def _all_chunks(self) -> list[CodeChunk]:
        # Pull every chunk for the repo. For larger repos, paginate.
        store = self.store
        # We rely on duck typing — ChromaStore exposes the underlying collection.
        if hasattr(store, "collection"):
            data = store.collection.get(  # type: ignore[attr-defined]
                where={"repo": self.repo}, include=["metadatas", "documents"],
            )
            chunks: list[CodeChunk] = []
            for chunk_id, content, metadata in zip(
                data.get("ids") or [], data.get("documents") or [], data.get("metadatas") or [],
                strict=False,
            ):
                chunks.append(store._hydrate(chunk_id, content, metadata))  # type: ignore[attr-defined]
            return chunks
        return []

    @staticmethod
    def _build_graph(chunks: list[CodeChunk]) -> nx.DiGraph:
        graph = nx.DiGraph()
        defs_by_name: dict[str, list[CodeChunk]] = defaultdict(list)

        for c in chunks:
            short = c.symbol.split(".")[-1]
            graph.add_node(
                c.id,
                symbol=c.symbol,
                short_name=short,
                path=c.path,
                start_line=c.start_line,
                kind=c.kind.value,
                is_definition=c.kind.value in {"function", "method", "class"},
            )
            if c.kind.value in {"function", "method", "class"}:
                defs_by_name[short].append(c)

        # Add edges: chunk content references a known short_name → edge.
        for c in chunks:
            tokens = {t for t in _IDENTIFIER_RE.findall(c.content) if t not in _STOPWORDS}
            own_short = c.symbol.split(".")[-1]
            for tok in tokens:
                if tok == own_short:
                    continue
                for target in defs_by_name.get(tok, []):
                    if target.id == c.id:
                        continue
                    graph.add_edge(c.id, target.id)
        return graph

    @staticmethod
    def _neighborhood(graph: nx.DiGraph, symbol: str) -> dict[str, Any]:
        matches = [n for n, attrs in graph.nodes(data=True) if attrs["symbol"].endswith(symbol)]
        if not matches:
            return {"error": f"symbol {symbol!r} not found"}
        node = matches[0]
        callers = [graph.nodes[p]["symbol"] for p in graph.predecessors(node)][:20]
        callees = [graph.nodes[s]["symbol"] for s in graph.successors(node)][:20]
        return {
            "symbol": graph.nodes[node]["symbol"],
            "path": graph.nodes[node]["path"],
            "callers": callers,
            "callees": callees,
        }