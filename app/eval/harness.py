"""Evaluation harness.

Compares (model × prompt × temperature) configurations on a YAML/JSON
ground-truth set. Metrics:

  - groundedness:   does the answer cite a real path that exists?
  - coverage:       does the answer reference any of the gold-standard files?
  - tool_use_rate:  did the agent actually call tools (vs. hallucinate)?
  - latency_p50/p95
  - cost_proxy:     total tokens in agent trace

Output: a JSON report and a Markdown leaderboard.
"""
from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.agent.loop import Agent
from app.config import get_settings
from app.indexing.chunker import Chunker
from app.indexing.embedder import OllamaEmbedder
from app.indexing.parser import TreeSitterParser
from app.indexing.pipeline import IndexingPipeline
from app.indexing.vector_store import ChromaStore
from app.llm.client import OllamaLLM
from app.retrieval.retriever import HybridRetriever
from app.tools import build_default_registry

logger = logging.getLogger(__name__)


@dataclass
class EvalCase:
    """One question with optional ground-truth references."""
    id: str
    query: str
    expected_paths: list[str] = field(default_factory=list)
    must_call_tool: str | None = None  # e.g. 'find_dead_code'


@dataclass
class EvalResult:
    case_id: str
    config: dict[str, Any]
    answer: str
    cited_paths: list[str]
    grounded: bool
    coverage_hit: bool
    tool_calls_made: list[str]
    elapsed_seconds: float


def parse_cited_paths(answer: str, repo_root: Path) -> list[str]:
    """Extract `path/to/file.ext[:line]` references and verify existence."""
    import re
    pattern = re.compile(r"[\w./\-]+\.(?:py|js|ts|tsx|jsx|go|rs|java)(?::\d+(?:-\d+)?)?")
    found = set()
    for match in pattern.findall(answer):
        path = match.split(":")[0]
        if (repo_root / path).is_file():
            found.add(path)
    return sorted(found)


async def run_one(agent: Agent, case: EvalCase, repo_root: Path, config: dict[str, Any]) -> EvalResult:
    start = time.time()
    trace = await agent.run(case.query)
    elapsed = time.time() - start

    cited = parse_cited_paths(trace.answer, repo_root)
    coverage_hit = any(
        any(p in c or c in p for p in case.expected_paths) for c in cited
    ) if case.expected_paths else True

    tool_calls = [s.tool_call.name for s in trace.steps if s.tool_call]
    tool_check = case.must_call_tool is None or case.must_call_tool in tool_calls

    return EvalResult(
        case_id=case.id,
        config=config,
        answer=trace.answer,
        cited_paths=cited,
        grounded=bool(cited),
        coverage_hit=coverage_hit and tool_check,
        tool_calls_made=tool_calls,
        elapsed_seconds=elapsed,
    )


async def run_eval(
    repo_path: Path,
    cases_path: Path,
    configs: list[dict[str, Any]],
    out_path: Path,
) -> None:
    settings = get_settings()
    cases = [EvalCase(**c) for c in json.loads(cases_path.read_text())]

    # Build shared infra once
    parser = TreeSitterParser()
    chunker = Chunker(parser, settings.max_chunk_tokens, settings.chunk_overlap_tokens)
    embedder = OllamaEmbedder(settings.ollama_host, settings.embed_model, settings.embed_batch_size)
    store = ChromaStore(settings.chroma_dir)
    retriever = HybridRetriever(store, embedder, settings.bm25_weight)
    pipeline = IndexingPipeline(settings, parser, chunker, embedder, store)
    repo_name = repo_path.name
    await pipeline.index_repo(repo_path, repo_name=repo_name)

    tools = build_default_registry(repo_path, repo_name, retriever, store)

    all_results: list[EvalResult] = []
    for cfg in configs:
        logger.info("Running config: %s", cfg)
        llm = OllamaLLM(
            host=settings.ollama_host,
            model=cfg["model"],
            temperature=cfg.get("temperature", 0.2),
            top_p=cfg.get("top_p", 0.9),
            num_ctx=cfg.get("num_ctx", 8192),
        )
        agent = Agent(llm=llm, tools=tools, prompt_variant=cfg.get("prompt", "strict"))

        for case in cases:
            try:
                result = await run_one(agent, case, repo_path, cfg)
            except Exception as e:  # noqa: BLE001 — never let a bad case kill the run
                logger.exception("Case %s failed", case.id)
                result = EvalResult(
                    case_id=case.id, config=cfg, answer=f"ERROR: {e}",
                    cited_paths=[], grounded=False, coverage_hit=False,
                    tool_calls_made=[], elapsed_seconds=0.0,
                )
            all_results.append(result)

    summary = summarize(all_results)
    out_path.write_text(json.dumps({
        "results": [r.__dict__ for r in all_results],
        "summary": summary,
    }, indent=2, default=str))
    print_leaderboard(summary)


def summarize(results: list[EvalResult]) -> list[dict[str, Any]]:
    by_config: dict[str, list[EvalResult]] = {}
    for r in results:
        key = json.dumps(r.config, sort_keys=True)
        by_config.setdefault(key, []).append(r)

    summary: list[dict[str, Any]] = []
    for key, group in by_config.items():
        latencies = [r.elapsed_seconds for r in group]
        summary.append({
            "config": json.loads(key),
            "n": len(group),
            "groundedness": round(sum(r.grounded for r in group) / len(group), 3),
            "coverage": round(sum(r.coverage_hit for r in group) / len(group), 3),
            "tool_use_rate": round(
                sum(bool(r.tool_calls_made) for r in group) / len(group), 3,
            ),
            "p50_latency": round(statistics.median(latencies), 2),
            "p95_latency": round(
                statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies), 2,
            ),
        })
    summary.sort(key=lambda s: (s["coverage"], s["groundedness"]), reverse=True)
    return summary


def print_leaderboard(summary: list[dict[str, Any]]) -> None:
    print("\n=== Evaluation Leaderboard ===")
    print(f"{'Model':<30} {'Prompt':<10} {'Temp':<6} {'Cov':<6} {'Grnd':<6} {'Tools':<6} {'p50s':<6}")
    for s in summary:
        c = s["config"]
        print(f"{c['model']:<30} {c.get('prompt','strict'):<10} "
              f"{c.get('temperature',0.2):<6} {s['coverage']:<6} {s['groundedness']:<6} "
              f"{s['tool_use_rate']:<6} {s['p50_latency']:<6}")


# Default config grid — used when a user runs `copilot eval` without a custom file.
DEFAULT_CONFIGS: list[dict[str, Any]] = [
    {"model": "qwen2.5-coder:7b",   "prompt": "strict",  "temperature": 0.2},
    {"model": "qwen2.5-coder:7b",   "prompt": "react",   "temperature": 0.2},
    {"model": "qwen2.5-coder:7b",   "prompt": "minimal", "temperature": 0.2},
    {"model": "qwen2.5-coder:7b",   "prompt": "strict",  "temperature": 0.0},
    {"model": "qwen2.5-coder:7b",   "prompt": "strict",  "temperature": 0.5},
    {"model": "llama3.1:8b",        "prompt": "strict",  "temperature": 0.2},
    {"model": "phi3:mini",          "prompt": "strict",  "temperature": 0.2},
]


def _main() -> None:  # pragma: no cover
    import sys
    if len(sys.argv) < 3:
        print("usage: python -m app.eval.harness <repo_path> <cases.json> [out.json]")
        return
    repo = Path(sys.argv[1])
    cases = Path(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("eval_report.json")
    asyncio.run(run_eval(repo, cases, DEFAULT_CONFIGS, out))


if __name__ == "__main__":
    _main()