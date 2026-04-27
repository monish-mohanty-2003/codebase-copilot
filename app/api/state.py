"""Application state holder.

We assemble all heavy singletons (parser, embedder, store, retriever, agent,
tool registry) once on startup and inject through FastAPI's app.state.
This keeps endpoints thin and tests trivially mockable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.agent.loop import Agent
from app.config import Settings
from app.indexing.chunker import Chunker
from app.indexing.embedder import OllamaEmbedder
from app.indexing.parser import TreeSitterParser
from app.indexing.pipeline import IndexingPipeline
from app.indexing.vector_store import ChromaStore, VectorStore
from app.llm.client import OllamaLLM
from app.retrieval.retriever import HybridRetriever
from app.tools import build_default_registry
from app.tools.base import ToolRegistry


@dataclass
class AppState:
    settings: Settings
    parser: TreeSitterParser
    chunker: Chunker
    embedder: OllamaEmbedder
    store: VectorStore
    retriever: HybridRetriever
    pipeline: IndexingPipeline
    llm: OllamaLLM
    # Per-repo lazy-built items
    repo_root: Path | None = None
    repo_name: str | None = None
    tools: ToolRegistry | None = None
    agent: Agent | None = None

    @classmethod
    def build(cls, settings: Settings) -> AppState:
        parser = TreeSitterParser()
        chunker = Chunker(parser, settings.max_chunk_tokens, settings.chunk_overlap_tokens)
        embedder = OllamaEmbedder(settings.ollama_host, settings.embed_model, settings.embed_batch_size)
        store = ChromaStore(settings.chroma_dir)
        retriever = HybridRetriever(store, embedder, settings.bm25_weight)
        pipeline = IndexingPipeline(settings, parser, chunker, embedder, store)
        llm = OllamaLLM(
            host=settings.ollama_host,
            model=settings.chat_model,
            temperature=settings.chat_temperature,
            top_p=settings.chat_top_p,
            num_ctx=settings.chat_num_ctx,
        )
        return cls(
            settings=settings, parser=parser, chunker=chunker, embedder=embedder,
            store=store, retriever=retriever, pipeline=pipeline, llm=llm,
        )

    def attach_repo(self, repo_root: Path, repo_name: str) -> None:
        """Wire tools and agent for a specific repo."""
        self.repo_root = repo_root.resolve()
        self.repo_name = repo_name
        self.tools = build_default_registry(
            repo_root=self.repo_root,
            repo_name=repo_name,
            retriever=self.retriever,
            store=self.store,
        )
        self.agent = Agent(
            llm=self.llm,
            tools=self.tools,
            prompt_variant=self.settings.system_prompt_variant,
            max_steps=self.settings.max_agent_steps,
        )