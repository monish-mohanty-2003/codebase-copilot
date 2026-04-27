"""Centralized configuration. Override via env vars or .env file."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    Single source of truth. Inject anywhere via `get_settings()`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="COPILOT_",
        case_sensitive=False,
        extra="ignore",
    )

    # --- paths ---
    data_dir: Path = Field(default=Path(".copilot"))
    chroma_dir: Path = Field(default=Path(".copilot/chroma"))
    cache_dir: Path = Field(default=Path(".copilot/cache"))

    # --- ollama ---
    ollama_host: str = "http://localhost:11434"
    chat_model: str = "qwen2.5-coder:7b"
    embed_model: str = "nomic-embed-text"
    chat_temperature: float = 0.2
    chat_top_p: float = 0.9
    chat_num_ctx: int = 8192

    # --- indexing ---
    max_chunk_tokens: int = 1500
    chunk_overlap_tokens: int = 100
    embed_batch_size: int = 64
    supported_languages: tuple[str, ...] = (
        "python", "javascript", "typescript", "tsx", "go", "rust", "java",
    )

    # --- retrieval ---
    retrieval_k: int = 16
    rerank_top_n: int = 6
    bm25_weight: float = 0.4  # vs vector similarity in fusion

    # --- agent ---
    max_agent_steps: int = 6
    system_prompt_variant: Literal["minimal", "react", "strict"] = "strict"

    # --- api ---
    host: str = "127.0.0.1"
    port: int = 8000

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.chroma_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Singleton accessor. Lazy so tests can override env first."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings