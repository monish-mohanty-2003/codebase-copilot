"""Embedding via Ollama. Batched, retry-on-error."""
from __future__ import annotations

import asyncio
import logging

import httpx
from ollama import AsyncClient

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    """Wraps Ollama's embedding endpoint.

    We use the async client because indexing thousands of chunks is the
    main bottleneck — overlapping HTTP calls with disk reads helps a lot.
    """

    def __init__(self, host: str, model: str, batch_size: int = 64):
        self.client = AsyncClient(host=host)
        self.model = model
        self.batch_size = batch_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text. Order preserved."""
        if not texts:
            return []

        results: list[list[float]] = [[] for _ in texts]

        # Sequential batches — Ollama serializes per model anyway,
        # so parallel requests don't help and risk OOM on small GPUs.
        for batch_start in range(0, len(texts), self.batch_size):
            batch = texts[batch_start: batch_start + self.batch_size]
            embeddings = await self._embed_batch(batch)
            for i, emb in enumerate(embeddings):
                results[batch_start + i] = emb
        return results

    async def _embed_batch(self, batch: list[str], retries: int = 3) -> list[list[float]]:
        """Embed a single batch with simple exponential backoff."""
        for attempt in range(retries):
            try:
                # ollama-python's embed() accepts a list of inputs in 0.4+
                response = await self.client.embed(model=self.model, input=batch)
                return list(response["embeddings"])
            except (httpx.HTTPError, KeyError) as e:
                wait = 2 ** attempt
                logger.warning(
                    "Embed batch failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, retries, e, wait,
                )
                await asyncio.sleep(wait)
        raise RuntimeError(f"Failed to embed batch after {retries} attempts")