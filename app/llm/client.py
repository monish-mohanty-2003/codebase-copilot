"""Thin wrapper around the Ollama async client.

We expose a single `chat()` method that returns either a final assistant
message or a list of tool calls. Hyperparameters are passed in `options`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ollama import AsyncClient

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[dict[str, Any]]
    raw: dict[str, Any]

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class OllamaLLM:
    def __init__(
        self,
        host: str,
        model: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
        num_ctx: int = 8192,
    ):
        self.client = AsyncClient(host=host)
        self.model = model
        self.options = {
            "temperature": temperature,
            "top_p": top_p,
            "num_ctx": num_ctx,
        }

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "options": self.options,
        }
        if tools:
            kwargs["tools"] = tools

        response = await self.client.chat(**kwargs)

        message = response.get("message", {})
        content = message.get("content", "") or ""
        tool_calls = message.get("tool_calls", []) or []
        return LLMResponse(content=content, tool_calls=tool_calls, raw=response)