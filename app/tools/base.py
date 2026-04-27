"""Tool contract & registry.

Tools are the only way the agent affects or inspects the world. Each tool:
  - Has a stable name and JSON schema (used for Ollama function calling).
  - Has a single async `run(**kwargs) -> Any` entry point.
  - Validates its own inputs and reports errors as a clean string.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from app.models import ToolCall, ToolResult

logger = logging.getLogger(__name__)


class Tool(ABC):
    """Base tool. Subclasses set `name`, `description`, and `parameters`."""

    name: str
    description: str
    # JSON Schema for the parameters object — exactly what Ollama expects.
    parameters: dict[str, Any]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    async def run(self, **kwargs: Any) -> Any: ...


class ToolRegistry:
    """Holds tools, dispatches calls, returns ToolResult."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    async def dispatch(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(name=call.name, error=f"Unknown tool: {call.name}")
        try:
            # Some tools are sync; allow either.
            output = tool.run(**call.arguments)
            if asyncio.iscoroutine(output):
                output = await output
            return ToolResult(name=call.name, output=output)
        except TypeError as e:
            return ToolResult(name=call.name, error=f"Invalid arguments: {e}")
        except Exception as e:  # noqa: BLE001 — surface to agent; never crash
            logger.exception("Tool %s failed", call.name)
            return ToolResult(name=call.name, error=f"{type(e).__name__}: {e}")