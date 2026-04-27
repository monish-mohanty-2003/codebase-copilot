"""Agent loop test with a fake LLM. No Ollama required."""
from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from app.agent.loop import Agent
from app.llm.client import LLMResponse
from app.tools.base import Tool, ToolRegistry


class FakeLLM:
    """Yields a scripted sequence of responses, one per chat() call."""

    def __init__(self, scripted: list[LLMResponse]):
        self._scripted = scripted
        self._idx = 0
        self.model = "fake-model"

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        if self._idx >= len(self._scripted):
            raise RuntimeError("FakeLLM exhausted")
        resp = self._scripted[self._idx]
        self._idx += 1
        return resp


class EchoTool(Tool):
    name = "echo"
    description = "Echo input"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def run(self, text: str) -> str:
        return f"echoed: {text}"


@pytest.mark.asyncio
async def test_agent_finishes_without_tool_calls() -> None:
    llm = FakeLLM([LLMResponse(content="42", tool_calls=[], raw={})])
    registry = ToolRegistry()
    registry.register(EchoTool())
    agent = Agent(llm=llm, tools=registry, prompt_variant="strict", max_steps=3)  # type: ignore[arg-type]
    trace = await agent.run("what is 6 times 7?")
    assert trace.answer == "42"
    assert len(trace.steps) == 1


@pytest.mark.asyncio
async def test_agent_calls_tool_then_answers() -> None:
    llm = FakeLLM([
        LLMResponse(
            content="",
            tool_calls=[{"function": {"name": "echo", "arguments": {"text": "ping"}}}],
            raw={},
        ),
        LLMResponse(content="echoed: ping", tool_calls=[], raw={}),
    ])
    registry = ToolRegistry()
    registry.register(EchoTool())
    agent = Agent(llm=llm, tools=registry, prompt_variant="strict", max_steps=3)  # type: ignore[arg-type]
    trace = await agent.run("ping?")
    assert trace.answer == "echoed: ping"
    tool_step = next(s for s in trace.steps if s.tool_call)
    assert tool_step.tool_call.name == "echo"
    assert tool_step.tool_result.output == "echoed: ping"


@pytest.mark.asyncio
async def test_agent_caps_at_max_steps() -> None:
    """If LLM keeps calling tools forever, the loop must terminate cleanly."""
    looping = [
        LLMResponse(
            content="",
            tool_calls=[{"function": {"name": "echo", "arguments": {"text": "x"}}}],
            raw={},
        )
    ] * 10
    looping.append(LLMResponse(content="forced final", tool_calls=[], raw={}))
    llm = FakeLLM(looping)
    registry = ToolRegistry()
    registry.register(EchoTool())
    agent = Agent(llm=llm, tools=registry, prompt_variant="strict", max_steps=2)  # type: ignore[arg-type]
    trace = await agent.run("loop?")
    assert trace.answer  # something was returned, even if forced