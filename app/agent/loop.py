"""ReAct-style agent loop driving tool calls via Ollama function calling.

Loop:
    while not done and steps < max_steps:
        response = llm.chat(messages, tools=schemas)
        if response has tool_calls:
            run each tool, append results as 'tool' role messages
        else:
            return response.content
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from app.llm.client import OllamaLLM
from app.llm.prompts import get_prompt
from app.models import AgentStep, AgentTrace, ToolCall
from app.tools.base import ToolRegistry

logger = logging.getLogger(__name__)


class Agent:
    """Code copilot agent."""

    def __init__(
        self,
        llm: OllamaLLM,
        tools: ToolRegistry,
        prompt_variant: str = "strict",
        max_steps: int = 6,
    ):
        self.llm = llm
        self.tools = tools
        self.prompt_variant = prompt_variant
        self.max_steps = max_steps

    async def run(self, query: str) -> AgentTrace:
        """Non-streaming run. Returns full trace."""
        trace = AgentTrace(query=query, model=self.llm.model)
        start = time.time()
        async for step in self.iter_steps(query):
            trace.steps.append(step)
            if step.final_answer is not None:
                trace.answer = step.final_answer
                break
        trace.elapsed_seconds = round(time.time() - start, 2)
        return trace

    async def iter_steps(self, query: str) -> AsyncIterator[AgentStep]:
        """Stream individual reasoning steps. Useful for live UI updates."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": get_prompt(self.prompt_variant)},
            {"role": "user", "content": query},
        ]
        tool_schemas = self.tools.schemas()

        for step_idx in range(1, self.max_steps + 1):
            response = await self.llm.chat(messages, tools=tool_schemas)

            # Append assistant message verbatim so subsequent calls have history
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": response.content}
            if response.tool_calls:
                assistant_msg["tool_calls"] = response.tool_calls
            messages.append(assistant_msg)

            if not response.has_tool_calls:
                # Final answer
                yield AgentStep(
                    step=step_idx,
                    thought=None,
                    final_answer=response.content,
                )
                return

            # Execute each tool call (sequential — keeps results grounded for next call)
            for raw_call in response.tool_calls:
                fn = raw_call.get("function") or {}
                name = fn.get("name", "")
                args = fn.get("arguments") or {}
                # Some Ollama versions stringify arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                tool_call = ToolCall(name=name, arguments=args)
                result = await self.tools.dispatch(tool_call)

                # Tool result back to model — keep payload compact
                payload = (
                    {"error": result.error}
                    if result.error else self._truncate(result.output)
                )
                messages.append({
                    "role": "tool",
                    "name": name,
                    "content": json.dumps(payload, default=str)[:8000],
                })

                yield AgentStep(
                    step=step_idx,
                    thought=response.content or None,
                    tool_call=tool_call,
                    tool_result=result,
                )

        # Hit max_steps without a final answer — force one
        messages.append({
            "role": "user",
            "content": "Please provide your final answer now based on what you have gathered.",
        })
        final = await self.llm.chat(messages, tools=None)
        yield AgentStep(step=self.max_steps + 1, final_answer=final.content)

    @staticmethod
    def _truncate(value: Any, limit: int = 6000) -> Any:
        """Bound size of tool results before sending back to LLM."""
        if isinstance(value, str):
            return value[:limit]
        if isinstance(value, list):
            return value[:30]  # cap list length
        return value