"""FastAPI routes."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    SearchHitPayload,
    SearchRequest,
    SearchResponse,
    ToolCallRequest,
    ToolCallResponse,
)
from app.api.state import AppState
from app.models import ToolCall

logger = logging.getLogger(__name__)
router = APIRouter()


def _state(request: Request) -> AppState:
    return request.app.state.app_state


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    state = _state(request)
    return HealthResponse(
        status="ok",
        repo_attached=state.repo_name,
        chunk_count=state.store.count(state.repo_name),
        model=state.settings.chat_model,
    )


@router.post("/index", response_model=IndexResponse)
async def index_repo(req: IndexRequest, request: Request) -> IndexResponse:
    state = _state(request)
    repo_path = Path(req.repo_path).expanduser().resolve()
    if not repo_path.is_dir():
        raise HTTPException(400, f"Not a directory: {repo_path}")

    repo_name = req.repo_name or repo_path.name
    stats = await state.pipeline.index_repo(repo_path, repo_name=repo_name, full=req.full)
    state.attach_repo(repo_path, repo_name)
    return IndexResponse(
        repo=repo_name,
        files_scanned=stats.files_scanned,
        files_indexed=stats.files_indexed,
        files_skipped_unchanged=stats.files_skipped_unchanged,
        chunks_embedded=stats.chunks_embedded,
        errors=stats.errors[:20],
    )


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, request: Request) -> SearchResponse:
    state = _state(request)
    if state.repo_name is None:
        raise HTTPException(409, "No repo attached. POST /index first.")
    hits = await state.retriever.search(
        query=req.query, k=req.k, repo=state.repo_name,
        language=req.language, path_prefix=req.path_prefix,
    )
    if req.path_prefix:
        hits = [h for h in hits if h.chunk.path.startswith(req.path_prefix)]
    return SearchResponse(hits=[
        SearchHitPayload(
            path=h.chunk.path,
            symbol=h.chunk.symbol,
            kind=h.chunk.kind.value,
            language=h.chunk.language,
            start_line=h.chunk.start_line,
            end_line=h.chunk.end_line,
            preview=h.chunk.content[:600],
            score=h.score,
        ) for h in hits
    ])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    state = _state(request)
    if state.agent is None:
        raise HTTPException(409, "No repo attached. POST /index first.")

    if req.stream:
        return _stream_chat(state, req.query)

    trace = await state.agent.run(req.query)
    return ChatResponse(
        answer=trace.answer,
        steps=[s.model_dump() for s in trace.steps],
        elapsed_seconds=trace.elapsed_seconds,
        model=trace.model,
    )


def _stream_chat(state: AppState, query: str) -> EventSourceResponse:
    """SSE streaming variant — emits each agent step as it happens."""
    async def event_gen():
        assert state.agent is not None
        async for step in state.agent.iter_steps(query):
            yield {
                "event": "step",
                "data": json.dumps(step.model_dump(), default=str),
            }
        yield {"event": "done", "data": "{}"}
    return EventSourceResponse(event_gen())


@router.post("/tools/call", response_model=ToolCallResponse)
async def call_tool(req: ToolCallRequest, request: Request) -> ToolCallResponse:
    """Direct tool invocation. Useful for the VS Code extension's quick actions."""
    state = _state(request)
    if state.tools is None:
        raise HTTPException(409, "No repo attached.")
    result = await state.tools.dispatch(ToolCall(name=req.name, arguments=req.arguments))
    return ToolCallResponse(name=result.name, output=result.output, error=result.error)


@router.get("/tools")
async def list_tools(request: Request) -> dict:
    state = _state(request)
    if state.tools is None:
        return {"tools": []}
    return {"tools": state.tools.schemas()}