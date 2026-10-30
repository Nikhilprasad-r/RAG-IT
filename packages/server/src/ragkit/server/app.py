"""HTTP gateway for a ragkit project.

Endpoints:
  GET  /health                  liveness
  GET  /agents                  agents in the project
  POST /agents/{name}/chat      one turn, full reply
  POST /agents/{name}/stream    one turn, reply streamed as server-sent events
  POST /agents/{name}/resume    continue a turn waiting for approval (add /stream for SSE)
  GET  /collections             collections with document and chunk counts
  POST /collections/{name}/search   hybrid search, reranked when configured
"""

import json
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ragkit.agents.runtime import (
    Agent,
    AgentNotFoundError,
    AgentResult,
    AgentRuntime,
    ApprovalDecision,
    ApprovalError,
    ModelFactory,
    PendingAction,
    ThreadWaitingError,
    new_thread_id,
)
from ragkit.core.config import ProjectConfig
from ragkit.core.errors import ConfigError
from ragkit.observe import configure_tracing, instrument_app
from ragkit.retrieval import KnowledgeBase


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str | None = Field(
        default=None, description="Continue a conversation; omit to start a new one."
    )


class PendingActionInfo(BaseModel):
    tool: str
    args: dict[str, Any]
    description: str
    allowed_decisions: list[str]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cost_usd: float | None = Field(
        default=None, description="Set when every model used has a price in ragkit.yaml."
    )
    guardrails: list[dict[str, str]] = Field(
        default_factory=list, description="Guardrails that acted during the turn."
    )


class ChatResponse(BaseModel):
    reply: str
    thread_id: str
    status: Literal["completed", "waiting_for_approval"]
    pending: list[PendingActionInfo] = Field(
        default_factory=list, description="Tool calls waiting for a decision, in order."
    )
    structured: dict[str, Any] | None = Field(
        default=None, description="The agent's output when it has an output_schema."
    )
    usage: Usage

    @classmethod
    def of(cls, result: AgentResult) -> "ChatResponse":
        return cls(
            reply=result.reply,
            thread_id=result.thread_id,
            status=result.status,
            pending=[_pending_info(p) for p in result.pending],
            structured=result.structured,
            usage=Usage(
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                cost_usd=result.usage.cost_usd if result.usage.priced else None,
                guardrails=result.usage.guardrails,
            ),
        )


class ResumeRequest(BaseModel):
    thread_id: str
    decisions: list[ApprovalDecision] = Field(
        min_length=1, description="One decision per pending action, in the same order."
    )


class AgentInfo(BaseModel):
    name: str
    description: str
    pattern: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=100)
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    filters: dict[str, str | int | float | bool] = Field(default_factory=dict)
    rerank: bool = True


class SearchHit(BaseModel):
    text: str
    score: float
    retriever: str
    uri: str
    title: str
    heading_path: list[str]
    page: int | None
    metadata: dict[str, Any]


class CollectionInfo(BaseModel):
    name: str
    description: str
    documents: int
    chunks: int
    embedding_model: str | None


def create_app(
    project: ProjectConfig | None = None, *, model_factory: ModelFactory | None = None
) -> FastAPI:
    project = project or ProjectConfig.load()
    configure_tracing(project)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            knowledge = None
            if project.collections:
                knowledge = await stack.enter_async_context(KnowledgeBase(project))
            app.state.knowledge = knowledge
            app.state.runtime = await stack.enter_async_context(
                AgentRuntime(project, model_factory=model_factory, knowledge=knowledge)
            )
            yield

    app = FastAPI(title=f"ragkit: {project.name}", lifespan=lifespan)
    instrument_app(app, project)

    @app.exception_handler(ApprovalError)
    @app.exception_handler(ThreadWaitingError)
    async def conflict(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    if project.server.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=project.server.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/agents")
    async def list_agents(request: Request) -> list[AgentInfo]:
        return [
            AgentInfo(
                name=agent.name,
                description=agent.manifest.description,
                pattern=agent.manifest.pattern,
            )
            for agent in _runtime(request).agents.values()
        ]

    @app.post("/agents/{name}/chat")
    async def chat(name: str, body: ChatRequest, request: Request) -> ChatResponse:
        return ChatResponse.of(await _agent(request, name).ainvoke(body.message, body.thread_id))

    @app.post("/agents/{name}/stream")
    async def stream(name: str, body: ChatRequest, request: Request) -> StreamingResponse:
        agent = _agent(request, name)
        thread_id = body.thread_id or new_thread_id()
        if await agent.pending(thread_id):  # fail with a status code, not mid-stream
            raise ThreadWaitingError(f"Thread {thread_id} is waiting for approval.")
        return _sse_response(agent, agent.astream(body.message, thread_id), thread_id)

    @app.post("/agents/{name}/resume")
    async def resume(name: str, body: ResumeRequest, request: Request) -> ChatResponse:
        result = await _agent(request, name).resume(body.thread_id, body.decisions)
        return ChatResponse.of(result)

    @app.post("/agents/{name}/resume/stream")
    async def resume_stream(name: str, body: ResumeRequest, request: Request) -> StreamingResponse:
        agent = _agent(request, name)
        tokens = await agent.astream_resume(body.thread_id, body.decisions)
        return _sse_response(agent, tokens, body.thread_id)

    @app.get("/collections")
    async def list_collections(request: Request) -> list[CollectionInfo]:
        knowledge = _knowledge(request)
        stats = {s.collection: s for s in await knowledge.store.stats()}
        return [
            CollectionInfo(
                name=name,
                description=config.description,
                documents=stats[name].documents if name in stats else 0,
                chunks=stats[name].chunks if name in stats else 0,
                embedding_model=stats[name].embedding_model if name in stats else None,
            )
            for name, config in project.collections.items()
        ]

    @app.post("/collections/{name}/search")
    async def search(name: str, body: SearchRequest, request: Request) -> list[SearchHit]:
        knowledge = _knowledge(request)
        if name not in project.collections:
            raise HTTPException(status_code=404, detail=f"No collection named {name!r}.")
        try:
            results = await knowledge.search(
                name,
                body.query,
                top_k=body.top_k,
                mode=body.mode,
                filters=body.filters,
                use_rerank=body.rerank,
            )
        except ConfigError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return [
            SearchHit(
                text=r.chunk.text,
                score=r.score,
                retriever=r.retriever,
                uri=r.uri,
                title=r.title,
                heading_path=r.chunk.metadata.get("heading_path") or [],
                page=r.chunk.metadata.get("page"),
                metadata=r.chunk.metadata,
            )
            for r in results
        ]

    return app


def _knowledge(request: Request) -> KnowledgeBase:
    knowledge = request.app.state.knowledge
    if knowledge is None:
        raise HTTPException(status_code=404, detail="No collections are defined in ragkit.yaml.")
    return knowledge


def _sse_response(agent: Agent, tokens: AsyncIterator[str], thread_id: str) -> StreamingResponse:
    return StreamingResponse(
        _sse(agent, tokens, thread_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Thread-Id": thread_id},
    )


async def _sse(agent: Agent, tokens: AsyncIterator[str], thread_id: str) -> AsyncIterator[str]:
    """Events: start, token*, then approval_required (with pending actions) or end."""
    yield _event("start", {"thread_id": thread_id})
    try:
        async for token in tokens:
            yield _event("token", {"text": token})
        pending = await agent.pending(thread_id)
    except Exception as exc:  # the status code is already sent, so report errors in-stream
        yield _event("error", {"message": str(exc)})
        return
    if pending:
        actions = [_pending_info(p).model_dump() for p in pending]
        yield _event("approval_required", {"thread_id": thread_id, "pending": actions})
    else:
        yield _event("end", {"thread_id": thread_id})


def _event(name: str, data: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def _pending_info(action: PendingAction) -> PendingActionInfo:
    return PendingActionInfo(
        tool=action.tool,
        args=action.args,
        description=action.description,
        allowed_decisions=action.allowed_decisions,
    )


def _runtime(request: Request) -> AgentRuntime:
    return request.app.state.runtime


def _agent(request: Request, name: str) -> Agent:
    try:
        return _runtime(request).get(name)
    except AgentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
