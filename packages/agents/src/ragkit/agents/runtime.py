"""Builds runnable agents from the folders in a project's agents directory.

An agent folder holds `agent.yaml`, a system prompt, and optionally `tools.py`
and an output schema. `AgentRuntime` loads every folder, opens the
checkpointers, knowledge base and MCP sessions they need, and keeps the
compiled graphs for the lifetime of the process.

A turn either completes or stops for human approval when the agent calls a tool
listed under `hitl.require_approval`; `Agent.resume` continues it with the
reviewer's decisions.
"""

import importlib.util
import json
import logging
import sys
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Self

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel

from ragkit.agents.checkpointers import open_checkpointers
from ragkit.agents.manifest import MANIFEST_FILENAME, AgentManifest
from ragkit.agents.mcp import open_mcp_tools, select_mcp_tools
from ragkit.agents.retrieval import RetrievedContextMiddleware, search_tool
from ragkit.agents.subagents import build_order, subagent_tool
from ragkit.core.config import ProjectConfig
from ragkit.core.errors import ConfigError, RagkitError
from ragkit.retrieval.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)

ModelFactory = Callable[[AgentManifest], str | BaseChatModel]


class AgentNotFoundError(RagkitError):
    pass


class ApprovalError(RagkitError):
    """A resume did not match the actions waiting for approval."""


class ThreadWaitingError(RagkitError):
    """A new message was sent to a thread that is waiting for approval."""


@dataclass(frozen=True)
class PendingAction:
    """A tool call waiting for a human decision."""

    tool: str
    args: dict[str, Any]
    description: str
    allowed_decisions: list[str]
    interrupt_id: str


class ApprovalDecision(BaseModel):
    type: Literal["approve", "edit", "reject"]
    args: dict[str, Any] | None = None
    """For `edit`: the arguments to run the tool with instead."""
    message: str | None = None
    """For `reject`: the reason, passed to the model."""


@dataclass
class AgentResult:
    reply: str
    thread_id: str
    pending: list[PendingAction] = field(default_factory=list)
    structured: dict[str, Any] | None = None

    @property
    def status(self) -> Literal["completed", "waiting_for_approval"]:
        return "waiting_for_approval" if self.pending else "completed"


@dataclass
class Agent:
    manifest: AgentManifest
    path: Path
    graph: CompiledStateGraph

    @property
    def name(self) -> str:
        return self.manifest.name

    async def ainvoke(self, message: str, thread_id: str | None = None) -> AgentResult:
        """Run one turn."""
        thread_id = thread_id or new_thread_id()
        await self._check_not_waiting(thread_id)
        result = await self.graph.ainvoke(_user_input(message), _config(thread_id))
        return await self._result(result, thread_id)

    async def resume(self, thread_id: str, decisions: Sequence[ApprovalDecision]) -> AgentResult:
        """Continue a turn that stopped for approval, with one decision per pending action."""
        command = await self._resume_command(thread_id, decisions)
        result = await self.graph.ainvoke(command, _config(thread_id))
        return await self._result(result, thread_id)

    async def astream(self, message: str, thread_id: str) -> AsyncIterator[str]:
        """Yield the reply's text as the model produces it. Check `pending` afterwards."""
        await self._check_not_waiting(thread_id)
        async for text in self._stream(_user_input(message), thread_id):
            yield text

    async def astream_resume(
        self, thread_id: str, decisions: Sequence[ApprovalDecision]
    ) -> AsyncIterator[str]:
        """Check the decisions now, then return an iterator over the continued reply."""
        command = await self._resume_command(thread_id, decisions)
        return self._stream(command, thread_id)

    async def pending(self, thread_id: str) -> list[PendingAction]:
        """Tool calls on this thread that are waiting for a decision."""
        state = await self.graph.aget_state(_config(thread_id))
        actions = []
        for interrupt in state.interrupts:
            value = interrupt.value
            if not isinstance(value, dict) or "action_requests" not in value:
                continue
            allowed = {c["action_name"]: c["allowed_decisions"] for c in value["review_configs"]}
            for request in value["action_requests"]:
                actions.append(
                    PendingAction(
                        tool=request["name"],
                        args=request["args"],
                        description=request.get("description", ""),
                        allowed_decisions=list(allowed.get(request["name"], [])),
                        interrupt_id=interrupt.id,
                    )
                )
        return actions

    async def _stream(self, graph_input: Any, thread_id: str) -> AsyncIterator[str]:
        async for chunk, metadata in self.graph.astream(
            graph_input, _config(thread_id), stream_mode="messages"
        ):
            # Sub-agents run inside tools and their tokens surface here too; keep ours only.
            if metadata.get("lc_agent_name", self.name) != self.name:
                continue
            if isinstance(chunk, AIMessageChunk) and chunk.text:
                yield chunk.text

    async def _check_not_waiting(self, thread_id: str) -> None:
        if await self.pending(thread_id):
            raise ThreadWaitingError(
                f"Thread {thread_id} is waiting for approval; resume it before sending more."
            )

    async def _resume_command(
        self, thread_id: str, decisions: Sequence[ApprovalDecision]
    ) -> Command:
        pending = await self.pending(thread_id)
        if not pending:
            raise ApprovalError(f"Thread {thread_id} has nothing waiting for approval.")
        if len(decisions) != len(pending):
            raise ApprovalError(
                f"{len(pending)} action(s) are waiting for approval but {len(decisions)} "
                "decision(s) were given."
            )
        by_interrupt: dict[str, list[dict[str, Any]]] = {}
        for action, decision in zip(pending, decisions, strict=True):
            if decision.type not in action.allowed_decisions:
                raise ApprovalError(
                    f"{decision.type!r} is not allowed for {action.tool}; "
                    f"allowed: {', '.join(action.allowed_decisions)}."
                )
            decisions_for = by_interrupt.setdefault(action.interrupt_id, [])
            decisions_for.append(_hitl_decision(action, decision))
        return Command(resume={key: {"decisions": value} for key, value in by_interrupt.items()})

    async def _result(self, result: dict[str, Any], thread_id: str) -> AgentResult:
        structured = result.get("structured_response")
        if isinstance(structured, BaseModel):
            structured = structured.model_dump(mode="json")
        reply = next(
            (m.text for m in reversed(result["messages"]) if isinstance(m, AIMessage) and m.text),
            "",
        )
        if structured is not None and not reply:
            reply = json.dumps(structured, indent=2)
        return AgentResult(
            reply=reply,
            thread_id=thread_id,
            pending=await self.pending(thread_id),
            structured=structured,
        )


class AgentRuntime:
    """Async context manager that owns every agent in a project.

    `model_factory` overrides how each agent's model is chosen; tests use it to
    inject fake chat models. `knowledge` is shared with the caller (the server
    passes its own); without it, one is opened when an agent uses retrieval.
    """

    def __init__(
        self,
        project: ProjectConfig,
        *,
        model_factory: ModelFactory | None = None,
        knowledge: KnowledgeBase | None = None,
    ):
        self.project = project
        self._model_factory = model_factory
        self._knowledge = knowledge
        self._stack = AsyncExitStack()
        self.agents: dict[str, Agent] = {}

    async def __aenter__(self) -> Self:
        try:
            await self._start()
        except BaseException:
            await self._stack.aclose()
            raise
        return self

    async def _start(self) -> None:
        folders = discover_agents(self.project.agents_path)
        by_path = {path: AgentManifest.from_file(path / MANIFEST_FILENAME) for path in folders}
        _check_unique_names(by_path)
        _check_collections(by_path, self.project)
        paths = {manifest.name: path for path, manifest in by_path.items()}
        manifests = {manifest.name: manifest for manifest in by_path.values()}
        order = build_order(manifests)

        savers = await self._stack.enter_async_context(
            open_checkpointers(self.project, {m.memory.checkpointer for m in manifests.values()})
        )
        knowledge = self._knowledge
        if knowledge is None and any(m.retrieval for m in manifests.values()):
            knowledge = await self._stack.enter_async_context(KnowledgeBase(self.project))
        mcp_tools = await open_mcp_tools(
            self.project,
            {spec.server for m in manifests.values() for spec in m.tools.mcp},
            self._stack,
        )

        for name in order:
            manifest = manifests[name]
            extra = select_mcp_tools(manifest, mcp_tools) + [
                subagent_tool(self.agents[child]) for child in manifest.subagents
            ]
            self.agents[name] = build_agent(
                paths[name],
                manifest,
                self.project,
                checkpointer=savers[manifest.memory.checkpointer],
                model=self._model_factory(manifest) if self._model_factory else None,
                knowledge=knowledge,
                extra_tools=extra,
            )

    async def __aexit__(self, *exc_info: object) -> None:
        self.agents.clear()
        await self._stack.aclose()

    def get(self, name: str) -> Agent:
        try:
            return self.agents[name]
        except KeyError:
            available = ", ".join(sorted(self.agents)) or "none"
            raise AgentNotFoundError(f"No agent named {name!r}. Available: {available}.") from None


def discover_agents(agents_dir: Path) -> list[Path]:
    if not agents_dir.is_dir():
        return []
    return sorted(p for p in agents_dir.iterdir() if (p / MANIFEST_FILENAME).is_file())


def build_agent(
    path: Path,
    manifest: AgentManifest,
    project: ProjectConfig,
    *,
    checkpointer: BaseCheckpointSaver,
    model: str | BaseChatModel | None = None,
    knowledge: KnowledgeBase | None = None,
    extra_tools: Sequence[BaseTool] = (),
) -> Agent:
    for feature in manifest.unwired_features():
        logger.warning("Agent %r declares %s, which is not wired yet.", manifest.name, feature)

    tools = [*_load_tools(path, manifest), *extra_tools]
    middleware: list[Any] = [
        ModelCallLimitMiddleware(run_limit=manifest.budget.max_steps, exit_behavior="end")
    ]
    if manifest.fallback_models:
        middleware.append(ModelFallbackMiddleware(*manifest.fallback_models))
    if manifest.retrieval:
        if knowledge is None:
            raise ConfigError(f"Agent {manifest.name!r} uses retrieval but no knowledge base.")
        if manifest.pattern == "rag-chat":
            middleware.append(RetrievedContextMiddleware(knowledge, manifest.retrieval))
        else:
            tools.append(search_tool(knowledge, manifest.retrieval, project))
    _check_tool_names(manifest, tools)
    if manifest.hitl.require_approval:
        middleware.append(
            HumanInTheLoopMiddleware(
                {
                    tool: {"allowed_decisions": list(manifest.hitl.decisions)}
                    for tool in manifest.hitl.require_approval
                }
            )
        )

    graph = create_agent(
        model=model or manifest.model or project.default_model,
        tools=tools,
        system_prompt=_read_prompt(path, manifest),
        middleware=middleware,
        response_format=_load_output_schema(path, manifest),
        checkpointer=checkpointer,
        name=manifest.name,
    )
    return Agent(manifest=manifest, path=path, graph=graph)


def new_thread_id() -> str:
    return uuid.uuid4().hex


def _hitl_decision(action: PendingAction, decision: ApprovalDecision) -> dict[str, Any]:
    if decision.type == "edit":
        args = decision.args if decision.args is not None else action.args
        return {"type": "edit", "edited_action": {"name": action.tool, "args": args}}
    if decision.type == "reject" and decision.message:
        return {"type": "reject", "message": decision.message}
    return {"type": decision.type}


def _user_input(message: str) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": message}]}


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _check_unique_names(manifests: dict[Path, AgentManifest]) -> None:
    seen: dict[str, Path] = {}
    for path, manifest in manifests.items():
        if manifest.name in seen:
            raise ConfigError(
                f"Agent name {manifest.name!r} is used by both {seen[manifest.name]} and {path}."
            )
        seen[manifest.name] = path


def _check_collections(manifests: dict[Path, AgentManifest], project: ProjectConfig) -> None:
    for manifest in manifests.values():
        for collection in manifest.retrieval.collections if manifest.retrieval else []:
            if collection not in project.collections:
                raise ConfigError(
                    f"Agent {manifest.name!r} searches collection {collection!r}, which is not "
                    f"defined under collections in ragkit.yaml."
                )


def _check_tool_names(manifest: AgentManifest, tools: Sequence[Any]) -> None:
    names = [getattr(tool, "name", getattr(tool, "__name__", "")) for tool in tools]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ConfigError(f"Agent {manifest.name!r} has two tools named {', '.join(duplicates)}.")
    unknown = [tool for tool in manifest.hitl.require_approval if tool not in names]
    if unknown:
        raise ConfigError(
            f"Agent {manifest.name!r} requires approval for {', '.join(unknown)}, but its tools "
            f"are: {', '.join(names) or 'none'}."
        )


def _read_prompt(path: Path, manifest: AgentManifest) -> str:
    prompt_path = path / manifest.system_prompt
    if not prompt_path.is_file():
        raise ConfigError(f"System prompt for agent {manifest.name!r} not found: {prompt_path}")
    return prompt_path.read_text().strip()


def _load_tools(path: Path, manifest: AgentManifest) -> list[Any]:
    names = manifest.tools.local
    if not names:
        return []
    module = _import_agent_file(path / "tools.py", manifest.name)
    missing = [name for name in names if not hasattr(module, name)]
    if missing:
        raise ConfigError(
            f"Agent {manifest.name!r} lists tools not defined in tools.py: {', '.join(missing)}"
        )
    return [getattr(module, name) for name in names]


def _load_output_schema(path: Path, manifest: AgentManifest) -> type[BaseModel] | None:
    if not manifest.output_schema:
        return None
    filename, _, attribute = manifest.output_schema.partition(":")
    module = _import_agent_file(path / filename, manifest.name)
    schema = getattr(module, attribute, None)
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        raise ConfigError(
            f"Agent {manifest.name!r}: output_schema {manifest.output_schema} must name a "
            "Pydantic model."
        )
    return schema


def _import_agent_file(file: Path, agent_name: str) -> ModuleType:
    if not file.is_file():
        raise ConfigError(f"Agent {agent_name!r} needs {file}, which does not exist.")
    module_name = f"ragkit_agents_user.{agent_name.replace('-', '_')}.{file.stem}"
    spec = importlib.util.spec_from_file_location(module_name, file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
