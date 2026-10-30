"""Agents that delegate to other agents.

Each sub-agent becomes an `ask_<name>` tool. The sub-agent runs on its own
thread derived from the caller's (`<thread>/<name>`), so it remembers earlier
delegations within the same conversation without seeing the whole parent chat.
"""

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool

from ragkit.agents.manifest import AgentManifest
from ragkit.core.errors import ConfigError

if TYPE_CHECKING:
    from ragkit.agents.runtime import Agent


def tool_name(agent_name: str) -> str:
    return "ask_" + agent_name.replace("-", "_")


def subagent_tool(agent: "Agent") -> BaseTool:
    async def delegate(task: str, config: RunnableConfig) -> str:
        from ragkit.agents.runtime import new_thread_id

        parent = config.get("configurable", {}).get("thread_id") or new_thread_id()
        result = await agent.ainvoke(task, f"{parent}/{agent.name}")
        if result.structured is not None:
            return json.dumps(result.structured)
        return result.reply

    description = agent.manifest.description or f"The {agent.name} agent."
    return StructuredTool.from_function(
        coroutine=delegate,
        name=tool_name(agent.name),
        description=(
            f"Delegate a task to the {agent.name} agent. {description} It cannot see this "
            "conversation, so give it the complete task in plain words."
        ),
    )


def build_order(manifests: Mapping[str, AgentManifest]) -> list[str]:
    """Agent names ordered so every sub-agent is built before the agents that use it."""
    for manifest in manifests.values():
        for child in manifest.subagents:
            if child not in manifests:
                raise ConfigError(
                    f"Agent {manifest.name!r} delegates to {child!r}, which is not an agent "
                    "in this project."
                )
            if manifests[child].hitl.require_approval:
                raise ConfigError(
                    f"Agent {manifest.name!r} delegates to {child!r}, which needs human "
                    "approval; approvals inside sub-agents are not supported yet."
                )

    order: list[str] = []
    state: dict[str, str] = {}

    def visit(name: str, path: list[str]) -> None:
        if state.get(name) == "done":
            return
        if state.get(name) == "visiting":
            cycle = " -> ".join([*path[path.index(name) :], name])
            raise ConfigError(f"Agents delegate to each other in a cycle: {cycle}.")
        state[name] = "visiting"
        for child in manifests[name].subagents:
            visit(child, [*path, name])
        state[name] = "done"
        order.append(name)

    for name in sorted(manifests):
        visit(name, [])
    return order
