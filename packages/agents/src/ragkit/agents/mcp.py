"""Tools from MCP servers declared under `mcp_servers` in ragkit.yaml.

Each server is started (stdio) or connected to (http, sse, websocket) once per
runtime, and its session stays open so tool calls do not pay a start-up cost.
Agents pick tools with `tools.mcp: [{server, allow}]`.
"""

from collections.abc import Iterable
from contextlib import AsyncExitStack
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.client.stdio import get_default_environment

from ragkit.agents.manifest import AgentManifest
from ragkit.core.config import McpServerConfig, ProjectConfig
from ragkit.core.errors import ConfigError


def connection(config: McpServerConfig, project: ProjectConfig) -> dict[str, Any]:
    if config.transport == "stdio":
        return {
            "transport": "stdio",
            "command": config.command,
            "args": config.args,
            # An explicit env replaces the child's environment, so keep PATH and HOME.
            "env": {**get_default_environment(), **config.env},
            "cwd": str(project.root / config.cwd) if config.cwd else str(project.root),
        }
    if config.transport == "http":
        return {
            "transport": "streamable_http",
            "url": config.url,
            "headers": config.headers,
            "timeout": config.timeout,
        }
    if config.transport == "sse":
        return {
            "transport": "sse",
            "url": config.url,
            "headers": config.headers,
            "timeout": config.timeout,
        }
    return {"transport": "websocket", "url": config.url}


async def open_mcp_tools(
    project: ProjectConfig, servers: Iterable[str], stack: AsyncExitStack
) -> dict[str, list[BaseTool]]:
    """Open a session to each server for the lifetime of `stack` and list its tools."""
    names = sorted(set(servers))
    for name in names:
        if name not in project.mcp_servers:
            available = ", ".join(sorted(project.mcp_servers)) or "none"
            raise ConfigError(
                f"MCP server {name!r} is not defined under mcp_servers in ragkit.yaml. "
                f"Defined: {available}."
            )
    client = MultiServerMCPClient(
        {name: connection(project.mcp_servers[name], project) for name in names}
    )
    tools: dict[str, list[BaseTool]] = {}
    for name in names:
        try:
            session = await stack.enter_async_context(client.session(name))
            tools[name] = await load_mcp_tools(session, server_name=name)
        except ConfigError:
            raise
        except Exception as exc:
            raise ConfigError(f"Could not start MCP server {name!r}: {exc}") from exc
    return tools


def select_mcp_tools(
    manifest: AgentManifest, available: dict[str, list[BaseTool]]
) -> list[BaseTool]:
    selected: list[BaseTool] = []
    for spec in manifest.tools.mcp:
        tools = {tool.name: tool for tool in available[spec.server]}
        missing = [name for name in spec.allow if name not in tools]
        if missing:
            raise ConfigError(
                f"Agent {manifest.name!r} allows {', '.join(missing)} from MCP server "
                f"{spec.server!r}, which offers: {', '.join(sorted(tools)) or 'no tools'}."
            )
        selected += [tools[name] for name in spec.allow] if spec.allow else list(tools.values())
    return selected
