import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

from conftest import FakeChatModel, fake_models, tool_call, write_project
from ragkit.agents import AgentManifest, AgentRuntime, add_agent
from ragkit.core.config import ProjectConfig
from ragkit.core.errors import ConfigError
from ragkit.server import create_app

MCP_SERVER = '''
from mcp.server.fastmcp import FastMCP

server = FastMCP("calc")


@server.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@server.tool()
def shutdown() -> str:
    """Pretend to do something dangerous."""
    return "shut down"


server.run()
'''


def _agent(root: Path, name: str, manifest: str, prompt: str = "You help.") -> None:
    folder = root / "agents" / name
    (folder / "prompts").mkdir(parents=True)
    (folder / "agent.yaml").write_text(f"name: {name}\n{manifest}")
    (folder / "prompts" / "system.md").write_text(prompt)


@pytest.fixture
def mcp_project(tmp_path: Path) -> ProjectConfig:
    (tmp_path / "calc_server.py").write_text(MCP_SERVER)
    config = (
        "name: mcp-test\n"
        "mcp_servers:\n"
        "  calc:\n"
        "    transport: stdio\n"
        f"    command: {json.dumps(sys.executable)}\n"
        "    args: [calc_server.py]\n"
    )
    return write_project(tmp_path, config)


async def test_agent_calls_an_mcp_tool(mcp_project: ProjectConfig) -> None:
    _agent(mcp_project.root, "math", "tools:\n  mcp:\n    - server: calc\n      allow: [add]\n")
    factory = fake_models(tool_call("add", a=2, b=3), "2 + 3 = 5")

    async with AgentRuntime(mcp_project, model_factory=factory) as runtime:
        agent = runtime.get("math")
        tools = {t.name for t in agent.graph.nodes["tools"].bound._tools_by_name.values()}
        result = await agent.ainvoke("What is 2 + 3?")
        state = await agent.graph.aget_state({"configurable": {"thread_id": result.thread_id}})

    assert tools == {"add"}  # `shutdown` is not on the allowlist
    (message,) = [m for m in state.values["messages"] if isinstance(m, ToolMessage)]
    assert message.text == "5"
    assert result.reply == "2 + 3 = 5"


async def test_mcp_tools_can_require_approval(mcp_project: ProjectConfig) -> None:
    _agent(
        mcp_project.root,
        "ops",
        "tools:\n  mcp: [{server: calc}]\nhitl:\n  require_approval: [shutdown]\n",
    )

    async with AgentRuntime(
        mcp_project, model_factory=fake_models(tool_call("shutdown"))
    ) as runtime:
        result = await runtime.get("ops").ainvoke("shut it down")

    assert [p.tool for p in result.pending] == ["shutdown"]


async def test_mcp_configuration_errors(mcp_project: ProjectConfig) -> None:
    _agent(mcp_project.root, "math", "tools:\n  mcp:\n    - server: calc\n      allow: [mul]\n")
    with pytest.raises(ConfigError, match="offers: add, shutdown"):
        async with AgentRuntime(mcp_project, model_factory=fake_models()):
            pass

    other = write_project(mcp_project.root / "other", "name: other\n")
    _agent(other.root, "math", "tools:\n  mcp: [{server: nope}]\n")
    with pytest.raises(ConfigError, match="'nope' is not defined under mcp_servers"):
        async with AgentRuntime(other, model_factory=fake_models()):
            pass


def test_mcp_server_config_is_validated(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="need a url"):
        write_project(tmp_path, "name: t\nmcp_servers:\n  x: {transport: http}\n")


def _models_by_agent(**replies: list) -> object:
    def factory(manifest: AgentManifest) -> FakeChatModel:
        messages = [AIMessage(r) if isinstance(r, str) else r for r in replies[manifest.name]]
        return FakeChatModel(messages=iter(messages))

    return factory


@pytest.fixture
def team(tmp_path: Path) -> ProjectConfig:
    project = write_project(tmp_path)
    _agent(tmp_path, "billing", "description: Answers billing questions.\n")
    _agent(tmp_path, "boss", "pattern: supervisor\nsubagents: [billing]\n")
    return project


async def test_supervisor_delegates_with_a_per_conversation_thread(team: ProjectConfig) -> None:
    factory = _models_by_agent(
        boss=[tool_call("ask_billing", task="What does Pro cost?"), "Pro costs $12."],
        billing=["$12 per user."],
    )

    async with AgentRuntime(team, model_factory=factory) as runtime:
        boss, billing = runtime.get("boss"), runtime.get("billing")
        result = await boss.ainvoke("How much is Pro?", "t1")
        sub_state = await billing.graph.aget_state({"configurable": {"thread_id": "t1/billing"}})
        boss_state = await boss.graph.aget_state({"configurable": {"thread_id": "t1"}})

    assert result.reply == "Pro costs $12."
    assert [m.text for m in sub_state.values["messages"]] == [
        "What does Pro cost?",
        "$12 per user.",
    ]
    (tool_message,) = [m for m in boss_state.values["messages"] if isinstance(m, ToolMessage)]
    assert tool_message.text == "$12 per user."


async def test_supervisor_streams_only_its_own_reply(team: ProjectConfig) -> None:
    factory = _models_by_agent(
        boss=[tool_call("ask_billing", task="price?"), "Final answer."],
        billing=["Internal sub agent words."],
    )

    async with AgentRuntime(team, model_factory=factory) as runtime:
        tokens = [t async for t in runtime.get("boss").astream("price?", "t2")]

    assert "".join(tokens) == "Final answer."


async def test_subagent_graph_errors(tmp_path: Path) -> None:
    project = write_project(tmp_path)
    _agent(tmp_path, "a", "subagents: [b]\n")
    _agent(tmp_path, "b", "subagents: [a]\n")
    with pytest.raises(ConfigError, match="cycle: a -> b -> a"):
        async with AgentRuntime(project, model_factory=fake_models()):
            pass

    project = write_project(tmp_path / "two")
    _agent(project.root, "a", "subagents: [ghost]\n")
    with pytest.raises(ConfigError, match="'ghost', which is not an agent"):
        async with AgentRuntime(project, model_factory=fake_models()):
            pass

    project = write_project(tmp_path / "three")
    add_agent("support-agent", project.agents_path)
    _agent(project.root, "boss", "subagents: [support-agent]\n")
    with pytest.raises(ConfigError, match="needs human approval"):
        async with AgentRuntime(project, model_factory=fake_models()):
            pass


async def test_extractor_returns_validated_structure(tmp_path: Path) -> None:
    project = write_project(tmp_path)
    add_agent("extractor", project.agents_path)
    invoice = {"vendor": "Acme GmbH", "invoice_number": "A-1043", "currency": "EUR", "total": 95.2}
    factory = fake_models(
        AIMessage("", tool_calls=[{"name": "Invoice", "args": invoice, "id": "c"}])
    )

    with TestClient(create_app(project, model_factory=factory)) as client:
        body = client.post("/agents/extractor/chat", json={"message": "INVOICE #A-1043 ..."}).json()

    assert body["structured"]["vendor"] == "Acme GmbH"
    assert body["structured"]["total"] == 95.2
    assert body["structured"]["line_items"] == []
    assert json.loads(body["reply"])["invoice_number"] == "A-1043"


def test_output_schema_must_be_a_pydantic_model(tmp_path: Path) -> None:
    project = write_project(tmp_path)
    add_agent("extractor", project.agents_path)
    manifest = project.agents_path / "extractor" / "agent.yaml"
    manifest.write_text(manifest.read_text().replace("schema.py:Invoice", "schema.py:Field"))

    import asyncio

    async def start() -> None:
        async with AgentRuntime(project, model_factory=fake_models()):
            pass

    with pytest.raises(ConfigError, match="must name a Pydantic model"):
        asyncio.run(start())


async def test_structured_output_survives_strict_checkpoint_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")
    project = write_project(tmp_path)
    add_agent("extractor", project.agents_path)
    call = {"name": "Invoice", "args": {"vendor": "Acme", "total": 1.0}, "id": "c"}
    factory = fake_models(AIMessage("", tool_calls=[call]), AIMessage("", tool_calls=[call]))

    async with AgentRuntime(project, model_factory=factory) as runtime:
        agent = runtime.get("extractor")
        first = await agent.ainvoke("invoice one", "t")
        second = await agent.ainvoke("invoice two", "t")  # reloads the stored Invoice
        state = await agent.graph.aget_state({"configurable": {"thread_id": "t"}})

    assert first.structured["vendor"] == second.structured["vendor"] == "Acme"
    assert type(state.values["structured_response"]).__name__ == "Invoice"


async def test_runtime_can_load_a_subset_of_agents(team: ProjectConfig) -> None:
    _agent(team.root, "broken", "tools:\n  mcp: [{server: missing}]\n")

    async with AgentRuntime(team, model_factory=fake_models(), only=["boss"]) as runtime:
        assert sorted(runtime.agents) == ["billing", "boss"]

    with pytest.raises(ConfigError, match="missing"):
        async with AgentRuntime(team, model_factory=fake_models()):
            pass
