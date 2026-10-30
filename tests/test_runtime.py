import logging
import shutil
from collections.abc import Iterator

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from conftest import fake_model_from, fake_models, tool_call
from ragkit.agents import AgentNotFoundError, AgentRuntime
from ragkit.core.config import ProjectConfig
from ragkit.core.errors import ConfigError


async def test_agent_calls_its_tool_and_replies(project: ProjectConfig) -> None:
    factory = fake_models(tool_call("lookup_order", order_id="42"), "Order 42 is on its way.")

    async with AgentRuntime(project, model_factory=factory) as runtime:
        agent = runtime.get("support-agent")
        result = await agent.ainvoke("Where is order 42?")
        reply, thread_id = result.reply, result.thread_id
        state = await agent.graph.aget_state({"configurable": {"thread_id": thread_id}})

    assert reply == "Order 42 is on its way."
    tool_messages = [m for m in state.values["messages"] if isinstance(m, ToolMessage)]
    assert [m.content for m in tool_messages] == ["Order 42 is shipped."]


async def test_thread_keeps_conversation_history(project: ProjectConfig) -> None:
    async with AgentRuntime(project, model_factory=fake_models("Hi!", "Still here.")) as runtime:
        agent = runtime.get("support-agent")
        result = await agent.ainvoke("Hello")
        thread_id = result.thread_id
        result = await agent.ainvoke("Are you there?", thread_id)
        reply, same_thread = result.reply, result.thread_id
        state = await agent.graph.aget_state({"configurable": {"thread_id": thread_id}})

    assert (reply, same_thread) == ("Still here.", thread_id)
    assert len(state.values["messages"]) == 4


async def test_step_budget_stops_a_looping_agent(project: ProjectConfig) -> None:
    def always_call_tool() -> Iterator[AIMessage]:
        n = 0
        while True:
            n += 1
            yield tool_call("lookup_order", call_id=f"call_{n}", order_id=str(n))

    async with AgentRuntime(project, model_factory=fake_model_from(always_call_tool())) as runtime:
        agent = runtime.get("support-agent")
        result = await agent.ainvoke("Loop forever")
        thread_id = result.thread_id
        state = await agent.graph.aget_state({"configurable": {"thread_id": thread_id}})

    model_calls = [m for m in state.values["messages"] if isinstance(m, AIMessage) and m.tool_calls]
    assert len(model_calls) == 8  # budget.max_steps in the template


async def test_unknown_agent_is_reported(project: ProjectConfig) -> None:
    async with AgentRuntime(project, model_factory=fake_models()) as runtime:
        with pytest.raises(AgentNotFoundError, match="support-agent"):
            runtime.get("missing")


async def test_tool_listed_but_not_defined_is_an_error(project: ProjectConfig) -> None:
    manifest = project.agents_path / "support-agent" / "agent.yaml"
    manifest.write_text(manifest.read_text().replace("issue_refund]", "issue_refund, refund]"))

    with pytest.raises(ConfigError, match="not defined in tools.py: refund"):
        async with AgentRuntime(project, model_factory=fake_models()):
            pass


async def test_duplicate_agent_names_are_an_error(project: ProjectConfig) -> None:
    shutil.copytree(project.agents_path / "support-agent", project.agents_path / "copy")

    with pytest.raises(ConfigError, match="used by both"):
        async with AgentRuntime(project, model_factory=fake_models()):
            pass


async def test_unwired_features_are_logged(
    project: ProjectConfig, caplog: pytest.LogCaptureFixture
) -> None:
    manifest = project.agents_path / "support-agent" / "agent.yaml"
    manifest.write_text(manifest.read_text() + "guardrails: [pii]\n")

    with caplog.at_level(logging.WARNING):
        async with AgentRuntime(project, model_factory=fake_models()):
            pass

    assert "guardrails" in caplog.text
