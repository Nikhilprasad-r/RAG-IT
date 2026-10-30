import json
import os
import uuid

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import ToolMessage
from typer.testing import CliRunner

from conftest import fake_models, tool_call
from ragkit.agents import AgentRuntime, ApprovalDecision, ApprovalError, ThreadWaitingError
from ragkit.cli.main import app as cli
from ragkit.core.config import ProjectConfig
from ragkit.core.errors import ConfigError
from ragkit.server import create_app

REFUND = tool_call("issue_refund", order_id="42", amount=30.0, reason="damaged")


async def _tool_messages(agent, thread_id: str) -> list[str]:
    state = await agent.graph.aget_state({"configurable": {"thread_id": thread_id}})
    return [m.text for m in state.values["messages"] if isinstance(m, ToolMessage)]


async def test_refund_waits_for_approval_then_runs(project: ProjectConfig) -> None:
    async with AgentRuntime(project, model_factory=fake_models(REFUND, "Refund issued.")) as rt:
        agent = rt.get("support-agent")
        first = await agent.ainvoke("Refund order 42, it arrived damaged")

        assert first.status == "waiting_for_approval"
        (action,) = first.pending
        assert action.tool == "issue_refund"
        assert action.args == {"order_id": "42", "amount": 30.0, "reason": "damaged"}
        assert action.allowed_decisions == ["approve", "edit", "reject"]
        assert await _tool_messages(agent, first.thread_id) == []

        done = await agent.resume(first.thread_id, [ApprovalDecision(type="approve")])

        assert done.status == "completed"
        assert done.reply == "Refund issued."
        assert await _tool_messages(agent, first.thread_id) == [
            "Refunded 30.00 on order 42 (damaged)."
        ]


async def test_edit_runs_the_changed_arguments(project: ProjectConfig) -> None:
    async with AgentRuntime(project, model_factory=fake_models(REFUND, "Partial refund.")) as rt:
        agent = rt.get("support-agent")
        first = await agent.ainvoke("Refund order 42")
        edit = ApprovalDecision(
            type="edit", args={"order_id": "42", "amount": 10.0, "reason": "goodwill"}
        )
        await agent.resume(first.thread_id, [edit])

        (message,) = await _tool_messages(agent, first.thread_id)
    assert message.endswith("Refunded 10.00 on order 42 (goodwill).")


async def test_reject_tells_the_model_and_skips_the_tool(project: ProjectConfig) -> None:
    async with AgentRuntime(project, model_factory=fake_models(REFUND, "Sorry.")) as rt:
        agent = rt.get("support-agent")
        first = await agent.ainvoke("Refund order 42")
        done = await agent.resume(
            first.thread_id, [ApprovalDecision(type="reject", message="over the limit")]
        )

        (message,) = await _tool_messages(agent, first.thread_id)
    assert done.reply == "Sorry."
    assert "over the limit" in message
    assert "Refunded" not in message


async def test_waiting_threads_reject_new_messages_and_bad_decisions(
    project: ProjectConfig,
) -> None:
    manifest = project.agents_path / "support-agent" / "agent.yaml"
    manifest.write_text(
        manifest.read_text().replace("[approve, edit, reject]", "[approve, reject]")
    )

    async with AgentRuntime(project, model_factory=fake_models(REFUND)) as rt:
        agent = rt.get("support-agent")
        first = await agent.ainvoke("Refund order 42")

        with pytest.raises(ThreadWaitingError):
            await agent.ainvoke("hello?", first.thread_id)
        with pytest.raises(ApprovalError, match="'edit' is not allowed"):
            await agent.resume(first.thread_id, [ApprovalDecision(type="edit", args={})])
        with pytest.raises(ApprovalError, match="1 action"):
            await agent.resume(first.thread_id, [ApprovalDecision(type="approve")] * 2)
        with pytest.raises(ApprovalError, match="nothing waiting"):
            await agent.resume("other-thread", [ApprovalDecision(type="approve")])


async def test_approval_for_unknown_tool_fails_at_startup(project: ProjectConfig) -> None:
    manifest = project.agents_path / "support-agent" / "agent.yaml"
    manifest.write_text(manifest.read_text().replace("[issue_refund]   #", "[refund]   #"))

    with pytest.raises(ConfigError, match="requires approval for refund"):
        async with AgentRuntime(project, model_factory=fake_models()):
            pass


def test_http_chat_resume_and_stream(project: ProjectConfig) -> None:
    factory = fake_models(REFUND, "Refund issued.", REFUND, "Done again.")
    with TestClient(create_app(project, model_factory=factory)) as client:
        waiting = client.post("/agents/support-agent/chat", json={"message": "refund 42"}).json()
        thread = waiting["thread_id"]
        blocked = client.post(
            "/agents/support-agent/chat", json={"message": "hi", "thread_id": thread}
        )
        resumed = client.post(
            "/agents/support-agent/resume",
            json={"thread_id": thread, "decisions": [{"type": "approve"}]},
        ).json()
        again = client.post(
            "/agents/support-agent/resume",
            json={"thread_id": thread, "decisions": [{"type": "approve"}]},
        )

        streamed = client.post(
            "/agents/support-agent/stream", json={"message": "refund 42", "thread_id": "s1"}
        )
        stream_resumed = client.post(
            "/agents/support-agent/resume/stream",
            json={"thread_id": "s1", "decisions": [{"type": "approve"}]},
        )

    assert waiting["status"] == "waiting_for_approval"
    assert waiting["pending"][0]["tool"] == "issue_refund"
    assert waiting["pending"][0]["args"]["amount"] == 30.0
    assert blocked.status_code == 409
    assert resumed["status"] == "completed" and resumed["reply"] == "Refund issued."
    assert again.status_code == 409

    events = [b.split("\n") for b in streamed.text.strip().split("\n\n")]
    assert events[-1][0] == "event: approval_required"
    assert json.loads(events[-1][1].removeprefix("data: "))["pending"][0]["tool"] == "issue_refund"
    assert "event: end" in stream_resumed.text
    assert "Done again." in "".join(
        json.loads(line.removeprefix("data: "))["text"]
        for line in stream_resumed.text.splitlines()
        if line.startswith("data: ") and '"text"' in line
    )


def test_cli_prompts_for_approval(project: ProjectConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(project.root)
    monkeypatch.setattr(
        "ragkit.cli.main.AgentRuntime",
        lambda p, **kw: AgentRuntime(p, model_factory=fake_models(REFUND, "Refund issued."), **kw),
    )

    result = CliRunner().invoke(cli, ["chat", "support-agent", "refund 42"], input="a\n")

    assert result.exit_code == 0, result.output
    assert "Approval needed: issue_refund" in result.output
    assert "Refund issued." in result.output


@pytest.mark.skipif(
    not os.environ.get("RAGKIT_TEST_DATABASE_URL"), reason="RAGKIT_TEST_DATABASE_URL is not set"
)
async def test_postgres_checkpointer_resumes_in_another_process(project: ProjectConfig) -> None:
    """A turn paused in one server can be approved from another (multi-pod deployments)."""
    project = project.model_copy(update={"database_url": os.environ["RAGKIT_TEST_DATABASE_URL"]})
    manifest = project.agents_path / "support-agent" / "agent.yaml"
    manifest.write_text(
        manifest.read_text().replace("checkpointer: memory", "checkpointer: postgres")
    )
    thread = f"pg-{uuid.uuid4().hex}"

    async with AgentRuntime(project, model_factory=fake_models(REFUND)) as first_server:
        waiting = await first_server.get("support-agent").ainvoke("Refund order 42", thread)
    async with AgentRuntime(project, model_factory=fake_models("Refund issued.")) as second_server:
        agent = second_server.get("support-agent")
        assert [p.tool for p in await agent.pending(thread)] == ["issue_refund"]
        done = await agent.resume(thread, [ApprovalDecision(type="approve")])
        messages = await _tool_messages(agent, thread)

    assert waiting.status == "waiting_for_approval"
    assert done.reply == "Refund issued."
    assert messages == ["Refunded 30.00 on order 42 (damaged)."]
