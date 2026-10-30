import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from typer.testing import CliRunner

from conftest import FakeChatModel, fake_models, tool_call
from ragkit.agents import AgentManifest, AgentRuntime, add_agent
from ragkit.cli.main import app as cli
from ragkit.core.config import ProjectConfig
from ragkit.core.errors import ConfigError
from ragkit.ingest import ingest
from ragkit.server import create_app


async def test_rag_chat_answers_from_retrieved_context(rag_project: ProjectConfig) -> None:
    await ingest(rag_project)
    add_agent("rag-chat", rag_project.agents_path)
    model = FakeChatModel(messages=iter([AIMessage("Links expire after 30 minutes [1].")]))

    async with AgentRuntime(rag_project, model_factory=lambda _m: model) as runtime:
        result = await runtime.get("rag-chat").ainvoke("How long does a password reset link last?")
        reply = result.reply

    assert reply == "Links expire after 30 minutes [1]."
    system = model.seen[0][0]
    assert isinstance(system, SystemMessage)
    assert "<retrieved_context>" in system.text
    assert "expires after 30 minutes" in system.text
    assert "source: account/password.md" in system.text
    assert "cite the passages" in system.text.lower()


async def test_knowledge_agent_searches_with_a_tool(rag_project: ProjectConfig) -> None:
    await ingest(rag_project)
    add_agent("knowledge-agent", rag_project.agents_path)
    factory = fake_models(tool_call("search_knowledge", query="API rate limit"), "600 per minute.")

    async with AgentRuntime(rag_project, model_factory=factory) as runtime:
        agent = runtime.get("knowledge-agent")
        result = await agent.ainvoke("What is the API rate limit?")
        reply, thread_id = result.reply, result.thread_id
        state = await agent.graph.aget_state({"configurable": {"thread_id": thread_id}})

    (tool_message,) = [m for m in state.values["messages"] if isinstance(m, ToolMessage)]
    assert reply == "600 per minute."
    assert "600 requests per minute" in tool_message.text
    assert tool_message.text.startswith("[1] ")


async def test_agent_with_unknown_collection_fails_at_startup(rag_project: ProjectConfig) -> None:
    add_agent("rag-chat", rag_project.agents_path)
    manifest = rag_project.agents_path / "rag-chat" / "agent.yaml"
    manifest.write_text(manifest.read_text().replace("[docs]", "[handbook]"))

    with pytest.raises(ConfigError, match="'handbook'"):
        async with AgentRuntime(rag_project, model_factory=fake_models()):
            pass


def test_rag_chat_without_retrieval_is_invalid() -> None:
    with pytest.raises(ValueError, match="needs a retrieval section"):
        AgentManifest.model_validate({"name": "x", "pattern": "rag-chat"})


async def test_server_collections_and_search(rag_project: ProjectConfig) -> None:
    await ingest(rag_project)

    with TestClient(create_app(rag_project, model_factory=fake_models())) as client:
        collections = client.get("/collections").json()
        hits = client.post(
            "/collections/docs/search", json={"query": "single sign-on", "top_k": 2}
        ).json()
        missing = client.post("/collections/nope/search", json={"query": "x"})

    assert collections == [
        {
            "name": "docs",
            "description": "Nimbus product docs",
            "documents": 3,
            "chunks": collections[0]["chunks"],
            "embedding_model": "hash:256",
        }
    ]
    assert hits[0]["uri"] == "plans.md"
    assert hits[0]["heading_path"] == ["Nimbus plans", "Team plan"]
    assert len(hits) == 2
    assert missing.status_code == 404


def test_cli_ingest_search_and_collections(
    rag_project: ProjectConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(rag_project.root)
    runner = CliRunner()

    ingested = runner.invoke(cli, ["ingest"])
    again = runner.invoke(cli, ["ingest", "docs"])
    found = runner.invoke(cli, ["search", "docs", "audit logs", "-k", "1"])
    listed = runner.invoke(cli, ["collections"])

    assert ingested.exit_code == 0, ingested.output
    assert "docs" in ingested.output
    assert again.exit_code == 0
    assert found.exit_code == 0, found.output
    assert "Team plan" in found.output
    assert "hash:256 (256d)" in listed.output
