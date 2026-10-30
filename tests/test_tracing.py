import base64
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from conftest import FakeChatModel, _agent_folder, fake_models, tool_call, write_project
from ragkit.agents import AgentRuntime, add_agent
from ragkit.core import telemetry
from ragkit.core.config import ObservabilityConfig, ProjectConfig
from ragkit.core.errors import ConfigError
from ragkit.ingest import ingest
from ragkit.observe import otlp_target

USAGE = {"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200}
PRICING = "pricing:\n  'fake:m': {input_per_million: 1.0, output_per_million: 10.0}\n"


def _reply(text: str = "", **kwargs) -> AIMessage:
    return AIMessage(text, usage_metadata=USAGE, **kwargs)


class PricedFake(FakeChatModel):
    """A fake model that reports provider 'fake' and model 'm', priced in PRICING."""

    model: str = "m"

    @property
    def _llm_type(self) -> str:
        return "fake"


def priced(*replies: AIMessage):
    return lambda _manifest: PricedFake(messages=iter(replies))


def _tree(spans) -> dict[str, list[str]]:
    """Span name -> names of its direct children."""
    by_id = {s.context.span_id: s for s in spans}
    children: dict[str, list[str]] = {s.name: [] for s in spans}
    for s in spans:
        if s.parent and s.parent.span_id in by_id:
            children[by_id[s.parent.span_id].name].append(s.name)
    return {k: sorted(v) for k, v in children.items()}


def _named(spans, name: str):
    return next(s for s in spans.get_finished_spans() if s.name == name)


async def test_agent_run_emits_genai_spans_with_usage_and_cost(tmp_path: Path, spans) -> None:
    project = write_project(tmp_path, "name: t\n" + PRICING)
    add_agent("support-agent", project.agents_path)
    factory = priced(
        _reply(tool_calls=tool_call("lookup_order", order_id="7").tool_calls), _reply("Shipped.")
    )

    async with AgentRuntime(project, model_factory=factory) as rt:
        result = await rt.get("support-agent").ainvoke("Where is order 7?", "thread-1")

    finished = spans.get_finished_spans()
    assert _tree(finished)["invoke_agent support-agent"] == [
        "chat m",
        "chat m",
        "execute_tool lookup_order",
    ]
    root = _named(spans, "invoke_agent support-agent").attributes
    assert root["gen_ai.operation.name"] == "invoke_agent"
    assert root["gen_ai.conversation.id"] == root["langfuse.session.id"] == "thread-1"
    assert root["gen_ai.usage.input_tokens"] == 2000
    assert root["gen_ai.usage.cost"] == pytest.approx(0.006)
    chat = _named(spans, "chat m").attributes
    assert chat["gen_ai.provider.name"] == "fake"
    assert chat["gen_ai.usage.output_tokens"] == 200
    assert chat["gen_ai.usage.cost"] == pytest.approx(0.003)
    assert "gen_ai.input.messages" not in chat  # content capture is off by default
    tool = _named(spans, "execute_tool lookup_order").attributes
    assert tool["gen_ai.tool.name"] == "lookup_order"
    assert result.usage.cost_usd == pytest.approx(0.006)
    assert result.usage.model_calls == 2


async def test_content_capture_records_prompts_and_tool_payloads(tmp_path: Path, spans) -> None:
    project = write_project(tmp_path)
    add_agent("support-agent", project.agents_path)
    factory = fake_models(tool_call("lookup_order", order_id="7"), "Shipped.")
    telemetry.set_capture_content(True)
    try:
        async with AgentRuntime(project, model_factory=factory) as rt:
            await rt.get("support-agent").ainvoke("Where is order 7?")
    finally:
        telemetry.set_capture_content(False)

    root = _named(spans, "invoke_agent support-agent").attributes
    assert root["langfuse.observation.input"] == "Where is order 7?"
    chat = next(s for s in spans.get_finished_spans() if s.name.startswith("chat "))
    assert "Where is order 7?" in chat.attributes["gen_ai.input.messages"]
    tool = _named(spans, "execute_tool lookup_order").attributes
    assert tool["gen_ai.tool.call.arguments"] == '{"order_id": "7"}'
    assert tool["gen_ai.tool.call.result"] == "Order 7 is shipped."


async def test_retrieval_spans_nest_under_the_step_that_searched(
    rag_project: ProjectConfig, spans
) -> None:
    await ingest(rag_project)
    add_agent("rag-chat", rag_project.agents_path)
    add_agent("knowledge-agent", rag_project.agents_path)
    spans.clear()
    replies = {
        "rag-chat": ["Answer [1]."],
        "knowledge-agent": [tool_call("search_knowledge", query="api limit"), "600 per minute."],
    }

    def factory(manifest):
        return fake_models(*replies[manifest.name])(manifest)

    async with AgentRuntime(rag_project, model_factory=factory) as rt:
        await rt.get("rag-chat").ainvoke("password reset?")
        await rt.get("knowledge-agent").ainvoke("api limit?")

    tree = _tree(spans.get_finished_spans())
    assert "retrieve docs" in tree["invoke_agent rag-chat"]  # before the chat call
    assert tree["execute_tool search_knowledge"] == ["retrieve docs"]
    retrieve = next(
        s
        for s in spans.get_finished_spans()
        if s.name == "retrieve docs" and s.attributes["ragkit.search.results"] > 0
    )
    assert retrieve.attributes["langfuse.observation.type"] == "retriever"
    assert retrieve.attributes["ragkit.search.mode"] == "hybrid"


async def test_subagent_runs_nest_and_add_to_the_parent_usage(tmp_path: Path, spans) -> None:
    project = write_project(tmp_path)
    _agent_folder(tmp_path, "billing", "description: Billing.\n")
    _agent_folder(tmp_path, "boss", "subagents: [billing]\n")

    def factory(manifest):
        replies = {
            "boss": [
                _reply(tool_calls=tool_call("ask_billing", task="price").tool_calls),
                _reply("Done."),
            ],
            "billing": [_reply("$12")],
        }[manifest.name]
        return fake_models(*replies)(manifest)

    async with AgentRuntime(project, model_factory=factory) as rt:
        result = await rt.get("boss").ainvoke("price?")

    tree = _tree(spans.get_finished_spans())
    assert tree["execute_tool ask_billing"] == ["invoke_agent billing"]
    assert result.usage.model_calls == 3
    assert result.usage.input_tokens == 3000


async def test_cost_budget_stops_the_run(tmp_path: Path, spans) -> None:
    project = write_project(tmp_path, "name: t\n" + PRICING)
    add_agent("support-agent", project.agents_path)
    manifest = project.agents_path / "support-agent" / "agent.yaml"
    manifest.write_text(
        manifest.read_text().replace("max_steps: 8", "max_steps: 8\n  max_cost_usd: 0.005")
    )
    looping = [
        _reply(tool_calls=tool_call("lookup_order", call_id=f"c{i}", order_id=str(i)).tool_calls)
        for i in range(5)
    ]

    async with AgentRuntime(project, model_factory=priced(*looping)) as rt:
        result = await rt.get("support-agent").ainvoke("loop")

    assert "cost budget" in result.reply
    assert result.usage.model_calls == 2  # $0.003 per call; stops before the third
    assert result.usage.guardrails[0]["guardrail"] == "cost_budget"


async def test_cost_budget_needs_a_price(tmp_path: Path) -> None:
    project = write_project(tmp_path, "name: t\ndefault_model: openai:gpt-5.5\n")
    add_agent("support-agent", project.agents_path)
    manifest = project.agents_path / "support-agent" / "agent.yaml"
    manifest.write_text(
        manifest.read_text().replace("max_steps: 8", "max_steps: 8\n  max_cost_usd: 1")
    )

    with pytest.raises(ConfigError, match="no pricing for openai:gpt-5.5"):
        async with AgentRuntime(project):
            pass


async def test_ingest_spans_report_outcomes(rag_project: ProjectConfig, spans) -> None:
    await ingest(rag_project)

    root = _named(spans, "ingest docs").attributes
    assert root["ragkit.ingest.added"] == 3
    outcomes = sorted(
        s.attributes["ragkit.ingest.outcome"]
        for s in spans.get_finished_spans()
        if s.name == "ingest_document"
    )
    assert outcomes == ["added", "added", "added"]


def test_langfuse_and_phoenix_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-1")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-2")

    url, headers = otlp_target(ObservabilityConfig(exporter="langfuse"))
    assert url == "https://cloud.langfuse.com/api/public/otel/v1/traces"
    assert headers["Authorization"] == "Basic " + base64.b64encode(b"pk-lf-1:sk-lf-2").decode()

    eu = ObservabilityConfig(
        exporter="langfuse", langfuse={"host": "https://us.cloud.langfuse.com/"}
    )
    assert otlp_target(eu)[0] == "https://us.cloud.langfuse.com/api/public/otel/v1/traces"
    assert otlp_target(ObservabilityConfig(exporter="phoenix"))[0] == (
        "http://localhost:6006/v1/traces"
    )

    monkeypatch.delenv("LANGFUSE_SECRET_KEY")
    with pytest.raises(ConfigError, match="LANGFUSE_SECRET_KEY"):
        otlp_target(ObservabilityConfig(exporter="langfuse"))
