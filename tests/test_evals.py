import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from typer.testing import CliRunner

from conftest import FakeChatModel, fake_models, tool_call, write_project
from ragkit.agents import AgentRuntime, add_agent
from ragkit.cli.main import app as cli
from ragkit.core.config import ProjectConfig
from ragkit.core.errors import ConfigError
from ragkit.evals import (
    AgentCase,
    Judge,
    SearchCase,
    check_gates,
    evaluate_agent,
    evaluate_search,
    load_cases,
)
from ragkit.ingest import ingest
from ragkit.retrieval import KnowledgeBase


def verdict(score: float, reason: str = "ok") -> AIMessage:
    call = {"name": "Verdict", "args": {"score": score, "reason": reason}, "id": "v"}
    return AIMessage("", tool_calls=[call])


def fake_judge(*scores: float) -> Judge:
    return Judge(FakeChatModel(messages=iter([verdict(s) for s in scores])))


async def test_agent_eval_scores_tools_sources_and_judged_answers(
    rag_project: ProjectConfig,
) -> None:
    await ingest(rag_project)
    add_agent("knowledge-agent", rag_project.agents_path)
    cases = [
        AgentCase(
            id="limits",
            input="What is the API rate limit?",
            reference="600 requests per minute per token.",
            expected_tools=["search_knowledge"],
            expected_sources=["api.md"],
        ),
        AgentCase(id="chitchat", input="Hi!", expected_tools=[]),
    ]
    factory = fake_models(
        tool_call("search_knowledge", query="API rate limit"),
        "600 requests per minute.",
        "Hello!",
    )
    judge = fake_judge(1.0, 0.5)  # correctness, then groundedness for the first case

    async with AgentRuntime(rag_project, model_factory=factory) as runtime:
        report = await evaluate_agent(
            runtime.get("knowledge-agent"),
            cases,
            judge=judge,
            concurrency=1,
            gate={"tool_recall": 1.0, "correctness": 0.9, "max_error_rate": 0.0},
        )

    limits, chitchat = report.cases
    assert limits.tools_called == ["search_knowledge"]
    assert "api.md" in limits.sources
    assert limits.metrics == {
        "tool_recall": 1.0,
        "source_recall": 1.0,
        "correctness": 1.0,
        "groundedness": 0.5,
    }
    assert chitchat.metrics == {"tool_recall": 1.0}
    assert report.metrics["tool_recall"] == 1.0
    assert report.metrics["error_rate"] == 0.0
    assert report.passed


async def test_approvals_are_rejected_during_evals(project: ProjectConfig) -> None:
    refund = tool_call("issue_refund", order_id="1", amount=5.0, reason="test")
    factory = fake_models(refund, "It was not approved.")

    async with AgentRuntime(project, model_factory=factory) as runtime:
        agent = runtime.get("support-agent")
        report = await evaluate_agent(
            agent, [AgentCase(input="refund order 1", expected_tools=["issue_refund"])]
        )
        state = await agent.graph.aget_state({"configurable": {"thread_id": "unused"}})

    (case,) = report.cases
    assert case.error is None
    assert case.metrics["tool_recall"] == 1.0
    assert case.reply == "It was not approved."
    assert state.values == {}


async def test_structured_output_field_accuracy(tmp_path: Path) -> None:
    project = write_project(tmp_path)
    add_agent("extractor", project.agents_path)
    extracted = {"vendor": "ACME GmbH ", "total": 95.2, "currency": "USD"}
    call = {"name": "Invoice", "args": extracted, "id": "c"}
    factory = fake_models(AIMessage("", tool_calls=[call]))
    reference = {"vendor": "Acme GmbH", "total": 95.20, "currency": "EUR"}

    async with AgentRuntime(project, model_factory=factory) as runtime:
        report = await evaluate_agent(
            runtime.get("extractor"),
            [AgentCase(input="invoice text", reference=reference)],
            gate={"field_accuracy": 0.9},
        )

    assert report.metrics["field_accuracy"] == pytest.approx(2 / 3)
    assert not report.passed


async def test_failing_cases_are_recorded(project: ProjectConfig) -> None:
    async with AgentRuntime(project, model_factory=fake_models()) as runtime:  # no replies
        report = await evaluate_agent(runtime.get("support-agent"), [AgentCase(input="hi")])

    assert report.cases[0].error is not None
    assert report.metrics["error_rate"] == 1.0


def test_gates_support_minimums_maximums_and_missing_metrics() -> None:
    gates = check_gates(
        {"recall": 0.7, "latency_p95_s": 2.0},
        {"recall": 0.8, "max_latency_p95_s": 3.0, "correctness": 0.5},
    )

    assert [(g.metric, g.passed) for g in gates] == [
        ("recall", False),
        ("max_latency_p95_s", True),
        ("correctness", False),
    ]


async def test_search_eval_hit_rate_recall_and_mrr(rag_project: ProjectConfig) -> None:
    await ingest(rag_project)
    cases = [
        SearchCase(query="how long is the reset link valid", expected_sources=["password.md"]),
        SearchCase(query="Retry-After header", expected_sources=["api.md"]),
        SearchCase(query="unrelated zebra migration", expected_sources=["nowhere.md"]),
    ]

    async with KnowledgeBase(rag_project) as kb:
        report = await evaluate_search(kb, "docs", cases, top_k=3, gate={"hit_rate": 0.6})

    assert [c.metrics["hit_rate"] for c in report.cases] == [1.0, 1.0, 0.0]
    assert report.metrics["hit_rate"] == pytest.approx(2 / 3)
    assert report.metrics["mrr"] > 0.5
    assert report.passed


def test_dataset_errors_point_at_the_line(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text('{"input": "ok"}\n\n{"inputt": "typo"}\n')

    with pytest.raises(ConfigError, match="cases.jsonl:3"):
        load_cases(path, AgentCase)


def test_cli_eval_search_writes_a_report_and_enforces_gates(
    rag_project: ProjectConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(rag_project.root)
    dataset = rag_project.root / "search.jsonl"
    dataset.write_text(
        json.dumps({"query": "audit logs SSO", "expected_sources": ["plans.md"]}) + "\n"
    )
    runner = CliRunner()
    assert runner.invoke(cli, ["ingest"]).exit_code == 0

    passed = runner.invoke(
        cli,
        [
            "eval",
            "search",
            "docs",
            str(dataset),
            "--mode",
            "sparse",
            "--mode",
            "hybrid",
            "--gate",
            "hit_rate=1",
            "--output",
            "out/search.json",
        ],
    )
    failed = runner.invoke(cli, ["eval", "search", "docs", str(dataset), "--gate", "mrr=2"])

    assert passed.exit_code == 0, passed.output
    assert "search: docs (sparse, k=5)" in passed.output
    reports = sorted((rag_project.root / "out").glob("search-docs-*.json"))
    assert len(reports) == 2
    assert json.loads(reports[0].read_text())["passed"] is True
    assert failed.exit_code == 1
    assert "Eval gate failed" in failed.output


def test_cli_eval_agent_without_judge(
    project: ProjectConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(project.root)
    (project.root / "ragkit.yaml").write_text("name: test\nevals: {concurrency: 1}\n")
    manifest = project.agents_path / "support-agent" / "agent.yaml"
    manifest.write_text(
        manifest.read_text().replace(
            "dataset: evals/golden.jsonl", "dataset: evals/golden.jsonl\n  gate: {tool_recall: 1}"
        )
    )
    replies = [
        tool_call("lookup_order", order_id="1042"),
        "Shipped.",
        tool_call("lookup_order", order_id="A-77"),
        "Shipped.",
        "I can't help with that.",
    ]
    monkeypatch.setattr(
        "ragkit.cli.main.AgentRuntime",
        lambda p, **kw: AgentRuntime(p, model_factory=fake_models(*replies), **kw),
    )
    monkeypatch.setattr("ragkit.cli.main.Judge", None)

    result = CliRunner().invoke(cli, ["eval", "agent", "support-agent", "--no-judge"])

    assert result.exit_code == 0, result.output
    assert "tool_recall" in result.output and "pass" in result.output
    assert list((project.root / ".ragkit" / "evals").glob("agent-support-agent-*.json"))
