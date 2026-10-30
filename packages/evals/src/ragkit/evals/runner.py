"""Run eval datasets against agents and collections, and check gates.

Agent metrics (each a mean over the cases that provide what it needs):

- `tool_recall`: share of `expected_tools` the agent called (an empty list
  expects no tool calls at all).
- `source_recall`: share of `expected_sources` found among retrieved passages.
- `field_accuracy`: share of fields in a dict `reference` that the structured
  output matches.
- `correctness`: LLM judge comparing the reply with a text `reference`.
- `groundedness`: LLM judge checking the reply against the retrieved passages.
- `error_rate`, `latency_p50_s`, `latency_p95_s`, `cost_usd` (total, when priced).

Search metrics: `hit_rate`, `recall`, `mrr` (all at the requested top_k),
`latency_p50_s`, `latency_p95_s`.

Gates map a metric to a minimum (`correctness: 0.8`) or, with a `max_`
prefix, a maximum (`max_latency_p95_s: 5`).
"""

import asyncio
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import AIMessage

from ragkit.agents.runtime import Agent, ApprovalDecision, new_thread_id
from ragkit.core.types import ScoredChunk
from ragkit.evals.dataset import AgentCase, SearchCase
from ragkit.evals.judge import Judge
from ragkit.retrieval.knowledge import KnowledgeBase, record_retrievals


@dataclass
class CaseResult:
    id: str
    input: str
    reply: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    tools_called: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    cost_usd: float | None = None
    error: str | None = None


@dataclass
class GateResult:
    metric: str
    threshold: float
    value: float | None
    passed: bool


@dataclass
class EvalReport:
    kind: Literal["agent", "search"]
    target: str
    dataset: str
    metrics: dict[str, float]
    cases: list[CaseResult]
    gates: list[GateResult] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.gates)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self) | {"passed": self.passed}, indent=2, default=str))
        return path


async def evaluate_agent(
    agent: Agent,
    cases: Sequence[AgentCase],
    *,
    dataset: str = "",
    judge: Judge | None = None,
    on_approval: Literal["approve", "reject"] = "reject",
    concurrency: int = 4,
    gate: dict[str, float] | None = None,
) -> EvalReport:
    semaphore = asyncio.Semaphore(concurrency)

    async def run(case: AgentCase) -> CaseResult:
        async with semaphore:
            return await _agent_case(agent, case, judge, on_approval)

    results = await asyncio.gather(*(run(case) for case in cases))
    metrics = _aggregate(results)
    metrics["error_rate"] = sum(r.error is not None for r in results) / len(results)
    if all(r.cost_usd is not None for r in results):
        metrics["cost_usd"] = sum(r.cost_usd or 0.0 for r in results)
    report = EvalReport("agent", agent.name, dataset, metrics, list(results))
    report.gates = check_gates(metrics, gate or {})
    return report


async def evaluate_search(
    knowledge: KnowledgeBase,
    collection: str,
    cases: Sequence[SearchCase],
    *,
    dataset: str = "",
    top_k: int = 5,
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid",
    rerank: bool = True,
    concurrency: int = 4,
    gate: dict[str, float] | None = None,
) -> EvalReport:
    semaphore = asyncio.Semaphore(concurrency)

    async def run(case: SearchCase) -> CaseResult:
        async with semaphore:
            started = time.perf_counter()
            hits = await knowledge.search(
                collection, case.query, top_k=top_k, mode=mode, use_rerank=rerank
            )
            uris = [h.uri for h in hits]
            first = next(
                (
                    rank
                    for rank, uri in enumerate(uris, 1)
                    if _matches_any(uri, case.expected_sources)
                ),
                None,
            )
            return CaseResult(
                id=case.id or "",
                input=case.query,
                sources=uris,
                latency_s=time.perf_counter() - started,
                metrics={
                    "hit_rate": float(first is not None),
                    "recall": _source_recall(case.expected_sources, uris),
                    "mrr": 1.0 / first if first else 0.0,
                },
            )

    results = await asyncio.gather(*(run(case) for case in cases))
    metrics = _aggregate(results)
    report = EvalReport("search", f"{collection} ({mode}, k={top_k})", dataset, metrics, results)
    report.gates = check_gates(metrics, gate or {})
    return report


def check_gates(metrics: dict[str, float], gate: dict[str, float]) -> list[GateResult]:
    results = []
    for name, threshold in gate.items():
        maximum = name.startswith("max_")
        metric = name.removeprefix("max_") if maximum else name
        value = metrics.get(metric)
        passed = value is not None and (value <= threshold if maximum else value >= threshold)
        results.append(GateResult(name, threshold, value, passed))
    return results


async def _agent_case(
    agent: Agent, case: AgentCase, judge: Judge | None, on_approval: str
) -> CaseResult:
    result = CaseResult(id=case.id or "", input=case.input)
    thread = new_thread_id()
    started = time.perf_counter()
    cost: float | None = 0.0
    passages: list[ScoredChunk] = []
    try:
        with record_retrievals() as passages:
            outcome = await agent.ainvoke(case.input, thread)
            cost = _add_cost(cost, outcome.usage)
            for _ in range(10):  # answer approval requests without real side effects
                if not outcome.pending:
                    break
                decisions = [
                    ApprovalDecision(
                        type=on_approval if on_approval in action.allowed_decisions else "reject",  # type: ignore[arg-type]
                    )
                    for action in outcome.pending
                ]
                outcome = await agent.resume(thread, decisions)
                cost = _add_cost(cost, outcome.usage)
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.latency_s = time.perf_counter() - started
        return result
    result.latency_s = time.perf_counter() - started
    result.cost_usd = cost
    result.reply = outcome.reply
    result.tools_called = await _tools_called(agent, thread)
    result.sources = list(dict.fromkeys(p.uri for p in passages))

    if case.expected_tools is not None:
        result.metrics["tool_recall"] = _tool_recall(case.expected_tools, result.tools_called)
    if case.expected_sources:
        result.metrics["source_recall"] = _source_recall(case.expected_sources, result.sources)
    if isinstance(case.reference, dict):
        result.metrics["field_accuracy"] = _field_accuracy(case.reference, outcome.structured)
    if judge is not None:
        if isinstance(case.reference, str):
            verdict = await judge.correctness(case.input, outcome.reply, case.reference)
            result.metrics["correctness"] = verdict.score
            result.notes["correctness"] = verdict.reason
        if passages:
            verdict = await judge.groundedness(case.input, outcome.reply, passages)
            result.metrics["groundedness"] = verdict.score
            result.notes["groundedness"] = verdict.reason
    return result


def _add_cost(total: float | None, usage: Any) -> float | None:
    if total is None or not usage.priced:
        return None
    return total + usage.cost_usd


async def _tools_called(agent: Agent, thread: str) -> list[str]:
    state = await agent.graph.aget_state({"configurable": {"thread_id": thread}})
    return [
        call["name"]
        for message in state.values.get("messages", [])
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]


def _tool_recall(expected: list[str], called: list[str]) -> float:
    if not expected:
        return float(not called)
    return sum(name in called for name in expected) / len(expected)


def _source_recall(expected: list[str], uris: list[str]) -> float:
    return sum(any(e in uri for uri in uris) for e in expected) / len(expected)


def _matches_any(uri: str, expected: list[str]) -> bool:
    return any(e in uri for e in expected)


def _field_accuracy(reference: dict[str, Any], structured: dict[str, Any] | None) -> float:
    if not reference:
        return 1.0
    if structured is None:
        return 0.0
    return sum(_same(value, structured.get(key)) for key, value in reference.items()) / len(
        reference
    )


def _same(expected: Any, actual: Any) -> bool:
    if isinstance(expected, int | float) and isinstance(actual, int | float):
        return math.isclose(float(expected), float(actual), rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().casefold() == actual.strip().casefold()
    return expected == actual


def _aggregate(results: Sequence[CaseResult]) -> dict[str, float]:
    names = sorted({name for r in results for name in r.metrics})
    metrics = {}
    for name in names:
        values = [r.metrics[name] for r in results if name in r.metrics]
        metrics[name] = sum(values) / len(values)
    latencies = sorted(r.latency_s for r in results)
    metrics["latency_p50_s"] = _percentile(latencies, 0.5)
    metrics["latency_p95_s"] = _percentile(latencies, 0.95)
    return metrics


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
    return values[index]
