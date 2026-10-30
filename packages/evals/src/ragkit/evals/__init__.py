"""Evaluate agents and search quality, with gates for CI."""

from ragkit.evals.dataset import AgentCase, SearchCase, load_cases
from ragkit.evals.judge import Judge, Verdict
from ragkit.evals.runner import (
    CaseResult,
    EvalReport,
    GateResult,
    check_gates,
    evaluate_agent,
    evaluate_search,
)

__all__ = [
    "AgentCase",
    "CaseResult",
    "EvalReport",
    "GateResult",
    "Judge",
    "SearchCase",
    "Verdict",
    "check_gates",
    "evaluate_agent",
    "evaluate_search",
    "load_cases",
]
