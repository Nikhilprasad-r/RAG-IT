"""Tracing and usage accounting shared by every package.

Libraries only use the OpenTelemetry API, which does nothing until an SDK is
configured (`ragkit.observe.configure_tracing`), so tracing costs nothing when
it is off. Span and attribute names follow the OpenTelemetry GenAI semantic
conventions (gen_ai.*), plus `langfuse.*` hints that Langfuse uses for its UI.

`RunUsage` totals tokens and cost for one agent run; it lives in a context
variable so model calls made anywhere inside the run add to it.
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

tracer = trace.get_tracer("ragkit")

# OpenTelemetry GenAI semantic conventions (development status; names as of v1.41).
OPERATION = "gen_ai.operation.name"
PROVIDER = "gen_ai.provider.name"
REQUEST_MODEL = "gen_ai.request.model"
RESPONSE_MODEL = "gen_ai.response.model"
INPUT_TOKENS = "gen_ai.usage.input_tokens"
OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
FINISH_REASONS = "gen_ai.response.finish_reasons"
AGENT_NAME = "gen_ai.agent.name"
CONVERSATION_ID = "gen_ai.conversation.id"
TOOL_NAME = "gen_ai.tool.name"
TOOL_CALL_ID = "gen_ai.tool.call.id"
TOOL_ARGUMENTS = "gen_ai.tool.call.arguments"
TOOL_RESULT = "gen_ai.tool.call.result"
INPUT_MESSAGES = "gen_ai.input.messages"
OUTPUT_MESSAGES = "gen_ai.output.messages"
COST = "gen_ai.usage.cost"
# Langfuse reads these for sessions, observation types and input/output panes.
LANGFUSE_SESSION = "langfuse.session.id"
LANGFUSE_TYPE = "langfuse.observation.type"
LANGFUSE_INPUT = "langfuse.observation.input"
LANGFUSE_OUTPUT = "langfuse.observation.output"

_capture_content = False


def set_capture_content(enabled: bool) -> None:
    global _capture_content
    _capture_content = enabled


def capture_content() -> bool:
    """Whether prompts, replies and tool payloads may be recorded on spans."""
    return _capture_content


def set_content(span: Span, *, input: Any = None, output: Any = None) -> None:
    """Attach input/output to a span, only when content capture is enabled."""
    if not _capture_content:
        return
    if input is not None:
        span.set_attribute(LANGFUSE_INPUT, _json(input))
    if output is not None:
        span.set_attribute(LANGFUSE_OUTPUT, _json(output))


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    """Start a span as the current one; record exceptions and mark it failed."""
    with tracer.start_as_current_span(
        name, attributes=_clean(attributes), record_exception=True, set_status_on_exception=True
    ) as current:
        yield current


def mark_error(target: Span, error: BaseException) -> None:
    target.record_exception(error)
    target.set_status(Status(StatusCode.ERROR, str(error)))


@dataclass
class RunUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model_calls: int = 0
    priced: bool = True
    guardrails: list[dict[str, Any]] = field(default_factory=list)


current_usage: ContextVar[RunUsage | None] = ContextVar("ragkit_run_usage", default=None)


def record_guardrail(kind: str, action: str, detail: str = "") -> None:
    """Note that a guardrail acted, on the current span and the run's usage."""
    trace.get_current_span().add_event(
        "ragkit.guardrail", {"guardrail": kind, "action": action, "detail": detail[:200]}
    )
    if (usage := current_usage.get()) is not None:
        usage.guardrails.append({"guardrail": kind, "action": action, "detail": detail[:200]})


def _clean(attributes: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in attributes.items() if v is not None}


def _json(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
