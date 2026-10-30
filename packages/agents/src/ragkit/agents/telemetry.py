"""Spans and usage accounting for agent runs.

`TelemetryMiddleware` wraps every model and tool call. It emits GenAI spans
(`chat <model>`, `execute_tool <tool>`) and adds each call's tokens and cost to
the run's `RunUsage`. Because the spans are current while the call runs,
retrieval and sub-agent spans nest under the tool call that caused them.

`CostBudgetMiddleware` ends a run politely once its cost reaches
`budget.max_cost_usd`.
"""

import json
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager, suppress
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, hook_config
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from opentelemetry.trace import Span, Status, StatusCode

from ragkit.core import telemetry as t
from ragkit.core.config import ModelPrice


def model_identity(model: str | BaseChatModel) -> tuple[str, str]:
    """(provider, model name) for a `provider:model` string or a chat model object."""
    if isinstance(model, str):
        provider, _, name = model.partition(":")
        return (provider, name) if name else ("unknown", provider)
    name = getattr(model, "model", None) or getattr(model, "model_name", None) or "unknown"
    return getattr(model, "_llm_type", "unknown"), str(name)


class TelemetryMiddleware(AgentMiddleware):
    def __init__(
        self, agent_name: str, model: str | BaseChatModel, pricing: dict[str, ModelPrice]
    ) -> None:
        super().__init__()
        self.agent_name = agent_name
        self.provider, self.model = model_identity(model)
        self.pricing = pricing

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> Any:
        attributes = {
            t.OPERATION: "chat",
            t.PROVIDER: self.provider,
            t.REQUEST_MODEL: self.model,
            t.AGENT_NAME: self.agent_name,
            t.LANGFUSE_TYPE: "generation",
        }
        with t.span(f"chat {self.model}", **attributes) as span:
            if t.capture_content():
                messages = [request.system_message, *request.messages]
                span.set_attribute(t.INPUT_MESSAGES, _messages_json(messages))
            response = await handler(request)
            message = _last_ai_message(response)
            if message is not None:
                self._record(span, message)
            return response

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[Any]]
    ) -> Any:
        call = request.tool_call
        attributes = {
            t.OPERATION: "execute_tool",
            t.TOOL_NAME: call["name"],
            t.TOOL_CALL_ID: call.get("id"),
            t.AGENT_NAME: self.agent_name,
            t.LANGFUSE_TYPE: "tool",
        }
        with t.span(f"execute_tool {call['name']}", **attributes) as span:
            if t.capture_content():
                span.set_attribute(t.TOOL_ARGUMENTS, json.dumps(call.get("args", {}), default=str))
            result = await handler(request)
            if isinstance(result, ToolMessage):
                if t.capture_content():
                    span.set_attribute(t.TOOL_RESULT, result.text[:10_000])
                if result.status == "error":
                    span.set_status(Status(StatusCode.ERROR, result.text[:200]))
            return result

    def _record(self, span: Span, message: AIMessage) -> None:
        usage = message.usage_metadata or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        span.set_attribute(t.INPUT_TOKENS, input_tokens)
        span.set_attribute(t.OUTPUT_TOKENS, output_tokens)
        metadata = message.response_metadata or {}
        if response_model := metadata.get("model_name"):
            span.set_attribute(t.RESPONSE_MODEL, str(response_model))
        if reason := metadata.get("finish_reason"):
            span.set_attribute(t.FINISH_REASONS, [str(reason)])
        if t.capture_content():
            span.set_attribute(t.OUTPUT_MESSAGES, _messages_json([message]))

        price = self.pricing.get(f"{self.provider}:{self.model}")
        cost = price.cost(input_tokens, output_tokens) if price else None
        if cost is not None:
            span.set_attribute(t.COST, cost)
        if (run := t.current_usage.get()) is not None:
            run.model_calls += 1
            run.input_tokens += input_tokens
            run.output_tokens += output_tokens
            if cost is None:
                run.priced = False
            else:
                run.cost_usd += cost


class CostBudgetMiddleware(AgentMiddleware):
    """Stops before a model call once the run has spent `max_usd`."""

    def __init__(self, max_usd: float) -> None:
        super().__init__()
        self.max_usd = max_usd

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        run = t.current_usage.get()
        if run is None or run.cost_usd < self.max_usd:
            return None
        t.record_guardrail("cost_budget", "stopped", f"spent ${run.cost_usd:.4f}")
        message = (
            f"I stopped because this request reached its cost budget (${self.max_usd:g}). "
            "Try a narrower question."
        )
        return {"messages": [AIMessage(message)], "jump_to": "end"}


@contextmanager
def agent_run(agent_name: str, thread_id: str, message: str | None) -> Iterator[t.RunUsage]:
    """The `invoke_agent` span and usage totals for one turn of an agent.

    A sub-agent's usage is added to its caller's, so budgets and reports cover
    the whole delegation tree.
    """
    usage = t.RunUsage()
    parent = t.current_usage.get()
    token = t.current_usage.set(usage)
    attributes = {
        t.OPERATION: "invoke_agent",
        t.AGENT_NAME: agent_name,
        t.CONVERSATION_ID: thread_id,
        t.LANGFUSE_SESSION: thread_id,
        t.LANGFUSE_TYPE: "agent",
    }
    try:
        with t.span(f"invoke_agent {agent_name}", **attributes) as span:
            t.set_content(span, input=message)
            try:
                yield usage
            finally:
                span.set_attribute(t.INPUT_TOKENS, usage.input_tokens)
                span.set_attribute(t.OUTPUT_TOKENS, usage.output_tokens)
                if usage.priced and usage.model_calls:
                    span.set_attribute(t.COST, usage.cost_usd)
    finally:
        # An async generator closed from another context cannot reset its token.
        with suppress(ValueError):
            t.current_usage.reset(token)
        if parent is not None:
            parent.input_tokens += usage.input_tokens
            parent.output_tokens += usage.output_tokens
            parent.cost_usd += usage.cost_usd
            parent.model_calls += usage.model_calls
            parent.priced = parent.priced and usage.priced
            parent.guardrails += usage.guardrails


def _last_ai_message(response: Any) -> AIMessage | None:
    messages = response.result if isinstance(response, ModelResponse) else [response]
    return next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)


def _messages_json(messages: list[BaseMessage | None]) -> str:
    out = []
    for message in messages:
        if message is None:
            continue
        entry: dict[str, Any] = {"role": message.type, "content": message.text}
        if isinstance(message, AIMessage) and message.tool_calls:
            entry["tool_calls"] = [
                {"name": c["name"], "args": c["args"]} for c in message.tool_calls
            ]
        out.append(entry)
    return json.dumps(out, default=str, ensure_ascii=False)
