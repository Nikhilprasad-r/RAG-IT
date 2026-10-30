"""Connects an agent's `retrieval` section to the project's knowledge base.

- `react` agents get a `search_knowledge` tool and decide when, and how often,
  to search (agentic retrieval).
- `rag-chat` agents retrieve once per user message and answer from that
  context in a single model call (single-pass RAG: cheaper and faster).
"""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool

from ragkit.agents.manifest import RetrievalSpec
from ragkit.core.config import ProjectConfig
from ragkit.retrieval.knowledge import KnowledgeBase, format_results


async def _search(knowledge: KnowledgeBase, spec: RetrievalSpec, query: str) -> str:
    results = await knowledge.search_many(
        spec.collections,
        query,
        top_k=spec.top_k,
        mode=spec.mode,
        candidates=spec.candidates,
        rerank=spec.rerank,
        use_rerank=spec.use_rerank,
    )
    return format_results(results)


def search_tool(knowledge: KnowledgeBase, spec: RetrievalSpec, project: ProjectConfig) -> BaseTool:
    described = "; ".join(
        f"{name}: {project.collection(name).description}".rstrip(": ") for name in spec.collections
    )

    async def search_knowledge(query: str) -> str:
        return await _search(knowledge, spec, query)

    return StructuredTool.from_function(
        coroutine=search_knowledge,
        name="search_knowledge",
        description=(
            "Search the knowledge base and return numbered passages with their sources. "
            "Use a short, specific query; search again with different words if the results "
            f"do not answer the question. Collections: {described}."
        ),
    )


class RetrievedContextMiddleware(AgentMiddleware):
    """Adds passages relevant to the latest user message to the system prompt."""

    def __init__(self, knowledge: KnowledgeBase, spec: RetrievalSpec) -> None:
        super().__init__()
        self.knowledge = knowledge
        self.spec = spec

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> Any:
        question = next(
            (m.text for m in reversed(request.messages) if isinstance(m, HumanMessage)), ""
        )
        if not question:
            return await handler(request)
        context = await _search(self.knowledge, self.spec, question)
        prompt = request.system_prompt or ""
        system = SystemMessage(f"{prompt}\n\n<retrieved_context>\n{context}\n</retrieved_context>")
        return await handler(request.override(system_message=system))
