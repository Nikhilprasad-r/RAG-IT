"""Rerankers re-score retrieved candidates with a model that reads query and text together.

Built in: `cohere` and `voyage` (hosted APIs, keys from COHERE_API_KEY /
VOYAGE_API_KEY) and `cross-encoder` (local sentence-transformers model, needs
`ragkit-retrieval[local-rerank]`). Add your own with
`registry.register("reranker", name, factory)`; the factory receives the
`RerankConfig`.
"""

import asyncio
import os
from collections.abc import Sequence
from typing import Any

import httpx

from ragkit.core.config import RerankConfig
from ragkit.core.errors import ConfigError
from ragkit.core.registry import registry
from ragkit.core.types import Query, ScoredChunk


class _HostedReranker:
    name: str
    url: str
    env_key: str
    default_model: str

    def __init__(self, model: str | None = None, *, client: httpx.AsyncClient | None = None):
        api_key = os.environ.get(self.env_key)
        if not api_key:
            raise ConfigError(f"The {self.name} reranker needs {self.env_key} set.")
        self.model = model or self.default_model
        self._client = client or httpx.AsyncClient(timeout=30)
        self._headers = {"Authorization": f"Bearer {api_key}"}

    async def rerank(
        self, query: Query, candidates: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        response = await self._client.post(
            self.url,
            headers=self._headers,
            json=self._payload(query.text, [c.chunk.indexed_text for c in candidates], top_n),
        )
        response.raise_for_status()
        return [
            candidates[index].model_copy(update={"score": score, "retriever": self.name})
            for index, score in self._results(response.json())
        ][:top_n]

    def _payload(self, query: str, documents: list[str], top_n: int) -> dict[str, Any]:
        raise NotImplementedError

    def _results(self, body: dict[str, Any]) -> list[tuple[int, float]]:
        raise NotImplementedError


class CohereReranker(_HostedReranker):
    name = "cohere"
    url = "https://api.cohere.com/v2/rerank"
    env_key = "COHERE_API_KEY"
    default_model = "rerank-v3.5"

    def _payload(self, query: str, documents: list[str], top_n: int) -> dict[str, Any]:
        return {"model": self.model, "query": query, "documents": documents, "top_n": top_n}

    def _results(self, body: dict[str, Any]) -> list[tuple[int, float]]:
        return [(r["index"], r["relevance_score"]) for r in body["results"]]


class VoyageReranker(_HostedReranker):
    name = "voyage"
    url = "https://api.voyageai.com/v1/rerank"
    env_key = "VOYAGE_API_KEY"
    default_model = "rerank-2.5"

    def _payload(self, query: str, documents: list[str], top_n: int) -> dict[str, Any]:
        return {"model": self.model, "query": query, "documents": documents, "top_k": top_n}

    def _results(self, body: dict[str, Any]) -> list[tuple[int, float]]:
        ranked = sorted(body["data"], key=lambda r: -r["relevance_score"])
        return [(r["index"], r["relevance_score"]) for r in ranked]


class CrossEncoderReranker:
    """Runs a cross-encoder locally, e.g. BAAI/bge-reranker-v2-m3."""

    name = "cross-encoder"
    default_model = "BAAI/bge-reranker-v2-m3"

    def __init__(self, model: str | None = None) -> None:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError as exc:
            raise ConfigError(
                "The cross-encoder reranker needs `ragkit-retrieval[local-rerank]` installed."
            ) from exc
        self.model = model or self.default_model
        self._encoder: Any = None
        self._load_lock = asyncio.Lock()

    async def rerank(
        self, query: Query, candidates: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        encoder = await self._load()
        pairs = [(query.text, c.chunk.indexed_text) for c in candidates]
        scores = await asyncio.to_thread(encoder.predict, pairs)
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda pair: -float(pair[1]))
        return [
            c.model_copy(update={"score": float(s), "retriever": self.name})
            for c, s in ranked[:top_n]
        ]

    async def _load(self) -> Any:
        # Loading (and possibly downloading) the model takes seconds; keep it off the
        # event loop and do it once, on first use.
        async with self._load_lock:
            if self._encoder is None:
                from sentence_transformers import CrossEncoder

                self._encoder = await asyncio.to_thread(CrossEncoder, self.model)
        return self._encoder


registry.register("reranker", "cohere", lambda config: CohereReranker(config.model))
registry.register("reranker", "voyage", lambda config: VoyageReranker(config.model))
registry.register("reranker", "cross-encoder", lambda config: CrossEncoderReranker(config.model))


def create_reranker(config: RerankConfig) -> Any:
    return registry.get("reranker", config.provider)(config)
