"""`KnowledgeBase`: the one search entry point used by agents, the CLI and the server."""

import asyncio
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Self

from ragkit.core import telemetry as t
from ragkit.core.config import ProjectConfig, RerankConfig
from ragkit.core.errors import ConfigError
from ragkit.core.types import Query, ScoredChunk
from ragkit.retrieval.embedders import Embedder, create_embedder
from ragkit.retrieval.hybrid import HybridRetriever, SearchMode, rrf_fuse
from ragkit.retrieval.rerankers import create_reranker
from ragkit.retrieval.stores import Store, create_store

DEFAULT_TOP_K = 8

_retrieval_log: ContextVar[list[ScoredChunk] | None] = ContextVar(
    "ragkit_retrieval_log", default=None
)


@contextmanager
def record_retrievals() -> Iterator[list[ScoredChunk]]:
    """Collect every passage returned by searches made inside the block (used by evals)."""
    log: list[ScoredChunk] = []
    token = _retrieval_log.set(log)
    try:
        yield log
    finally:
        _retrieval_log.reset(token)


class KnowledgeBase:
    """Owns the store, embedder and rerankers for a project.

    Use as an async context manager. A `store` or `embedder` passed in is used
    as is and not closed on exit.
    """

    def __init__(
        self,
        project: ProjectConfig,
        *,
        store: Store | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.project = project
        self._owns_store = store is None
        self.store = store or create_store(project)
        self._embedder = embedder
        self._rerankers: dict[tuple[str, str | None], Any] = {}
        self._checked_models: set[str] = set()

    async def __aenter__(self) -> Self:
        if self._owns_store:
            await self.store.open()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._owns_store:
            await self.store.close()

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = create_embedder(self.project.embeddings)
        return self._embedder

    async def search(
        self,
        collection: str,
        text: str,
        *,
        top_k: int | None = None,
        mode: SearchMode = "hybrid",
        candidates: int = 50,
        rerank: RerankConfig | None = None,
        use_rerank: bool = True,
        filters: dict[str, Any] | None = None,
        principals: Sequence[str] = (),
        tenant_id: str = "default",
    ) -> list[ScoredChunk]:
        """Hybrid search, then rerank with `rerank` or the project's default reranker."""
        self.project.collection(collection)
        if mode != "sparse":
            await self._check_embedding_model(collection)
        rerank = (rerank or self.project.rerank) if use_rerank else None
        final_k = top_k or (rerank.top_n if rerank else DEFAULT_TOP_K)

        query = Query(
            text=text,
            collection=collection,
            tenant_id=tenant_id,
            principals=list(principals),
            filters=filters or {},
            top_k=max(candidates, final_k) if rerank else final_k,
        )
        attributes = {
            t.LANGFUSE_TYPE: "retriever",
            "ragkit.collection": collection,
            "ragkit.search.mode": mode,
            "ragkit.search.top_k": final_k,
            "ragkit.search.candidates": candidates,
            "ragkit.rerank.provider": rerank.provider if rerank else None,
        }
        with t.span(f"retrieve {collection}", **attributes) as span:
            retriever = HybridRetriever(self.store, self.embedder, mode=mode, candidates=candidates)
            results = await retriever.retrieve(query)
            if rerank and results:
                with t.span(f"rerank {rerank.provider}", **{"ragkit.rerank.model": rerank.model}):
                    results = await self._reranker(rerank).rerank(query, results, final_k)
            results = results[:final_k]
            span.set_attribute("ragkit.search.results", len(results))
            t.set_content(
                span,
                input=text,
                output=[
                    {"uri": r.uri, "score": r.score, "text": r.chunk.text[:300]} for r in results
                ],
            )
        if (log := _retrieval_log.get()) is not None:
            log.extend(results)
        return results

    async def search_many(
        self, collections: Sequence[str], text: str, *, top_k: int | None = None, **options: Any
    ) -> list[ScoredChunk]:
        """Search several collections and merge their rankings with RRF."""
        if len(collections) == 1:
            return await self.search(collections[0], text, top_k=top_k, **options)
        rankings = await asyncio.gather(
            *(self.search(name, text, top_k=top_k, **options) for name in collections)
        )
        merged = rrf_fuse(rankings)
        return merged[: top_k or max(len(r) for r in rankings)]

    def _reranker(self, config: RerankConfig) -> Any:
        key = (config.provider, config.model)
        if key not in self._rerankers:
            self._rerankers[key] = create_reranker(config)
        return self._rerankers[key]

    async def _check_embedding_model(self, collection: str) -> None:
        if collection in self._checked_models:
            return
        stats = await self.store.collection_stats(collection)
        if stats and stats.embedding_model and stats.embedding_model != self.embedder.model:
            raise ConfigError(
                f"Collection {collection!r} was indexed with {stats.embedding_model} but the "
                f"project now uses {self.embedder.model}. Run `ragkit ingest {collection} --full`."
            )
        self._checked_models.add(collection)


def format_results(results: Sequence[ScoredChunk]) -> str:
    """Render results as numbered, citable passages for a model's context."""
    if not results:
        return "No matching passages found."
    blocks = []
    for number, result in enumerate(results, start=1):
        meta = result.chunk.metadata
        where = " > ".join(meta.get("heading_path") or [])
        page = f", page {meta['page']}" if meta.get("page") else ""
        label = result.title or result.uri
        heading = f"[{number}] {label}{page}" + (f" — {where}" if where else "")
        blocks.append(f"{heading}\nsource: {result.uri}\n{result.chunk.text}")
    return "\n\n".join(blocks)
