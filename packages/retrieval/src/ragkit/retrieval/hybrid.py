"""Hybrid retrieval: dense and keyword search fused with Reciprocal Rank Fusion."""

import asyncio
from collections.abc import Sequence
from typing import Literal

from ragkit.core.types import Query, ScoredChunk
from ragkit.retrieval.embedders import Embedder
from ragkit.retrieval.stores import Store

SearchMode = Literal["dense", "sparse", "hybrid"]


def rrf_fuse(rankings: Sequence[Sequence[ScoredChunk]], *, k: int = 60) -> list[ScoredChunk]:
    """Merge ranked lists; each list adds 1 / (k + rank) to a chunk's score.

    RRF needs no score calibration between retrievers, which is why it is the
    usual way to combine BM25 and vector results.
    """
    scores: dict[str, float] = {}
    best: dict[str, ScoredChunk] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            key = item.chunk.id
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            best.setdefault(key, item)
    return [
        best[key].model_copy(update={"score": score, "retriever": "hybrid"})
        for key, score in sorted(scores.items(), key=lambda kv: -kv[1])
    ]


class HybridRetriever:
    name = "hybrid"

    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        *,
        mode: SearchMode = "hybrid",
        candidates: int = 50,
        rrf_k: int = 60,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.mode = mode
        self.candidates = candidates
        self.rrf_k = rrf_k

    async def retrieve(self, query: Query) -> list[ScoredChunk]:
        wide = query.model_copy(update={"top_k": max(self.candidates, query.top_k)})
        searches = []
        if self.mode in ("dense", "hybrid"):
            searches.append(self._dense(wide))
        if self.mode in ("sparse", "hybrid"):
            searches.append(self.store.keyword_search(wide))
        rankings = await asyncio.gather(*searches)
        if len(rankings) == 1:
            return list(rankings[0][: query.top_k])
        return rrf_fuse(rankings, k=self.rrf_k)[: query.top_k]

    async def _dense(self, query: Query) -> list[ScoredChunk]:
        embedding = await self.embedder.embed_query(query.text)
        return await self.store.search(embedding, query)
