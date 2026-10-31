"""In-process caches.

- Query embeddings (LRU): repeated questions skip an embedding API call.
  Document embeddings are never cached; ingestion skips unchanged items instead.
- Search results (TTL): identical searches within `search_ttl_seconds` reuse
  the earlier results. Off by default, because results can be stale for that
  long after an ingest. Keys include the tenant and principals, so callers never
  see results they could not read.

Each server process has its own caches.
"""

import time
from collections import OrderedDict
from collections.abc import Hashable, Sequence
from typing import Any

from ragkit.retrieval.embedders import Embedder


class TTLCache:
    def __init__(self, max_entries: int, ttl_seconds: float) -> None:
        self.max_entries = max_entries
        self.ttl = ttl_seconds
        self._items: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()

    def get(self, key: Hashable) -> Any:
        if self.ttl <= 0 or (entry := self._items.get(key)) is None:
            return None
        expires, value = entry
        if expires < time.monotonic():
            del self._items[key]
            return None
        self._items.move_to_end(key)
        return value

    def put(self, key: Hashable, value: Any) -> None:
        if self.ttl <= 0:
            return
        self._items[key] = (time.monotonic() + self.ttl, value)
        self._items.move_to_end(key)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()


class CachedQueryEmbedder:
    """Wraps an embedder; `embed_query` results are kept in an LRU cache."""

    def __init__(self, inner: Embedder, max_entries: int) -> None:
        self.inner = inner
        self.model = inner.model
        self.max_entries = max_entries
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return await self.inner.embed(texts)

    async def embed_query(self, text: str) -> list[float]:
        if (vector := self._cache.get(text)) is not None:
            self._cache.move_to_end(text)
            self.hits += 1
            return vector
        self.misses += 1
        vector = await self.inner.embed_query(text)
        self._cache[text] = vector
        if len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return vector


def cache_query_embeddings(embedder: Embedder, max_entries: int) -> Any:
    return CachedQueryEmbedder(embedder, max_entries) if max_entries > 0 else embedder
