import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self

from ragkit.core.errors import ConfigError
from ragkit.core.types import Chunk, Document, Query, ScoredChunk

_FILTER_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class StoredDocument:
    id: str
    uri: str
    checksum: str


@dataclass(frozen=True)
class CollectionStats:
    collection: str
    documents: int
    chunks: int
    embedding_model: str | None
    dimensions: int | None


class Store(ABC):
    """A document store that is both a vector store and a keyword index.

    Implementations keep documents, their chunks, chunk embeddings and a lexical
    index in one place, so a document and its chunks are replaced atomically.
    """

    async def __aenter__(self) -> Self:
        await self.open()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    @abstractmethod
    async def open(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def upsert(self, document: Document, chunks: Sequence[Chunk]) -> None:
        """Store `document` and replace all of its previous chunks."""

    @abstractmethod
    async def delete(self, collection: str, document_ids: Sequence[str]) -> None: ...

    @abstractmethod
    async def documents(self, collection: str, source: str) -> dict[str, StoredDocument]:
        """Documents previously ingested from `source`, keyed by URI."""

    @abstractmethod
    async def search(self, embedding: Sequence[float], query: Query) -> list[ScoredChunk]:
        """Nearest chunks by cosine similarity, after tenant, ACL and metadata filters."""

    @abstractmethod
    async def keyword_search(self, query: Query) -> list[ScoredChunk]:
        """Best lexical matches, after tenant, ACL and metadata filters."""

    @abstractmethod
    async def stats(self) -> list[CollectionStats]: ...

    @abstractmethod
    async def drop_collection(self, collection: str) -> None: ...

    async def collection_stats(self, collection: str) -> CollectionStats | None:
        return next((s for s in await self.stats() if s.collection == collection), None)


def check_filters(filters: dict[str, Any]) -> dict[str, Any]:
    for key, value in filters.items():
        if not _FILTER_KEY.match(key):
            raise ConfigError(f"Invalid metadata filter key {key!r}.")
        if not isinstance(value, str | int | float | bool):
            raise ConfigError(f"Metadata filter {key!r} must be a string, number or boolean.")
    return filters
