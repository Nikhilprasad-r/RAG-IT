"""Plugin interfaces.

Each pipeline stage is a structural `Protocol`: a plugin only needs the right
attributes and methods, not a base class. Register implementations with
`ragkit.core.registry` or expose them through a package entry point.
"""

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable

from ragkit.core.types import Chunk, Document, ParsedContent, Query, RawDocument, ScoredChunk


@runtime_checkable
class Connector(Protocol):
    """Pulls raw items from a source system."""

    name: str
    """Stable identity of this source, e.g. 'files:docs' or 'github:org/repo'."""

    def fetch(self) -> AsyncIterator[RawDocument]:
        """Yield every item currently in the source.

        Ingestion compares content hashes to skip unchanged items and deletes
        items that are no longer yielded, so a connector must yield the full set.
        """
        ...


@runtime_checkable
class Parser(Protocol):
    """Turns raw bytes of the MIME types it supports into Markdown-flavoured text."""

    mime_types: tuple[str, ...]

    async def parse(self, raw: RawDocument) -> ParsedContent: ...


@runtime_checkable
class Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]: ...


@runtime_checkable
class Embedder(Protocol):
    model: str

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents for indexing."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query (some models encode queries differently)."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    async def upsert(self, document: Document, chunks: Sequence[Chunk]) -> None:
        """Store a document and replace all of its previous chunks."""
        ...

    async def search(self, embedding: Sequence[float], query: Query) -> list[ScoredChunk]:
        """Return the nearest chunks the query's principals are allowed to read."""
        ...

    async def delete(self, collection: str, document_ids: Sequence[str]) -> None: ...


@runtime_checkable
class KeywordIndex(Protocol):
    async def keyword_search(self, query: Query) -> list[ScoredChunk]:
        """Return the best lexical (BM25-style) matches the query may read."""
        ...


@runtime_checkable
class Retriever(Protocol):
    name: str

    async def retrieve(self, query: Query) -> list[ScoredChunk]: ...


@runtime_checkable
class Reranker(Protocol):
    name: str

    async def rerank(
        self, query: Query, candidates: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]: ...
