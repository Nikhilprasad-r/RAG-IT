"""Data model that flows through ingestion and retrieval.

A connector yields `RawDocument`s, a parser turns each into a `Document`, a
chunker splits it into `Chunk`s, and retrievers return `ScoredChunk`s for a
`Query`. Every record carries its collection, tenant and ACL so access control
can be enforced at query time, not just at ingestion.
"""

import hashlib
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Modality(StrEnum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    CODE = "code"


def content_hash(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


class RawDocument(BaseModel):
    """Unparsed bytes as fetched by a connector."""

    source: str = Field(description="Name of the connector instance that fetched it.")
    uri: str = Field(description="Location of the item within its source.")
    data: bytes
    mime_type: str = "application/octet-stream"
    metadata: dict[str, Any] = Field(default_factory=dict)
    acl: list[str] = Field(default_factory=list)
    tenant_id: str = "default"


class ParsedContent(BaseModel):
    """What a parser extracts from a `RawDocument`."""

    text: str = Field(description="Markdown-flavoured text; headings start with '#'.")
    title: str = ""
    modality: Modality = Modality.TEXT
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Parser facts such as 'page_offsets' (character offset of each page).",
    )


class Document(BaseModel):
    """A parsed, normalized item ready for chunking.

    `content` is Markdown-flavoured text: parsers mark headings with `#` so the
    chunker can split along the document's structure.
    """

    id: str
    collection: str = "default"
    source: str
    uri: str
    content: str
    mime_type: str = "text/plain"
    modality: Modality = Modality.TEXT
    metadata: dict[str, Any] = Field(default_factory=dict)
    acl: list[str] = Field(
        default_factory=list,
        description="Principals allowed to read; empty means everyone in the tenant.",
    )
    tenant_id: str = "default"
    title: str = ""
    checksum: str = Field(
        default="", description="Hash of the raw source bytes; unchanged items are skipped."
    )
    version: int = 1

    @property
    def content_hash(self) -> str:
        return content_hash(self.content)


class Chunk(BaseModel):
    id: str
    document_id: str
    collection: str = "default"
    text: str
    context: str = Field(
        default="",
        description="Short header (title, section path) prepended when indexing the chunk.",
    )
    index: int = Field(description="Position of the chunk within its document.")
    metadata: dict[str, Any] = Field(default_factory=dict)
    acl: list[str] = Field(default_factory=list)
    tenant_id: str = "default"
    embedding: list[float] | None = None
    embedding_model: str | None = Field(
        default=None,
        description="Model that produced `embedding`, so indexes can be rebuilt per model.",
    )

    @property
    def indexed_text(self) -> str:
        """Text that is embedded and keyword-indexed: the context header plus the body."""
        return f"{self.context}\n\n{self.text}" if self.context else self.text


class Query(BaseModel):
    text: str
    collection: str = "default"
    tenant_id: str = "default"
    principals: list[str] = Field(
        default_factory=list, description="Identities of the caller, matched against ACLs."
    )
    filters: dict[str, Any] = Field(
        default_factory=dict, description="Equality filters on chunk metadata."
    )
    top_k: int = Field(default=20, ge=1)


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    retriever: str = Field(description="Name of the retriever or reranker that scored it.")
    uri: str = ""
    title: str = ""
