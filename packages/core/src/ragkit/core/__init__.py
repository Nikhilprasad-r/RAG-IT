"""Core building blocks shared by every ragkit package."""

from ragkit.core.config import (
    ChunkingConfig,
    CollectionConfig,
    EmbeddingsConfig,
    ProjectConfig,
    RerankConfig,
    ServerConfig,
    SourceConfig,
    StoreConfig,
    find_project_root,
)
from ragkit.core.errors import (
    ConfigError,
    PluginNotFoundError,
    ProjectNotFoundError,
    RagkitError,
)
from ragkit.core.interfaces import (
    Chunker,
    Connector,
    Embedder,
    KeywordIndex,
    Parser,
    Reranker,
    Retriever,
    VectorStore,
)
from ragkit.core.registry import PluginKind, Registry, registry
from ragkit.core.types import (
    Chunk,
    Document,
    Modality,
    ParsedContent,
    Query,
    RawDocument,
    ScoredChunk,
)

__all__ = [
    "Chunk",
    "Chunker",
    "ChunkingConfig",
    "CollectionConfig",
    "ConfigError",
    "Connector",
    "Document",
    "Embedder",
    "EmbeddingsConfig",
    "KeywordIndex",
    "Modality",
    "Parser",
    "ParsedContent",
    "PluginKind",
    "PluginNotFoundError",
    "ProjectConfig",
    "ProjectNotFoundError",
    "Query",
    "RagkitError",
    "RawDocument",
    "Registry",
    "RerankConfig",
    "Reranker",
    "Retriever",
    "ScoredChunk",
    "ServerConfig",
    "SourceConfig",
    "StoreConfig",
    "VectorStore",
    "find_project_root",
    "registry",
]
