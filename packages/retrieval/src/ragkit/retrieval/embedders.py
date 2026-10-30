"""Embedding models.

`create_embedder` accepts `provider:model` strings. Real providers go through
LangChain's `init_embeddings` (google_genai, openai, cohere, ollama, bedrock,
...). `hash:<dims>` is a dependency-free, offline embedder for tests and demos;
it captures word overlap only, not meaning.
"""

import hashlib
import math
import re
from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import Any

from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings

from ragkit.core import telemetry as t
from ragkit.core.config import EmbeddingsConfig
from ragkit.core.errors import ConfigError

_WORD = re.compile(r"\w+")

# Constructor argument each provider uses for shortened (Matryoshka) output.
_DIMENSION_ARGS = {
    "google_genai": "output_dimensionality",
    "google_vertexai": "output_dimensionality",
    "openai": "dimensions",
    "azure_openai": "dimensions",
}


class LangChainEmbedder:
    def __init__(self, model: str, *, dimensions: int | None, batch_size: int) -> None:
        self.model = model
        self._batch_size = batch_size
        provider = model.split(":", 1)[0]
        kwargs: dict[str, Any] = {}
        if dimensions and provider in _DIMENSION_ARGS:
            kwargs[_DIMENSION_ARGS[provider]] = dimensions
        try:
            self._embeddings: Embeddings = init_embeddings(model, **kwargs)
        except (ImportError, ValueError) as exc:
            raise ConfigError(f"Cannot load embedding model {model!r}: {exc}") from exc

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        with self._span(len(texts)):
            for start in range(0, len(texts), self._batch_size):
                batch = list(texts[start : start + self._batch_size])
                vectors.extend(await self._embeddings.aembed_documents(batch))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        with self._span(1):
            return await self._embeddings.aembed_query(text)

    def _span(self, count: int) -> AbstractContextManager[Any]:
        provider, _, name = self.model.partition(":")
        return t.span(
            f"embeddings {name}",
            **{
                t.OPERATION: "embeddings",
                t.PROVIDER: provider,
                t.REQUEST_MODEL: name,
                "ragkit.embeddings.inputs": count,
            },
        )


class HashEmbedder:
    """Bag-of-words feature hashing. Offline and deterministic; for tests and demos."""

    def __init__(self, dimensions: int = 256) -> None:
        self.model = f"hash:{dimensions}"
        self.dimensions = dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for word in _WORD.findall(text.lower()):
            digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimensions
            vector[bucket] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


Embedder = LangChainEmbedder | HashEmbedder


def create_embedder(config: EmbeddingsConfig) -> Embedder:
    provider, _, name = config.model.partition(":")
    if not name:
        raise ConfigError(f"Embedding model {config.model!r} must look like 'provider:model'.")
    if provider == "hash":
        try:
            return HashEmbedder(int(name))
        except ValueError:
            raise ConfigError("The hash embedder is configured as 'hash:<dimensions>'.") from None
    return LangChainEmbedder(
        config.model, dimensions=config.dimensions, batch_size=config.batch_size
    )
