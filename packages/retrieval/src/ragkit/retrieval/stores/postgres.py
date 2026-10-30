"""Postgres store: pgvector for vectors, built-in full-text search for keywords.

All collections share one chunks table. Each collection gets its own partial
HNSW index on `embedding::vector(<dims>)`, created on first write, so
collections can use embedding models of different sizes.
"""

import json
from collections.abc import Sequence
from typing import Any

from ragkit.core.errors import ConfigError
from ragkit.core.types import Chunk, Document, Query, ScoredChunk
from ragkit.retrieval.stores.base import CollectionStats, Store, StoredDocument, check_filters

try:
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool
except ImportError:  # pragma: no cover - exercised only without the extra installed
    AsyncConnectionPool = None  # type: ignore[assignment,misc]

# pgvector's HNSW index supports up to 2,000 dimensions for `vector` and 4,000 for `halfvec`.
_MAX_VECTOR_DIMS = 2000

_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS ragkit_documents (
    id TEXT PRIMARY KEY,
    collection TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    source TEXT NOT NULL,
    uri TEXT NOT NULL,
    title TEXT NOT NULL,
    checksum TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    metadata JSONB NOT NULL,
    acl TEXT[] NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ragkit_documents_by_source ON ragkit_documents (collection, source);

CREATE TABLE IF NOT EXISTS ragkit_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES ragkit_documents (id) ON DELETE CASCADE,
    collection TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    text TEXT NOT NULL,
    context TEXT NOT NULL,
    metadata JSONB NOT NULL,
    acl TEXT[] NOT NULL,
    embedding vector,
    embedding_model TEXT,
    dims INTEGER,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', context || ' ' || text)) STORED
);
CREATE INDEX IF NOT EXISTS ragkit_chunks_by_collection ON ragkit_chunks (collection, tenant_id);
CREATE INDEX IF NOT EXISTS ragkit_chunks_by_document ON ragkit_chunks (document_id);
CREATE INDEX IF NOT EXISTS ragkit_chunks_tsv ON ragkit_chunks USING gin (tsv);
"""

_CHUNK_COLUMNS = (
    "c.id, c.document_id, c.collection, c.tenant_id, c.idx, c.text, c.context, c.metadata, "
    "c.acl, c.embedding_model, d.uri, d.title"
)


class PostgresStore(Store):
    def __init__(self, database_url: str, *, min_size: int = 1, max_size: int = 10) -> None:
        if AsyncConnectionPool is None:
            raise ConfigError("The postgres store needs `ragkit-retrieval[postgres]` installed.")
        self._pool = AsyncConnectionPool(
            database_url,
            min_size=min_size,
            max_size=max_size,
            open=False,
            kwargs={"row_factory": dict_row, "autocommit": True},
        )
        self._indexed: set[tuple[str, int]] = set()
        self._iterative_scan = False

    async def open(self) -> None:
        await self._pool.open(wait=True)
        async with self._pool.connection() as conn:
            await conn.execute(_SCHEMA)
            row = await (
                await conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            ).fetchone()
            # Iterative index scans (pgvector 0.8+) keep filtered HNSW queries from
            # returning fewer rows than requested.
            self._iterative_scan = _version(row["extversion"]) >= (0, 8)

    async def close(self) -> None:
        await self._pool.close()

    async def upsert(self, document: Document, chunks: Sequence[Chunk]) -> None:
        async with self._pool.connection() as conn, conn.transaction():
            await conn.execute("DELETE FROM ragkit_documents WHERE id = %s", (document.id,))
            await conn.execute(
                "INSERT INTO ragkit_documents (id, collection, tenant_id, source, uri, title, "
                "checksum, mime_type, metadata, acl) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, "
                "%s::jsonb, %s)",
                (
                    document.id,
                    document.collection,
                    document.tenant_id,
                    document.source,
                    document.uri,
                    document.title,
                    document.checksum,
                    document.mime_type,
                    json.dumps(document.metadata),
                    sorted(document.acl),
                ),
            )
            async with conn.cursor() as cursor:
                await cursor.executemany(
                    "INSERT INTO ragkit_chunks (id, document_id, collection, tenant_id, idx, text, "
                    "context, metadata, acl, embedding, embedding_model, dims) VALUES (%s, %s, %s, "
                    "%s, %s, %s, %s, %s::jsonb, %s, %s::vector, %s, %s)",
                    [
                        (
                            chunk.id,
                            chunk.document_id,
                            chunk.collection,
                            chunk.tenant_id,
                            chunk.index,
                            chunk.text,
                            chunk.context,
                            json.dumps(chunk.metadata),
                            sorted(chunk.acl),
                            _vector_literal(chunk.embedding),
                            chunk.embedding_model,
                            len(chunk.embedding) if chunk.embedding else None,
                        )
                        for chunk in chunks
                    ],
                )
        dims = next((len(c.embedding) for c in chunks if c.embedding), None)
        if dims:
            await self._ensure_vector_index(document.collection, dims)

    async def delete(self, collection: str, document_ids: Sequence[str]) -> None:
        if not document_ids:
            return
        async with self._pool.connection() as conn:
            await conn.execute(
                "DELETE FROM ragkit_documents WHERE collection = %s AND id = ANY(%s)",
                (collection, list(document_ids)),
            )

    async def documents(self, collection: str, source: str) -> dict[str, StoredDocument]:
        async with self._pool.connection() as conn:
            rows = await (
                await conn.execute(
                    "SELECT id, uri, checksum FROM ragkit_documents "
                    "WHERE collection = %s AND source = %s",
                    (collection, source),
                )
            ).fetchall()
        return {r["uri"]: StoredDocument(r["id"], r["uri"], r["checksum"]) for r in rows}

    async def search(self, embedding: Sequence[float], query: Query) -> list[ScoredChunk]:
        dims = len(embedding)
        cast = _cast(dims)
        where, params = _where(query)
        # The nearest-neighbour ordering runs in a subquery over chunks alone, with the
        # collection and dims predicates inlined, so the planner can use the collection's
        # partial HNSW index; documents are joined only to the top rows.
        sql = (
            f"SELECT {_CHUNK_COLUMNS}, 1 - c.distance AS score FROM ("
            f"SELECT c.*, (c.embedding::{cast}) <=> %s::{cast} AS distance FROM ragkit_chunks c "
            f"WHERE c.collection = {_literal(query.collection)} AND c.dims = {dims} AND {where} "
            f"ORDER BY distance LIMIT %s) c "
            f"JOIN ragkit_documents d ON d.id = c.document_id ORDER BY c.distance"
        )
        vector = _vector_literal(list(embedding))
        async with self._pool.connection() as conn, conn.transaction():
            if self._iterative_scan:
                await conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            rows = await (await conn.execute(sql, (vector, *params, query.top_k))).fetchall()
        return sorted(
            (_scored(row, float(row["score"]), "vector") for row in rows),
            key=lambda s: -s.score,
        )

    async def keyword_search(self, query: Query) -> list[ScoredChunk]:
        where, params = _where(query)
        # plainto_tsquery ANDs the terms; OR them instead so partial matches still rank.
        tsquery = "replace(plainto_tsquery('english', %s)::text, '&', '|')::tsquery"
        sql = (
            f"SELECT {_CHUNK_COLUMNS}, ts_rank_cd(c.tsv, q) AS score "
            f"FROM ragkit_chunks c JOIN ragkit_documents d ON d.id = c.document_id, "
            f"NULLIF({tsquery}, ''::tsquery) AS q "
            f"WHERE c.collection = %s AND c.tsv @@ q AND {where} ORDER BY score DESC LIMIT %s"
        )
        async with self._pool.connection() as conn:
            rows = await (
                await conn.execute(sql, (query.text, query.collection, *params, query.top_k))
            ).fetchall()
        return [_scored(row, float(row["score"]), "keyword") for row in rows]

    async def stats(self) -> list[CollectionStats]:
        async with self._pool.connection() as conn:
            rows = await (
                await conn.execute(
                    "SELECT d.collection, COUNT(DISTINCT d.id) AS documents, "
                    "COUNT(c.id) AS chunks, "
                    "MAX(c.embedding_model) AS model, MAX(c.dims) AS dims "
                    "FROM ragkit_documents d LEFT JOIN ragkit_chunks c ON c.document_id = d.id "
                    "GROUP BY d.collection ORDER BY d.collection"
                )
            ).fetchall()
        return [
            CollectionStats(r["collection"], r["documents"], r["chunks"], r["model"], r["dims"])
            for r in rows
        ]

    async def drop_collection(self, collection: str) -> None:
        async with self._pool.connection() as conn:
            await conn.execute("DELETE FROM ragkit_documents WHERE collection = %s", (collection,))

    async def _ensure_vector_index(self, collection: str, dims: int) -> None:
        if (collection, dims) in self._indexed:
            return
        cast = _cast(dims)
        ops = "halfvec_cosine_ops" if dims > _MAX_VECTOR_DIMS else "vector_cosine_ops"
        name = f"ragkit_hnsw_{collection.replace('-', '_')}_{dims}"
        async with self._pool.connection() as conn:
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS {name} ON ragkit_chunks USING hnsw "
                f"((embedding::{cast}) {ops}) "
                f"WHERE collection = {_literal(collection)} AND dims = {dims}"
            )
        self._indexed.add((collection, dims))


def _where(query: Query) -> tuple[str, list[Any]]:
    clauses = ["c.tenant_id = %s"]
    params: list[Any] = [query.tenant_id]
    # Compare the column itself (not cardinality(acl)) so the planner has statistics;
    # a misestimate here makes it skip the HNSW index.
    if query.principals:
        clauses.append("(c.acl = '{}' OR c.acl && %s::text[])")
        params.append(list(query.principals))
    else:
        clauses.append("c.acl = '{}'")
    if query.filters:
        clauses.append("c.metadata @> %s::jsonb")
        params.append(json.dumps(check_filters(query.filters)))
    return " AND ".join(clauses), params


def _cast(dims: int) -> str:
    return f"halfvec({dims})" if dims > _MAX_VECTOR_DIMS else f"vector({dims})"


def _literal(collection: str) -> str:
    # Collection names are validated to [a-z0-9_-] by the project config. They are
    # inlined so the planner can match each collection's partial HNSW index.
    if not collection.replace("-", "").replace("_", "").isalnum():
        raise ConfigError(f"Invalid collection name {collection!r}.")
    return f"'{collection}'"


def _vector_literal(embedding: list[float] | None) -> str | None:
    return None if embedding is None else "[" + ",".join(f"{v:.7g}" for v in embedding) + "]"


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(".")[:2])


def _scored(row: dict[str, Any], score: float, retriever: str) -> ScoredChunk:
    chunk = Chunk(
        id=row["id"],
        document_id=row["document_id"],
        collection=row["collection"],
        tenant_id=row["tenant_id"],
        index=row["idx"],
        text=row["text"],
        context=row["context"],
        metadata=row["metadata"],
        acl=row["acl"],
        embedding_model=row["embedding_model"],
    )
    return ScoredChunk(
        chunk=chunk, score=score, retriever=retriever, uri=row["uri"], title=row["title"]
    )
