"""Single-file local store: SQLite with FTS5 for keywords and NumPy for vectors.

Vector search scans every chunk in the collection, which is fast up to roughly
a hundred thousand chunks. Use the Postgres store beyond that, or when several
servers share one index.
"""

import asyncio
import json
import re
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from ragkit.core.types import Chunk, Document, Query, ScoredChunk
from ragkit.retrieval.stores.base import CollectionStats, Store, StoredDocument, check_filters

T = TypeVar("T")

_WORD = re.compile(r"\w+")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    collection TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    source TEXT NOT NULL,
    uri TEXT NOT NULL,
    title TEXT NOT NULL,
    checksum TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    metadata TEXT NOT NULL,
    acl TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS documents_by_source ON documents (collection, source);

CREATE TABLE IF NOT EXISTS chunks (
    rowid INTEGER PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    document_id TEXT NOT NULL,
    collection TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    text TEXT NOT NULL,
    context TEXT NOT NULL,
    metadata TEXT NOT NULL,
    acl TEXT NOT NULL,
    embedding BLOB,
    embedding_model TEXT,
    dims INTEGER
);
CREATE INDEX IF NOT EXISTS chunks_by_collection ON chunks (collection, tenant_id);
CREATE INDEX IF NOT EXISTS chunks_by_document ON chunks (document_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5 (body, tokenize = 'porter unicode61');
"""

_CHUNK_COLUMNS = (
    "c.id, c.document_id, c.collection, c.tenant_id, c.idx, c.text, c.context, c.metadata, "
    "c.acl, c.embedding_model, d.uri, d.title"
)


class SQLiteStore(Store):
    def __init__(self, path: Path) -> None:
        self.path = path
        self._db: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode = WAL")
        db.execute("PRAGMA busy_timeout = 5000")
        db.executescript(_SCHEMA)
        self._db = db

    async def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    async def upsert(self, document: Document, chunks: Sequence[Chunk]) -> None:
        def write(db: sqlite3.Connection) -> None:
            with _transaction(db):
                _delete_documents(db, [document.id])
                db.execute(
                    "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                        json.dumps(sorted(document.acl)),
                        datetime.now(UTC).isoformat(),
                    ),
                )
                for chunk in chunks:
                    cursor = db.execute(
                        "INSERT INTO chunks (id, document_id, collection, tenant_id, idx, text, "
                        "context, metadata, acl, embedding, embedding_model, dims) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            chunk.id,
                            chunk.document_id,
                            chunk.collection,
                            chunk.tenant_id,
                            chunk.index,
                            chunk.text,
                            chunk.context,
                            json.dumps(chunk.metadata),
                            json.dumps(sorted(chunk.acl)),
                            _to_blob(chunk.embedding),
                            chunk.embedding_model,
                            len(chunk.embedding) if chunk.embedding else None,
                        ),
                    )
                    db.execute(
                        "INSERT INTO chunks_fts (rowid, body) VALUES (?, ?)",
                        (cursor.lastrowid, chunk.indexed_text),
                    )

        await self._run(write)

    async def delete(self, collection: str, document_ids: Sequence[str]) -> None:
        def write(db: sqlite3.Connection) -> None:
            with _transaction(db):
                ids = [
                    row["id"]
                    for row in db.execute(
                        f"SELECT id FROM documents WHERE collection = ? AND id IN "
                        f"({', '.join('?' for _ in document_ids)})",
                        (collection, *document_ids),
                    )
                ]
                _delete_documents(db, ids)

        if document_ids:
            await self._run(write)

    async def documents(self, collection: str, source: str) -> dict[str, StoredDocument]:
        rows = await self._run(
            lambda db: db.execute(
                "SELECT id, uri, checksum FROM documents WHERE collection = ? AND source = ?",
                (collection, source),
            ).fetchall()
        )
        return {row["uri"]: StoredDocument(row["id"], row["uri"], row["checksum"]) for row in rows}

    async def search(self, embedding: Sequence[float], query: Query) -> list[ScoredChunk]:
        where, params = _where(query)

        def read(db: sqlite3.Connection) -> list[ScoredChunk]:
            rows = db.execute(
                f"SELECT c.rowid, c.embedding FROM chunks c WHERE c.embedding IS NOT NULL "
                f"AND {where}",
                params,
            ).fetchall()
            if not rows:
                return []
            matrix = np.vstack([np.frombuffer(row["embedding"], dtype=np.float32) for row in rows])
            vector = np.asarray(embedding, dtype=np.float32)
            if matrix.shape[1] != vector.shape[0]:
                raise ValueError(
                    f"Query embedding has {vector.shape[0]} dimensions but the collection "
                    f"has {matrix.shape[1]}. Re-ingest with --full after changing models."
                )
            norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(vector) or 1.0)
            scores = matrix @ vector / np.where(norms == 0, 1.0, norms)
            top = np.argsort(-scores)[: query.top_k]
            rowids = [rows[i]["rowid"] for i in top]
            by_rowid = _fetch_chunks(db, rowids)
            return [
                _scored(by_rowid[rows[i]["rowid"]], float(scores[i]), "vector")
                for i in top
                if rows[i]["rowid"] in by_rowid
            ]

        return await self._run(read)

    async def keyword_search(self, query: Query) -> list[ScoredChunk]:
        terms = sorted({word.lower() for word in _WORD.findall(query.text)})
        if not terms:
            return []
        match = " OR ".join(f'"{term}"' for term in terms)
        where, params = _where(query)

        def read(db: sqlite3.Connection) -> list[ScoredChunk]:
            rows = db.execute(
                f"SELECT {_CHUNK_COLUMNS}, bm25(chunks_fts) AS rank FROM chunks_fts "
                f"JOIN chunks c ON c.rowid = chunks_fts.rowid "
                f"JOIN documents d ON d.id = c.document_id "
                f"WHERE chunks_fts MATCH ? AND {where} ORDER BY rank LIMIT ?",
                (match, *params, query.top_k),
            ).fetchall()
            return [_scored(row, -float(row["rank"]), "keyword") for row in rows]

        return await self._run(read)

    async def stats(self) -> list[CollectionStats]:
        def read(db: sqlite3.Connection) -> list[CollectionStats]:
            rows = db.execute(
                "SELECT d.collection, COUNT(DISTINCT d.id) AS documents, COUNT(c.id) AS chunks, "
                "MAX(c.embedding_model) AS model, MAX(c.dims) AS dims "
                "FROM documents d LEFT JOIN chunks c ON c.document_id = d.id "
                "GROUP BY d.collection ORDER BY d.collection"
            ).fetchall()
            return [
                CollectionStats(r["collection"], r["documents"], r["chunks"], r["model"], r["dims"])
                for r in rows
            ]

        return await self._run(read)

    async def drop_collection(self, collection: str) -> None:
        def write(db: sqlite3.Connection) -> None:
            with _transaction(db):
                ids = [
                    row["id"]
                    for row in db.execute(
                        "SELECT id FROM documents WHERE collection = ?", (collection,)
                    )
                ]
                _delete_documents(db, ids)

        await self._run(write)

    async def _run(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        if self._db is None:
            raise RuntimeError("SQLiteStore is not open.")
        db = self._db
        async with self._lock:
            return await asyncio.to_thread(fn, db)


@contextmanager
def _transaction(db: sqlite3.Connection) -> Iterator[None]:
    db.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        db.execute("ROLLBACK")
        raise
    db.execute("COMMIT")


def _delete_documents(db: sqlite3.Connection, document_ids: Sequence[str]) -> None:
    for document_id in document_ids:
        db.execute(
            "DELETE FROM chunks_fts "
            "WHERE rowid IN (SELECT rowid FROM chunks WHERE document_id = ?)",
            (document_id,),
        )
        db.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        db.execute("DELETE FROM documents WHERE id = ?", (document_id,))


def _where(query: Query) -> tuple[str, list[Any]]:
    clauses = ["c.collection = ?", "c.tenant_id = ?"]
    params: list[Any] = [query.collection, query.tenant_id]
    if query.principals:
        marks = ", ".join("?" for _ in query.principals)
        clauses.append(
            f"(c.acl = '[]' OR EXISTS (SELECT 1 FROM json_each(c.acl) WHERE value IN ({marks})))"
        )
        params.extend(query.principals)
    else:
        clauses.append("c.acl = '[]'")
    for key, value in check_filters(query.filters).items():
        clauses.append(f"json_extract(c.metadata, '$.{key}') = ?")
        params.append(value)
    return " AND ".join(clauses), params


def _fetch_chunks(db: sqlite3.Connection, rowids: Sequence[int]) -> dict[int, sqlite3.Row]:
    if not rowids:
        return {}
    marks = ", ".join("?" for _ in rowids)
    rows = db.execute(
        f"SELECT c.rowid AS rid, {_CHUNK_COLUMNS} FROM chunks c "
        f"JOIN documents d ON d.id = c.document_id WHERE c.rowid IN ({marks})",
        list(rowids),
    ).fetchall()
    return {row["rid"]: row for row in rows}


def _scored(row: sqlite3.Row, score: float, retriever: str) -> ScoredChunk:
    chunk = Chunk(
        id=row["id"],
        document_id=row["document_id"],
        collection=row["collection"],
        tenant_id=row["tenant_id"],
        index=row["idx"],
        text=row["text"],
        context=row["context"],
        metadata=json.loads(row["metadata"]),
        acl=json.loads(row["acl"]),
        embedding_model=row["embedding_model"],
    )
    return ScoredChunk(
        chunk=chunk, score=score, retriever=retriever, uri=row["uri"], title=row["title"]
    )


def _to_blob(embedding: list[float] | None) -> bytes | None:
    return None if embedding is None else np.asarray(embedding, dtype=np.float32).tobytes()
