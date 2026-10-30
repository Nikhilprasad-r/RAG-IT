"""Store contract tests. Postgres runs too when RAGKIT_TEST_DATABASE_URL is set."""

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from ragkit.core.types import Chunk, Document, Query
from ragkit.retrieval import HashEmbedder, SQLiteStore, Store

POSTGRES_URL = os.environ.get("RAGKIT_TEST_DATABASE_URL")
EMBEDDER = HashEmbedder(64)


@pytest.fixture(params=["sqlite", "postgres"])
async def store(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[Store]:
    if request.param == "sqlite":
        backend: Store = SQLiteStore(tmp_path / "index.db")
    else:
        if not POSTGRES_URL:
            pytest.skip("RAGKIT_TEST_DATABASE_URL is not set")
        from ragkit.retrieval.stores.postgres import PostgresStore

        backend = PostgresStore(POSTGRES_URL)
    async with backend:
        if request.param == "postgres":
            await _reset_postgres(backend)
        yield backend


async def _reset_postgres(store: Store) -> None:
    async with store._pool.connection() as conn:  # type: ignore[attr-defined]
        await conn.execute("TRUNCATE ragkit_documents CASCADE")


async def _add(
    store: Store,
    doc_id: str,
    texts: list[str],
    *,
    collection: str = "kb",
    acl: list[str] | None = None,
    metadata: dict | None = None,
    source: str = "src",
) -> None:
    document = Document(
        id=doc_id,
        collection=collection,
        source=source,
        uri=f"{doc_id}.md",
        content="\n\n".join(texts),
        title=doc_id.title(),
        checksum=f"sum-{doc_id}-{len(texts)}",
        acl=acl or [],
    )
    vectors = await EMBEDDER.embed(texts)
    chunks = [
        Chunk(
            id=f"{doc_id}:{i}",
            document_id=doc_id,
            collection=collection,
            text=text,
            context=document.title,
            index=i,
            acl=acl or [],
            metadata=metadata or {},
            embedding=vector,
            embedding_model=EMBEDDER.model,
        )
        for i, (text, vector) in enumerate(zip(texts, vectors, strict=True))
    ]
    await store.upsert(document, chunks)


async def _vector(store: Store, text: str, **query: object) -> list[str]:
    q = Query(text=text, collection="kb", **query)
    return [r.chunk.id for r in await store.search(await EMBEDDER.embed_query(text), q)]


async def _keyword(store: Store, text: str, **query: object) -> list[str]:
    return [
        r.chunk.id for r in await store.keyword_search(Query(text=text, collection="kb", **query))
    ]


async def test_vector_and_keyword_search_find_the_right_chunk(store: Store) -> None:
    await _add(store, "billing", ["Invoices are sent monthly by email", "Refunds take five days"])
    await _add(store, "security", ["Passwords expire after ninety days"])

    assert (await _vector(store, "refunds take days"))[0] == "billing:1"
    assert (await _keyword(store, "password expiry"))[0] == "security:0"
    results = await store.keyword_search(Query(text="refunds", collection="kb"))
    assert results[0].uri == "billing.md"
    assert results[0].title == "Billing"
    assert results[0].retriever == "keyword"


async def test_upsert_replaces_previous_chunks(store: Store) -> None:
    await _add(store, "doc", ["old alpha text", "old beta text"])
    await _add(store, "doc", ["new gamma text"])

    assert await _keyword(store, "alpha beta") == []
    assert await _keyword(store, "gamma") == ["doc:0"]
    (stats,) = await store.stats()
    assert (stats.documents, stats.chunks, stats.embedding_model) == (1, 1, "hash:64")


async def test_acl_and_tenant_filters(store: Store) -> None:
    await _add(store, "public", ["roadmap overview"])
    await _add(store, "secret", ["roadmap for finance only"], acl=["finance"])

    assert await _keyword(store, "roadmap") == ["public:0"]
    assert sorted(await _keyword(store, "roadmap", principals=["finance"])) == [
        "public:0",
        "secret:0",
    ]
    assert sorted(await _vector(store, "roadmap", principals=["finance", "eng"])) == [
        "public:0",
        "secret:0",
    ]
    assert await _vector(store, "roadmap", tenant_id="other") == []


async def test_metadata_filters(store: Store) -> None:
    await _add(store, "en", ["install guide"], metadata={"lang": "en", "version": 2})
    await _add(store, "de", ["install guide"], metadata={"lang": "de", "version": 2})

    assert await _keyword(store, "install", filters={"lang": "de"}) == ["de:0"]
    assert await _vector(store, "install", filters={"lang": "en", "version": 2}) == ["en:0"]


async def test_documents_delete_and_drop(store: Store) -> None:
    await _add(store, "a", ["first"], source="s1")
    await _add(store, "b", ["second"], source="s1")
    await _add(store, "c", ["third"], source="s2")
    await _add(store, "z", ["other collection"], collection="other", source="s1")

    docs = await store.documents("kb", "s1")
    assert {uri: d.id for uri, d in docs.items()} == {"a.md": "a", "b.md": "b"}
    assert docs["a.md"].checksum == "sum-a-1"

    await store.delete("kb", ["a", "z"])  # "z" is in another collection and must survive
    assert set(await store.documents("kb", "s1")) == {"b.md"}
    assert set(await store.documents("other", "s1")) == {"z.md"}

    await store.drop_collection("kb")
    assert [s.collection for s in await store.stats()] == ["other"]
