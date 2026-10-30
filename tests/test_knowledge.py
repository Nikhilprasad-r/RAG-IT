import httpx
import pytest

from ragkit.core.config import ProjectConfig, RerankConfig
from ragkit.core.errors import ConfigError, PluginNotFoundError
from ragkit.core.registry import registry
from ragkit.core.types import Chunk, Query, ScoredChunk
from ragkit.ingest import ingest
from ragkit.retrieval import (
    CohereReranker,
    HashEmbedder,
    KnowledgeBase,
    VoyageReranker,
    format_results,
    rrf_fuse,
)


def _hit(chunk_id: str, score: float = 1.0, text: str = "") -> ScoredChunk:
    chunk = Chunk(id=chunk_id, document_id="d", text=text or chunk_id, index=0)
    return ScoredChunk(chunk=chunk, score=score, retriever="x", uri=f"{chunk_id}.md")


def test_rrf_rewards_chunks_found_by_both_retrievers() -> None:
    dense = [_hit("a"), _hit("b"), _hit("c")]
    sparse = [_hit("c"), _hit("d"), _hit("a")]

    fused = rrf_fuse([dense, sparse])

    assert [h.chunk.id for h in fused][:2] == ["a", "c"]
    assert {h.chunk.id for h in fused} == {"a", "b", "c", "d"}
    assert all(h.retriever == "hybrid" for h in fused)


async def test_hybrid_search_finds_the_answer(rag_project: ProjectConfig) -> None:
    await ingest(rag_project)

    async with KnowledgeBase(rag_project) as kb:
        results = await kb.search("docs", "how long is the password reset link valid", top_k=3)
        sparse = await kb.search("docs", "Retry-After", mode="sparse", top_k=1)

    assert results[0].uri == "account/password.md"
    assert "30 minutes" in results[0].chunk.text
    assert results[0].retriever == "hybrid"
    assert sparse[0].uri == "api.md"


async def test_search_many_merges_collections(rag_project: ProjectConfig) -> None:
    await ingest(rag_project)

    async with KnowledgeBase(rag_project) as kb:
        merged = await kb.search_many(["docs", "docs"], "Pro plan storage", top_k=2)

    assert merged[0].chunk.metadata["heading_path"] == ["Nimbus plans", "Pro plan"]
    assert len(merged) == 2


async def test_reranker_from_registry_reorders_results(rag_project: ProjectConfig) -> None:
    class ReverseReranker:
        name = "reverse"

        async def rerank(self, query: Query, candidates: list[ScoredChunk], top_n: int):
            return [c.model_copy(update={"retriever": "reverse"}) for c in candidates[::-1]][:top_n]

    registry.register("reranker", "reverse-test", lambda config: ReverseReranker())
    await ingest(rag_project)

    async with KnowledgeBase(rag_project) as kb:
        plain = await kb.search("docs", "storage", top_k=20, candidates=20)
        reranked = await kb.search(
            "docs", "storage", top_k=2, candidates=20, rerank=RerankConfig(provider="reverse-test")
        )

    assert [r.chunk.id for r in reranked] == [r.chunk.id for r in plain[::-1][:2]]
    assert {r.retriever for r in reranked} == {"reverse"}


async def test_unknown_reranker_and_collection_are_errors(rag_project: ProjectConfig) -> None:
    await ingest(rag_project)

    async with KnowledgeBase(rag_project) as kb:
        with pytest.raises(PluginNotFoundError, match="cohere"):
            await kb.search("docs", "x", rerank=RerankConfig(provider="nope"))
        with pytest.raises(ConfigError, match="No collection named 'nope'"):
            await kb.search("nope", "x")


async def test_search_detects_embedding_model_mismatch(rag_project: ProjectConfig) -> None:
    await ingest(rag_project)

    async with KnowledgeBase(rag_project, embedder=HashEmbedder(64)) as kb:
        with pytest.raises(ConfigError, match="ingest docs --full"):
            await kb.search("docs", "storage")
        assert await kb.search("docs", "storage", mode="sparse")


def test_format_results_numbers_and_cites_passages() -> None:
    hit = _hit("a", text="The Pro plan includes 2 TB.")
    hit.chunk.metadata.update(heading_path=["Plans", "Pro"], page=3)
    hit.title = "Plans"

    text = format_results([hit])

    assert text.startswith("[1] Plans, page 3 — Plans > Pro\nsource: a.md\n")
    assert format_results([]) == "No matching passages found."


@pytest.mark.parametrize(
    ("cls", "env", "body"),
    [
        (
            CohereReranker,
            "COHERE_API_KEY",
            {
                "results": [
                    {"index": 2, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.4},
                ]
            },
        ),
        (
            VoyageReranker,
            "VOYAGE_API_KEY",
            {"data": [{"index": 0, "relevance_score": 0.4}, {"index": 2, "relevance_score": 0.9}]},
        ),
    ],
)
async def test_hosted_rerankers(
    cls: type, env: str, body: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(env, "test-key")
    sent: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["auth"] = request.headers["authorization"]
        sent["json"] = request.read()
        return httpx.Response(200, json=body)

    reranker = cls(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    candidates = [_hit("a"), _hit("b"), _hit("c")]

    ranked = await reranker.rerank(Query(text="q"), candidates, top_n=2)

    assert [r.chunk.id for r in ranked] == ["c", "a"]
    assert ranked[0].score == 0.9
    assert sent["auth"] == "Bearer test-key"
    assert b'"query":"q"' in sent["json"]


def test_hosted_reranker_needs_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COHERE_API_KEY", raising=False)

    with pytest.raises(ConfigError, match="COHERE_API_KEY"):
        CohereReranker()
