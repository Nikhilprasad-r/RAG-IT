from pathlib import Path

import pytest

from conftest import write_project
from ragkit.core.config import ProjectConfig
from ragkit.core.errors import ConfigError
from ragkit.core.types import RawDocument
from ragkit.ingest import IngestPipeline, ingest
from ragkit.retrieval import HashEmbedder, KnowledgeBase, SQLiteStore


async def _run(project: ProjectConfig, *, full: bool = False, embedder=None):
    (report,) = await ingest(project, ["docs"], full=full, embedder=embedder)
    return report


async def test_first_run_adds_every_document(rag_project: ProjectConfig) -> None:
    report = await _run(rag_project)

    assert (report.added, report.updated, report.unchanged, report.deleted) == (3, 0, 0, 0)
    assert report.chunks >= 5
    assert report.ok


async def test_reruns_only_process_changes(rag_project: ProjectConfig) -> None:
    await _run(rag_project)
    data = rag_project.root / "data"

    assert (await _run(rag_project)).unchanged == 3

    (data / "api.md").write_text("# API limits\n\nNow 1200 requests per minute.\n")
    (data / "account" / "password.md").unlink()
    (data / "new.md").write_text("# New page\n\nFresh content.\n")
    report = await _run(rag_project)

    assert (report.added, report.updated, report.unchanged, report.deleted) == (1, 1, 1, 1)

    full = await _run(rag_project, full=True)
    assert (full.updated, full.unchanged) == (3, 0)


async def test_unsupported_and_empty_files_are_skipped(rag_project: ProjectConfig) -> None:
    data = rag_project.root / "data"
    (data / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00")
    (data / "empty.md").write_text("\n\n")

    report = await _run(rag_project)

    assert {uri for uri, _ in report.skipped} == {"logo.png", "empty.md"}
    assert report.added == 3
    assert report.ok


async def test_changing_the_embedding_model_requires_full(rag_project: ProjectConfig) -> None:
    await _run(rag_project)

    with pytest.raises(ConfigError, match="--full"):
        await _run(rag_project, embedder=HashEmbedder(128))

    report = await _run(rag_project, full=True, embedder=HashEmbedder(128))
    assert report.added == 3


async def test_failed_listing_does_not_delete_documents(
    rag_project: ProjectConfig, tmp_path: Path
) -> None:
    await _run(rag_project)

    class BrokenSource:
        name = "files:data"

        async def fetch(self):
            yield RawDocument(source=self.name, uri="plans.md", data=b"# x", mime_type="text/plain")
            raise ConnectionError("source went away")

    async with SQLiteStore(rag_project.root / rag_project.store.path) as store:
        pipeline = IngestPipeline(rag_project, store, HashEmbedder(256))
        report = await _sync_with(pipeline, BrokenSource())
        docs = await store.documents("docs", "files:data")

    assert not report.ok
    assert "source went away" in report.failed[0][1]
    assert len(docs) == 3


async def _sync_with(pipeline: IngestPipeline, connector):
    from ragkit.ingest.chunking import MarkdownChunker
    from ragkit.ingest.parsers import Parsers
    from ragkit.ingest.pipeline import IngestReport

    report = IngestReport("docs")
    await pipeline._sync_source("docs", connector, Parsers(), MarkdownChunker(), report, full=False)
    return report


async def test_embedding_failures_are_retried_then_reported(
    rag_project: ProjectConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("ragkit.ingest.pipeline.asyncio.sleep", _no_sleep)
    calls = {"n": 0}

    class Flaky(HashEmbedder):
        async def embed(self, texts):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("first call times out")
            if "429" in " ".join(texts):
                raise RuntimeError("always fails for the API page")
            return await super().embed(texts)

    report = await _run(rag_project, embedder=Flaky(256))

    assert report.added == 2
    assert [uri for uri, _ in report.failed] == ["api.md"]


async def _no_sleep(_seconds: float) -> None:
    return None


async def test_collection_acl_restricts_search(tmp_path: Path) -> None:
    (tmp_path / "hr").mkdir()
    (tmp_path / "hr" / "salaries.md").write_text("# Salaries\n\nBands are reviewed yearly.\n")
    project = write_project(
        tmp_path,
        "name: t\nembeddings: {model: 'hash:64'}\ncollections:\n  hr:\n    acl: [hr-team]\n"
        "    sources: [{type: files, path: hr}]\n",
    )
    await ingest(project)

    async with KnowledgeBase(project) as kb:
        assert await kb.search("hr", "salary bands") == []
        assert len(await kb.search("hr", "salary bands", principals=["hr-team"])) == 1
