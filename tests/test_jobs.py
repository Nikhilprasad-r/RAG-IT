import asyncio
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from conftest import RAG_CONFIG, write_project
from ragkit.cli.main import app as cli
from ragkit.core.config import ProjectConfig
from ragkit.ingest import Worker, create_queue
from ragkit.ingest.jobs import JobQueue, PostgresJobQueue, SQLiteJobQueue
from ragkit.retrieval import HashEmbedder, KnowledgeBase, SQLiteStore
from ragkit.retrieval.cache import CachedQueryEmbedder
from ragkit.server import create_app

POSTGRES_URL = os.environ.get("RAGKIT_TEST_DATABASE_URL")


@pytest.fixture(params=["sqlite", "postgres"])
async def queue(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[JobQueue]:
    if request.param == "postgres":
        if not POSTGRES_URL:
            pytest.skip("RAGKIT_TEST_DATABASE_URL is not set")
        backend: JobQueue = PostgresJobQueue(POSTGRES_URL)
    else:
        backend = SQLiteJobQueue(tmp_path / "jobs.db")
    async with backend:
        if request.param == "postgres":
            async with backend._pool.connection() as conn:  # type: ignore[attr-defined]
                await conn.execute("TRUNCATE ragkit_jobs")
        yield backend


async def test_requests_merge_and_one_job_runs_per_collection(queue: JobQueue) -> None:
    first = await queue.enqueue("docs")
    merged = await queue.enqueue("docs", full=True)
    other = await queue.enqueue("faq")

    assert merged.id == first.id and merged.full
    running = await queue.claim("w1", lease_seconds=60, max_attempts=3)
    assert running is not None and running.id == first.id and running.attempts == 1

    follow_up = await queue.enqueue("docs")  # queued behind the running one
    assert follow_up.id != first.id
    second = await queue.claim("w2", lease_seconds=60, max_attempts=3)
    assert second is not None and second.id == other.id  # docs is busy, so faq goes next
    assert await queue.claim("w3", lease_seconds=60, max_attempts=3) is None

    await queue.finish(first.id, "w1", report={"added": 2})
    third = await queue.claim("w3", lease_seconds=60, max_attempts=3)
    assert third is not None and third.id == follow_up.id
    done = await queue.get(first.id)
    assert done.status == "succeeded" and done.report == {"added": 2}


async def test_expired_leases_are_retried_then_failed(queue: JobQueue) -> None:
    job = await queue.enqueue("docs")
    for attempt in (1, 2):
        claimed = await queue.claim(f"w{attempt}", lease_seconds=1, max_attempts=2)
        assert claimed is not None and claimed.id == job.id and claimed.attempts == attempt
        assert await queue.heartbeat(job.id, f"w{attempt}", lease_seconds=1)
        await asyncio.sleep(1.1)  # the worker "dies": the lease runs out

    assert await queue.claim("w3", lease_seconds=1, max_attempts=2) is None
    failed = await queue.get(job.id)
    assert failed.status == "failed" and "lease" in failed.error
    assert not await queue.heartbeat(job.id, "w2", lease_seconds=1)


async def test_failed_jobs_are_retried_until_out_of_attempts(queue: JobQueue) -> None:
    job = await queue.enqueue("docs")
    claimed = await queue.claim("w", lease_seconds=60, max_attempts=2)
    await queue.finish(claimed.id, "w", error="boom", retry=True)
    assert (await queue.get(job.id)).status == "queued"

    claimed = await queue.claim("w", lease_seconds=60, max_attempts=2)
    assert claimed.attempts == 2
    await queue.finish(claimed.id, "w", error="boom again", retry=False)
    assert (await queue.get(job.id)).status == "failed"
    assert [j.id for j in await queue.list(collection="docs")] == [job.id]


def _project(root: Path, extra: str = "") -> ProjectConfig:
    for name, text in {"a.md": "# A\n\nAlpha text.", "b.md": "# B\n\nBeta text."}.items():
        (root / "data").mkdir(exist_ok=True)
        (root / "data" / name).write_text(text)
    return write_project(root, RAG_CONFIG + extra)


async def _worker(project: ProjectConfig, **kwargs) -> tuple[Worker, JobQueue, SQLiteStore]:
    queue = await create_queue(project).__aenter__()
    store = SQLiteStore(project.root / project.store.path)
    await store.open()
    return Worker(project, queue, store, embedder=HashEmbedder(256), **kwargs), queue, store


async def test_worker_runs_jobs_and_records_reports(tmp_path: Path) -> None:
    project = _project(tmp_path)
    worker, queue, store = await _worker(project)
    try:
        job = await queue.enqueue("docs")
        await worker.run(asyncio.Event(), until_idle=True)
        done = await queue.get(job.id)
        stats = await store.stats()
    finally:
        await queue.close()
        await store.close()

    assert done.status == "succeeded"
    assert done.report["added"] == 2 and done.report["collection"] == "docs"
    assert stats[0].documents == 2


async def test_source_failures_are_reported_in_the_job(tmp_path: Path) -> None:
    project = write_project(
        tmp_path,
        "name: t\nembeddings: {model: 'hash:64'}\n"
        "collections:\n  docs:\n    sources: [{type: files, path: missing}]\n",
    )
    worker, queue, store = await _worker(project)
    try:
        job = await queue.enqueue("docs")
        await worker.run(asyncio.Event(), until_idle=True)
        final = await queue.get(job.id)
    finally:
        await queue.close()
        await store.close()

    # A source that cannot be read is part of the report; the job itself completes.
    assert final.status == "succeeded"
    assert final.report["failed"][0][1].startswith("source failed")


async def test_crashing_jobs_are_retried_then_marked_failed(tmp_path: Path) -> None:
    project = write_project(
        tmp_path,
        "name: t\nembeddings: {model: 'hash:64'}\nqueue: {max_attempts: 2}\n"
        "collections:\n  docs:\n    sources: [{type: no-such-connector}]\n",
    )
    worker, queue, store = await _worker(project)
    try:
        job = await queue.enqueue("docs")
        await worker.run(asyncio.Event(), until_idle=True)
        final = await queue.get(job.id)
    finally:
        await queue.close()
        await store.close()

    assert final.status == "failed"
    assert final.attempts == 2
    assert "no-such-connector" in final.error
    assert [j.status for j in worker.completed] == ["queued", "failed"]


async def test_scheduler_queues_due_collections_once(tmp_path: Path) -> None:
    project = _project(tmp_path)
    text = (
        (tmp_path / "ragkit.yaml").read_text().replace("chunking:", "sync_every: 1h\n    chunking:")
    )
    project = write_project(tmp_path, text)
    worker, queue, store = await _worker(project, schedule=True)
    try:
        first = await worker.enqueue_due()
        again = await worker.enqueue_due()
        await worker.run(asyncio.Event(), until_idle=True)
        after_run = await worker.enqueue_due()
    finally:
        await queue.close()
        await store.close()

    assert [j.collection for j in first] == ["docs"]
    assert again == [] and after_run == []  # requested less than an hour ago


async def test_stop_lets_the_running_job_finish(tmp_path: Path) -> None:
    project = _project(tmp_path)
    worker, queue, store = await _worker(project)
    original = worker.process

    async def slow(job, name):
        stop.set()  # the stop signal arrives mid-job
        await asyncio.sleep(0.2)
        await original(job, name)

    worker.process = slow  # type: ignore[method-assign]
    stop = asyncio.Event()
    try:
        job = await queue.enqueue("docs")
        await asyncio.wait_for(worker.run(stop), timeout=10)
        status = (await queue.get(job.id)).status
    finally:
        await queue.close()
        await store.close()
    assert status == "succeeded"


def test_cli_queue_worker_and_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _project(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    queued = runner.invoke(cli, ["ingest", "--queue"])
    work = runner.invoke(cli, ["worker", "--until-idle", "--no-schedule"])
    listing = runner.invoke(cli, ["jobs"])

    assert queued.exit_code == 0 and "queued docs" in queued.output
    assert work.exit_code == 0, work.output
    assert "docs: succeeded" in work.output
    assert "succeeded" in listing.output and "+2" in listing.output


def test_http_ingest_endpoint_queues_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEY_OPS", "k-ops")
    monkeypatch.setenv("KEY_READER", "k-read")
    project = _project(
        tmp_path,
        "auth:\n  mode: api_key\n  api_keys:\n"
        "    - {key_env: KEY_OPS, scopes: [ingest, search]}\n"
        "    - {key_env: KEY_READER, scopes: [search]}\n",
    )

    with TestClient(create_app(project)) as client:
        queued = client.post(
            "/collections/docs/ingest", json={"full": True}, headers={"X-API-Key": "k-ops"}
        )
        forbidden = client.post("/collections/docs/ingest", headers={"X-API-Key": "k-read"})
        job_id = queued.json()["id"]

        async def drain() -> None:
            worker, queue, store = await _worker(project)
            try:
                await worker.run(asyncio.Event(), until_idle=True)
            finally:
                await queue.close()
                await store.close()

        asyncio.run(drain())
        job = client.get(f"/jobs/{job_id}", headers={"X-API-Key": "k-ops"}).json()
        listed = client.get("/jobs?collection=docs", headers={"X-API-Key": "k-ops"}).json()
        ready = client.get("/ready")

    assert queued.status_code == 202 and queued.json()["full"] is True
    assert forbidden.status_code == 403
    assert job["status"] == "succeeded" and job["report"]["added"] == 2
    assert [j["id"] for j in listed] == [job_id]
    assert ready.json() == {"status": "ready"}


async def test_query_embeddings_and_search_results_are_cached(tmp_path: Path) -> None:
    project = _project(tmp_path, "cache: {search_ttl_seconds: 60}\n")
    from ragkit.ingest import ingest

    await ingest(project)
    async with KnowledgeBase(project) as kb:
        first = await kb.search("docs", "alpha")
        second = await kb.search("docs", "alpha")
        other_principals = await kb.search("docs", "alpha", principals=["x"])
        embedder = kb.embedder

    assert isinstance(embedder, CachedQueryEmbedder)
    assert embedder.misses == 1 and embedder.hits >= 1  # same query text embedded once
    assert [r.chunk.id for r in first] == [r.chunk.id for r in second]
    assert [r.chunk.id for r in other_principals] == [r.chunk.id for r in first]


async def test_search_cache_is_off_by_default_and_expires(tmp_path: Path) -> None:
    from ragkit.retrieval.cache import TTLCache

    cache = TTLCache(max_entries=2, ttl_seconds=0.05)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert cache.get("a") is None and cache.get("c") == 3  # oldest evicted
    time.sleep(0.06)
    assert cache.get("c") is None
    assert TTLCache(10, 0).get("x") is None
    assert _project(tmp_path).cache.search_ttl_seconds == 0


async def test_concurrent_claims_never_share_a_job(queue: JobQueue, tmp_path: Path) -> None:
    for i in range(6):
        await queue.enqueue(f"c{i}")
    if isinstance(queue, PostgresJobQueue):
        claimers = [queue]  # one pool; SKIP LOCKED arbitrates between its connections
    else:  # separate SQLite connections, as separate worker processes would have
        claimers = [SQLiteJobQueue(tmp_path / "jobs.db") for _ in range(3)]
        for claimer in claimers:
            await claimer.open()
    try:
        claims = await asyncio.gather(
            *(
                claimers[i % len(claimers)].claim(f"w{i}", lease_seconds=60, max_attempts=3)
                for i in range(10)
            )
        )
    finally:
        if not isinstance(queue, PostgresJobQueue):
            for claimer in claimers:
                await claimer.close()

    taken = [job.id for job in claims if job is not None]
    assert len(taken) == 6 and len(set(taken)) == 6
