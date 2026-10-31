"""Ingestion workers and the sync scheduler.

`ragkit worker` runs `Worker.run`: it claims jobs from the queue, renews each
job's lease while ingesting, and records the report. With `schedule=True` it
also enqueues collections whose `sync_every` interval has passed; several
workers can schedule at once, because the queue merges duplicate requests.

On SIGTERM or SIGINT a worker stops claiming, finishes the jobs it is running,
and exits, so rolling deploys and scale-downs do not lose work.
"""

import asyncio
import contextlib
import logging
import os
import socket
import time
import uuid
from dataclasses import asdict

from ragkit.core import telemetry
from ragkit.core.config import ProjectConfig
from ragkit.ingest.jobs import Job, JobQueue
from ragkit.ingest.pipeline import IngestPipeline
from ragkit.retrieval.embedders import Embedder, create_embedder
from ragkit.retrieval.stores import Store

logger = logging.getLogger(__name__)


def worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"


class Worker:
    def __init__(
        self,
        project: ProjectConfig,
        queue: JobQueue,
        store: Store,
        *,
        embedder: Embedder | None = None,
        concurrency: int = 1,
        schedule: bool = False,
        name: str | None = None,
    ) -> None:
        self.project = project
        self.queue = queue
        self.store = store
        self.embedder = embedder or create_embedder(project.embeddings)
        self.concurrency = concurrency
        self.schedule = schedule
        self.name = name or worker_id()
        self.completed: list[Job] = []

    async def run(self, stop: asyncio.Event, *, until_idle: bool = False) -> None:
        """Process jobs until `stop` is set (or, with `until_idle`, the queue is empty)."""

        async def loop(slot: int) -> None:
            while not stop.is_set():
                if self.schedule and slot == 0:
                    await self.enqueue_due()
                job = await self.queue.claim(
                    f"{self.name}/{slot}",
                    lease_seconds=self.project.queue.lease_seconds,
                    max_attempts=self.project.queue.max_attempts,
                )
                if job is None:
                    if until_idle:
                        return
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(stop.wait(), self.project.queue.poll_seconds)
                    continue
                await self.process(job, f"{self.name}/{slot}")

        await asyncio.gather(*(loop(slot) for slot in range(self.concurrency)))

    async def process(self, job: Job, worker: str) -> None:
        lease = self.project.queue.lease_seconds
        heartbeat = asyncio.create_task(self._heartbeat(job, worker, lease))
        attributes = {"ragkit.job.id": job.id, "ragkit.job.attempt": job.attempts}
        try:
            with telemetry.span(f"ingest_job {job.collection}", **attributes):
                pipeline = IngestPipeline(self.project, self.store, self.embedder)
                report = await pipeline.run(job.collection, full=job.full)
        except Exception as exc:
            retry = job.attempts < self.project.queue.max_attempts
            logger.exception("Job %s (%s) failed", job.id, job.collection)
            await self.queue.finish(
                job.id, worker, error=f"{type(exc).__name__}: {exc}", retry=retry
            )
        else:
            await self.queue.finish(job.id, worker, report=asdict(report))
            logger.info("Job %s (%s) done: %s", job.id, job.collection, report)
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
        self.completed.append(await self.queue.get(job.id))

    async def enqueue_due(self) -> list[Job]:
        """Enqueue every collection whose sync interval has elapsed."""
        now = time.time()
        jobs = []
        for name, config in self.project.collections.items():
            interval = config.sync_seconds
            if interval is None:
                continue
            last = await self.queue.last_requested(name)
            if last is None or now - last >= interval:
                jobs.append(await self.queue.enqueue(name, requested_by="scheduler"))
        return jobs

    async def _heartbeat(self, job: Job, worker: str, lease: int) -> None:
        while True:
            await asyncio.sleep(max(1.0, lease / 3))
            if not await self.queue.heartbeat(job.id, worker, lease_seconds=lease):
                logger.warning("Lost the lease on job %s; another worker may retry it.", job.id)
                return
