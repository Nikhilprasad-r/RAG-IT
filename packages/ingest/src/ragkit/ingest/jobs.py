"""A durable ingestion job queue in the project's own database.

Postgres (cluster mode) claims jobs with `FOR UPDATE SKIP LOCKED`, so any number
of worker pods can share it; SQLite (local mode) serialises claims with
`BEGIN IMMEDIATE`, so several worker processes on one machine can.

Rules:

- At most one queued job per collection: enqueueing again returns the queued job
  (upgrading it to a full re-ingest if asked).
- At most one running job per collection.
- A running job holds a lease that its worker renews. If the worker dies, the
  lease expires and another worker retries the job, up to `max_attempts`.
"""

import asyncio
import json
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self, TypeVar

from ragkit.core.config import ProjectConfig
from ragkit.core.errors import ConfigError, RagkitError

T = TypeVar("T")

Status = Literal["queued", "running", "succeeded", "failed"]


class JobNotFoundError(RagkitError):
    pass


@dataclass
class Job:
    id: str
    collection: str
    full: bool = False
    status: Status = "queued"
    attempts: int = 0
    requested_at: float = field(default_factory=time.time)
    requested_by: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    lease_until: float | None = None
    worker: str | None = None
    report: dict[str, Any] | None = None
    error: str | None = None


_COLUMNS = (
    "id, collection, full_sync, status, attempts, requested_at, requested_by, started_at, "
    "finished_at, lease_until, worker, report, error"
)


class JobQueue(ABC):
    async def __aenter__(self) -> Self:
        await self.open()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    @abstractmethod
    async def open(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def enqueue(
        self, collection: str, *, full: bool = False, requested_by: str | None = None
    ) -> Job: ...

    @abstractmethod
    async def claim(self, worker: str, *, lease_seconds: int, max_attempts: int) -> Job | None:
        """Take the oldest runnable job, or None."""

    @abstractmethod
    async def heartbeat(self, job_id: str, worker: str, *, lease_seconds: int) -> bool:
        """Extend the lease; False if the job is no longer this worker's."""

    @abstractmethod
    async def finish(
        self,
        job_id: str,
        worker: str,
        *,
        report: dict[str, Any] | None = None,
        error: str | None = None,
        retry: bool = False,
    ) -> None:
        """Record success (no error), a retry (error and retry) or a final failure."""

    @abstractmethod
    async def get(self, job_id: str) -> Job: ...

    @abstractmethod
    async def list(self, *, collection: str | None = None, limit: int = 50) -> list[Job]: ...

    @abstractmethod
    async def last_requested(self, collection: str) -> float | None: ...


def _row_to_job(row: Any) -> Job:
    get = row.__getitem__
    report = get("report")
    return Job(
        id=get("id"),
        collection=get("collection"),
        full=bool(get("full_sync")),
        status=get("status"),
        attempts=get("attempts"),
        requested_at=get("requested_at"),
        requested_by=get("requested_by"),
        started_at=get("started_at"),
        finished_at=get("finished_at"),
        lease_until=get("lease_until"),
        worker=get("worker"),
        report=json.loads(report) if isinstance(report, str) else report,
        error=get("error"),
    )


class SQLiteJobQueue(JobQueue):
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS ragkit_jobs (
        id TEXT PRIMARY KEY,
        collection TEXT NOT NULL,
        full_sync INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0,
        requested_at REAL NOT NULL,
        requested_by TEXT,
        started_at REAL,
        finished_at REAL,
        lease_until REAL,
        worker TEXT,
        report TEXT,
        error TEXT
    );
    CREATE INDEX IF NOT EXISTS ragkit_jobs_status ON ragkit_jobs (status, requested_at);
    CREATE UNIQUE INDEX IF NOT EXISTS ragkit_jobs_one_queued
        ON ragkit_jobs (collection) WHERE status = 'queued';
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._db: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode = WAL")
        db.execute("PRAGMA busy_timeout = 10000")
        db.executescript(self._SCHEMA)
        self._db = db

    async def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    async def enqueue(
        self, collection: str, *, full: bool = False, requested_by: str | None = None
    ) -> Job:
        def write(db: sqlite3.Connection) -> Job:
            with _immediate(db):
                row = db.execute(
                    f"SELECT {_COLUMNS} FROM ragkit_jobs "
                    "WHERE collection = ? AND status = 'queued'",
                    (collection,),
                ).fetchone()
                if row is not None:
                    if full and not row["full_sync"]:
                        db.execute(
                            "UPDATE ragkit_jobs SET full_sync = 1 WHERE id = ?", (row["id"],)
                        )
                    return _row_to_job(row) if not full else _job(db, row["id"])
                job = Job(
                    id=uuid.uuid4().hex, collection=collection, full=full, requested_by=requested_by
                )
                db.execute(
                    "INSERT INTO ragkit_jobs (id, collection, full_sync, status, requested_at, "
                    "requested_by) VALUES (?, ?, ?, 'queued', ?, ?)",
                    (job.id, collection, int(full), job.requested_at, requested_by),
                )
                return job

        return await self._run(write)

    async def claim(self, worker: str, *, lease_seconds: int, max_attempts: int) -> Job | None:
        def write(db: sqlite3.Connection) -> Job | None:
            now = time.time()
            with _immediate(db):
                # Running jobs whose worker vanished and that are out of attempts fail for good.
                db.execute(
                    "UPDATE ragkit_jobs SET status = 'failed', finished_at = ?, "
                    "error = COALESCE(error, 'worker stopped renewing its lease') "
                    "WHERE status = 'running' AND lease_until < ? AND attempts >= ?",
                    (now, now, max_attempts),
                )
                row = db.execute(
                    f"SELECT {_COLUMNS} FROM ragkit_jobs j "
                    "WHERE (j.status = 'queued' OR (j.status = 'running' AND j.lease_until < ?)) "
                    "AND j.attempts < ? AND NOT EXISTS (SELECT 1 FROM ragkit_jobs r "
                    "WHERE r.collection = j.collection AND r.id != j.id "
                    "AND r.status = 'running' AND r.lease_until >= ?) "
                    "ORDER BY j.requested_at LIMIT 1",
                    (now, max_attempts, now),
                ).fetchone()
                if row is None:
                    return None
                if row["status"] == "running":
                    # A stale running job and a queued one for the same collection: keep one.
                    db.execute(
                        "DELETE FROM ragkit_jobs WHERE collection = ? AND status = 'queued'",
                        (row["collection"],),
                    )
                db.execute(
                    "UPDATE ragkit_jobs SET status = 'running', attempts = attempts + 1, "
                    "started_at = ?, lease_until = ?, worker = ? WHERE id = ?",
                    (now, now + lease_seconds, worker, row["id"]),
                )
                return _job(db, row["id"])

        return await self._run(write)

    async def heartbeat(self, job_id: str, worker: str, *, lease_seconds: int) -> bool:
        def write(db: sqlite3.Connection) -> bool:
            cursor = db.execute(
                "UPDATE ragkit_jobs SET lease_until = ? "
                "WHERE id = ? AND worker = ? AND status = 'running'",
                (time.time() + lease_seconds, job_id, worker),
            )
            return cursor.rowcount == 1

        return await self._run(write)

    async def finish(
        self,
        job_id: str,
        worker: str,
        *,
        report: dict[str, Any] | None = None,
        error: str | None = None,
        retry: bool = False,
    ) -> None:
        status = "succeeded" if error is None else ("queued" if retry else "failed")

        def write(db: sqlite3.Connection) -> None:
            with _immediate(db):
                if (
                    status == "queued"
                    and db.execute(
                        "SELECT 1 FROM ragkit_jobs WHERE status = 'queued' AND collection = "
                        "(SELECT collection FROM ragkit_jobs WHERE id = ?)",
                        (job_id,),
                    ).fetchone()
                ):
                    final = "failed"  # a newer queued job will run anyway
                else:
                    final = status
                db.execute(
                    "UPDATE ragkit_jobs SET status = ?, finished_at = ?, lease_until = NULL, "
                    "report = ?, error = ? WHERE id = ? AND worker = ?",
                    (
                        final,
                        time.time() if final != "queued" else None,
                        json.dumps(report) if report is not None else None,
                        error,
                        job_id,
                        worker,
                    ),
                )

        await self._run(write)

    async def get(self, job_id: str) -> Job:
        return await self._run(lambda db: _job(db, job_id))

    async def list(self, *, collection: str | None = None, limit: int = 50) -> list[Job]:
        def read(db: sqlite3.Connection) -> list[Job]:
            where, params = ("WHERE collection = ?", [collection]) if collection else ("", [])
            rows = db.execute(
                f"SELECT {_COLUMNS} FROM ragkit_jobs {where} ORDER BY requested_at DESC LIMIT ?",
                [*params, limit],
            ).fetchall()
            return [_row_to_job(row) for row in rows]

        return await self._run(read)

    async def last_requested(self, collection: str) -> float | None:
        row = await self._run(
            lambda db: db.execute(
                "SELECT MAX(requested_at) AS last FROM ragkit_jobs WHERE collection = ?",
                (collection,),
            ).fetchone()
        )
        return row["last"]

    async def _run(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        if self._db is None:
            raise RuntimeError("SQLiteJobQueue is not open.")
        db = self._db
        async with self._lock:
            return await asyncio.to_thread(fn, db)


class _immediate:
    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def __enter__(self) -> None:
        self.db.execute("BEGIN IMMEDIATE")

    def __exit__(self, exc_type: object, *_: object) -> None:
        self.db.execute("ROLLBACK" if exc_type else "COMMIT")


def _job(db: sqlite3.Connection, job_id: str) -> Job:
    row = db.execute(f"SELECT {_COLUMNS} FROM ragkit_jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise JobNotFoundError(f"No job {job_id}.")
    return _row_to_job(row)


class PostgresJobQueue(JobQueue):
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS ragkit_jobs (
        id TEXT PRIMARY KEY,
        collection TEXT NOT NULL,
        full_sync BOOLEAN NOT NULL DEFAULT false,
        status TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0,
        requested_at DOUBLE PRECISION NOT NULL,
        requested_by TEXT,
        started_at DOUBLE PRECISION,
        finished_at DOUBLE PRECISION,
        lease_until DOUBLE PRECISION,
        worker TEXT,
        report JSONB,
        error TEXT
    );
    CREATE INDEX IF NOT EXISTS ragkit_jobs_status ON ragkit_jobs (status, requested_at);
    CREATE UNIQUE INDEX IF NOT EXISTS ragkit_jobs_one_queued
        ON ragkit_jobs (collection) WHERE status = 'queued';
    """

    def __init__(self, database_url: str) -> None:
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import AsyncConnectionPool
        except ImportError as exc:
            raise ConfigError("The postgres queue needs `ragkit-retrieval[postgres]`.") from exc
        self._pool = AsyncConnectionPool(
            database_url,
            min_size=1,
            max_size=4,
            open=False,
            kwargs={"row_factory": dict_row, "autocommit": True},
        )

    async def open(self) -> None:
        await self._pool.open(wait=True)
        from ragkit.retrieval.stores.postgres import schema_lock

        async with self._pool.connection() as conn, schema_lock(conn):
            await conn.execute(self._SCHEMA)

    async def close(self) -> None:
        await self._pool.close()

    async def enqueue(
        self, collection: str, *, full: bool = False, requested_by: str | None = None
    ) -> Job:
        async with self._pool.connection() as conn:
            row = await (
                await conn.execute(
                    f"INSERT INTO ragkit_jobs (id, collection, full_sync, status, requested_at, "
                    f"requested_by) VALUES (%s, %s, %s, 'queued', %s, %s) "
                    f"ON CONFLICT (collection) WHERE status = 'queued' "
                    f"DO UPDATE SET full_sync = ragkit_jobs.full_sync OR EXCLUDED.full_sync "
                    f"RETURNING {_COLUMNS}",
                    (uuid.uuid4().hex, collection, full, time.time(), requested_by),
                )
            ).fetchone()
        return _row_to_job(row)

    async def claim(self, worker: str, *, lease_seconds: int, max_attempts: int) -> Job | None:
        now = time.time()
        async with self._pool.connection() as conn, conn.transaction():
            await conn.execute(
                "UPDATE ragkit_jobs SET status = 'failed', finished_at = %s, "
                "error = COALESCE(error, 'worker stopped renewing its lease') "
                "WHERE status = 'running' AND lease_until < %s AND attempts >= %s",
                (now, now, max_attempts),
            )
            row = await (
                await conn.execute(
                    "SELECT id, collection, status FROM ragkit_jobs j "
                    "WHERE (j.status = 'queued' OR (j.status = 'running' AND j.lease_until < %s)) "
                    "AND j.attempts < %s AND NOT EXISTS (SELECT 1 FROM ragkit_jobs r "
                    "WHERE r.collection = j.collection AND r.id != j.id "
                    "AND r.status = 'running' AND r.lease_until >= %s) "
                    "ORDER BY j.requested_at LIMIT 1 FOR UPDATE SKIP LOCKED",
                    (now, max_attempts, now),
                )
            ).fetchone()
            if row is None:
                return None
            if row["status"] == "running":
                await conn.execute(
                    "DELETE FROM ragkit_jobs WHERE collection = %s AND status = 'queued'",
                    (row["collection"],),
                )
            claimed = await (
                await conn.execute(
                    f"UPDATE ragkit_jobs SET status = 'running', attempts = attempts + 1, "
                    f"started_at = %s, lease_until = %s, worker = %s WHERE id = %s "
                    f"RETURNING {_COLUMNS}",
                    (now, now + lease_seconds, worker, row["id"]),
                )
            ).fetchone()
        return _row_to_job(claimed)

    async def heartbeat(self, job_id: str, worker: str, *, lease_seconds: int) -> bool:
        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                "UPDATE ragkit_jobs SET lease_until = %s "
                "WHERE id = %s AND worker = %s AND status = 'running'",
                (time.time() + lease_seconds, job_id, worker),
            )
            return cursor.rowcount == 1

    async def finish(
        self,
        job_id: str,
        worker: str,
        *,
        report: dict[str, Any] | None = None,
        error: str | None = None,
        retry: bool = False,
    ) -> None:
        status = "succeeded" if error is None else ("queued" if retry else "failed")
        async with self._pool.connection() as conn, conn.transaction():
            if status == "queued":
                newer = await (
                    await conn.execute(
                        "SELECT 1 FROM ragkit_jobs WHERE status = 'queued' AND collection = "
                        "(SELECT collection FROM ragkit_jobs WHERE id = %s)",
                        (job_id,),
                    )
                ).fetchone()
                status = "failed" if newer else status
            await conn.execute(
                "UPDATE ragkit_jobs SET status = %s, finished_at = %s, lease_until = NULL, "
                "report = %s::jsonb, error = %s WHERE id = %s AND worker = %s",
                (
                    status,
                    time.time() if status != "queued" else None,
                    json.dumps(report) if report is not None else None,
                    error,
                    job_id,
                    worker,
                ),
            )

    async def get(self, job_id: str) -> Job:
        async with self._pool.connection() as conn:
            row = await (
                await conn.execute(f"SELECT {_COLUMNS} FROM ragkit_jobs WHERE id = %s", (job_id,))
            ).fetchone()
        if row is None:
            raise JobNotFoundError(f"No job {job_id}.")
        return _row_to_job(row)

    async def list(self, *, collection: str | None = None, limit: int = 50) -> list[Job]:
        where, params = ("WHERE collection = %s", [collection]) if collection else ("", [])
        async with self._pool.connection() as conn:
            rows = await (
                await conn.execute(
                    f"SELECT {_COLUMNS} FROM ragkit_jobs {where} "
                    f"ORDER BY requested_at DESC LIMIT %s",
                    [*params, limit],
                )
            ).fetchall()
        return [_row_to_job(row) for row in rows]

    async def last_requested(self, collection: str) -> float | None:
        async with self._pool.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT MAX(requested_at) AS last FROM ragkit_jobs WHERE collection = %s",
                    (collection,),
                )
            ).fetchone()
        return row["last"]


def create_queue(project: ProjectConfig) -> JobQueue:
    """The queue lives next to the index: Postgres in cluster mode, SQLite locally."""
    if project.store.backend == "postgres":
        if not project.database_url:
            raise ConfigError("store.backend is 'postgres' but database_url is not set.")
        return PostgresJobQueue(project.database_url)
    return SQLiteJobQueue((project.root / project.store.path).with_name("jobs.db"))
