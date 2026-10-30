"""Incremental ingestion: fetch → parse → chunk → embed → index.

Each item's raw bytes are hashed; items whose hash has not changed since the
last run are skipped, and items a source no longer yields are deleted. Deletes
only happen when the source's listing completed, so a network failure never
empties a collection. One bad item is recorded in the report and does not stop
the run.
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field

from ragkit.core import telemetry
from ragkit.core.config import ProjectConfig
from ragkit.core.errors import ConfigError
from ragkit.core.interfaces import Connector
from ragkit.core.types import Document, RawDocument, content_hash
from ragkit.ingest.chunking import MarkdownChunker
from ragkit.ingest.connectors import create_connector
from ragkit.ingest.parsers import Parsers, UnsupportedFormatError
from ragkit.retrieval.embedders import Embedder, create_embedder
from ragkit.retrieval.stores import Store, StoredDocument, create_store

logger = logging.getLogger(__name__)

EMBED_ATTEMPTS = 4


@dataclass
class IngestReport:
    collection: str
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    chunks: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed


def document_id(collection: str, tenant_id: str, source: str, uri: str) -> str:
    key = "\0".join((collection, tenant_id, source, uri))
    return hashlib.sha256(key.encode()).hexdigest()[:32]


class IngestPipeline:
    def __init__(
        self,
        project: ProjectConfig,
        store: Store,
        embedder: Embedder,
        *,
        concurrency: int = 4,
    ) -> None:
        self.project = project
        self.store = store
        self.embedder = embedder
        self.concurrency = concurrency

    async def run(self, collection: str, *, full: bool = False) -> IngestReport:
        """Bring `collection` up to date with its sources. `full` re-processes every item."""
        config = self.project.collection(collection)
        await self._check_embedding_model(collection, full=full)
        report = IngestReport(collection)
        parsers = Parsers(config.parser)
        chunker = MarkdownChunker(config.chunking)

        with telemetry.span(f"ingest {collection}", **{"ragkit.ingest.full": full}) as span:
            for source in config.sources:
                connector = create_connector(source, root=self.project.root)
                with telemetry.span(f"ingest_source {connector.name}"):
                    await self._sync_source(
                        collection, connector, parsers, chunker, report, full=full
                    )
            for field_name in ("added", "updated", "unchanged", "deleted", "chunks"):
                span.set_attribute(f"ragkit.ingest.{field_name}", getattr(report, field_name))
            span.set_attribute("ragkit.ingest.skipped", len(report.skipped))
            span.set_attribute("ragkit.ingest.failed", len(report.failed))
        return report

    async def _sync_source(
        self,
        collection: str,
        connector: Connector,
        parsers: Parsers,
        chunker: MarkdownChunker,
        report: IngestReport,
        *,
        full: bool,
    ) -> None:
        existing = await self.store.documents(collection, connector.name)
        seen: set[str] = set()
        semaphore = asyncio.Semaphore(self.concurrency)
        tasks: set[asyncio.Task[None]] = set()

        async def process(raw: RawDocument) -> None:
            with telemetry.span("ingest_document", **{"ragkit.document.uri": raw.uri}) as span:
                try:
                    outcome = await self._process(
                        collection, raw, existing.get(raw.uri), parsers, chunker, report, full=full
                    )
                except Exception as exc:
                    logger.exception("Ingesting %s failed", raw.uri)
                    report.failed.append((raw.uri, str(exc)))
                    outcome = "failed"
                finally:
                    semaphore.release()
                span.set_attribute("ragkit.ingest.outcome", outcome)

        listing_complete = True
        try:
            async for raw in connector.fetch():
                seen.add(raw.uri)
                await semaphore.acquire()
                task = asyncio.create_task(process(raw))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
        except Exception as exc:
            listing_complete = False
            logger.exception("Source %s failed", connector.name)
            report.failed.append((connector.name, f"source failed: {exc}"))
        finally:
            if tasks:
                await asyncio.gather(*tasks)

        listing_complete = listing_complete and getattr(connector, "complete", True)
        stale = [doc.id for uri, doc in existing.items() if uri not in seen]
        if stale and not listing_complete:
            logger.warning(
                "Keeping %d documents from %s because its listing was incomplete.",
                len(stale),
                connector.name,
            )
        elif stale:
            await self.store.delete(collection, stale)
            report.deleted += len(stale)

    async def _process(
        self,
        collection: str,
        raw: RawDocument,
        previous: StoredDocument | None,
        parsers: Parsers,
        chunker: MarkdownChunker,
        report: IngestReport,
        *,
        full: bool,
    ) -> str:
        """Ingest one item and return what happened to it."""
        checksum = content_hash(raw.data)
        if previous and previous.checksum == checksum and not full:
            report.unchanged += 1
            return "unchanged"
        try:
            parsed = await parsers.parse(raw)
        except UnsupportedFormatError as exc:
            report.skipped.append((raw.uri, str(exc)))
            return "skipped"
        except Exception as exc:
            logger.warning("Parsing %s failed: %s", raw.uri, exc)
            report.failed.append((raw.uri, f"parse: {exc}"))
            return "failed"

        acl = raw.acl or self.project.collection(collection).acl
        document = Document(
            id=document_id(collection, raw.tenant_id, raw.source, raw.uri),
            collection=collection,
            source=raw.source,
            uri=raw.uri,
            content=parsed.text,
            mime_type=raw.mime_type,
            modality=parsed.modality,
            metadata={**raw.metadata, **parsed.metadata},
            acl=acl,
            tenant_id=raw.tenant_id,
            title=parsed.title,
            checksum=checksum,
        )
        chunks = chunker.chunk(document)
        if not chunks:
            report.skipped.append((raw.uri, "no text"))
            return "skipped"
        try:
            vectors = await self._embed([chunk.indexed_text for chunk in chunks])
            for chunk, vector in zip(chunks, vectors, strict=True):
                chunk.embedding = vector
                chunk.embedding_model = self.embedder.model
            await self.store.upsert(document, chunks)
        except Exception as exc:
            logger.warning("Indexing %s failed: %s", raw.uri, exc)
            report.failed.append((raw.uri, f"index: {exc}"))
            return "failed"

        report.chunks += len(chunks)
        if previous:
            report.updated += 1
            return "updated"
        report.added += 1
        return "added"

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        delay = 1.0
        for attempt in range(1, EMBED_ATTEMPTS + 1):
            try:
                return await self.embedder.embed(texts)
            except Exception as exc:
                if attempt == EMBED_ATTEMPTS:
                    raise
                logger.info("Embedding failed (%s); retrying in %.0fs", exc, delay)
                await asyncio.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")

    async def _check_embedding_model(self, collection: str, *, full: bool) -> None:
        stats = await self.store.collection_stats(collection)
        if not stats or not stats.embedding_model or stats.embedding_model == self.embedder.model:
            return
        if not full:
            raise ConfigError(
                f"Collection {collection!r} was indexed with {stats.embedding_model}, but the "
                f"project now uses {self.embedder.model}. Re-run with --full to rebuild it."
            )
        logger.info("Embedding model changed; dropping %s before re-indexing.", collection)
        await self.store.drop_collection(collection)


async def ingest(
    project: ProjectConfig,
    collections: list[str] | None = None,
    *,
    full: bool = False,
    store: Store | None = None,
    embedder: Embedder | None = None,
) -> list[IngestReport]:
    """Ingest the given collections (default: all) with the project's store and embedder."""
    names = collections or list(project.collections)
    for name in names:
        project.collection(name)
    embedder = embedder or create_embedder(project.embeddings)
    owned = store is None
    store = store or create_store(project)
    if owned:
        await store.open()
    try:
        pipeline = IngestPipeline(project, store, embedder)
        return [await pipeline.run(name, full=full) for name in names]
    finally:
        if owned:
            await store.close()
