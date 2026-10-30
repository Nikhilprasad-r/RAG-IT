"""Breadth-first website crawler.

Starts from `urls`, follows links up to `max_depth` hops, stays under each start
URL's host and path prefix, and honours robots.txt.
"""

import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Self
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup
from pydantic import Field

from ragkit.core.registry import registry
from ragkit.core.types import RawDocument
from ragkit.ingest.connectors.base import ConnectorOptions, bounded_map, glob_match

logger = logging.getLogger(__name__)

USER_AGENT = "ragkit-crawler/0.1"


class WebOptions(ConnectorOptions):
    urls: list[str] = Field(min_length=1)
    max_pages: int = Field(default=100, ge=1)
    max_depth: int = Field(default=2, ge=0)
    include: list[str] = Field(
        default_factory=list, description="URL globs to keep; empty keeps every in-scope page."
    )
    exclude: list[str] = Field(default_factory=list)
    respect_robots: bool = True
    concurrency: int = Field(default=4, ge=1)
    timeout: float = Field(default=20, gt=0)


class WebConnector:
    def __init__(self, options: WebOptions, *, client: httpx.AsyncClient | None = None) -> None:
        self.options = options
        self.name = options.name or f"web:{options.urls[0]}"
        self._client = client
        self._robots: dict[str, RobotFileParser | None] = {}
        self._scopes = [_scope(url) for url in options.urls]
        # Set to False when a page could not be fetched for a transient reason, so
        # ingestion keeps its earlier copy instead of deleting it.
        self.complete = True

    @classmethod
    def from_options(cls, options: dict[str, Any], *, root: Path) -> Self:
        return cls(WebOptions.model_validate(options))

    async def fetch(self) -> AsyncIterator[RawDocument]:
        client = self._client or httpx.AsyncClient(
            timeout=self.options.timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        try:
            async for document in self._crawl(client):
                yield document
        finally:
            if self._client is None:
                await client.aclose()

    async def _crawl(self, client: httpx.AsyncClient) -> AsyncIterator[RawDocument]:
        seen = {_normalize(url) for url in self.options.urls}
        frontier = list(seen)
        fetched = 0
        for depth in range(self.options.max_depth + 1):
            frontier = frontier[: self.options.max_pages - fetched]
            if not frontier:
                return
            next_frontier: list[str] = []
            visits = bounded_map(
                frontier, lambda url: self._visit(client, url), self.options.concurrency
            )
            async for document, links in visits:
                fetched += 1
                if document is not None and self._keep(document.uri):
                    yield document
                if depth < self.options.max_depth:
                    for link in links:
                        if link not in seen and self._in_scope(link):
                            seen.add(link)
                            next_frontier.append(link)
            frontier = sorted(next_frontier)

    async def _visit(
        self, client: httpx.AsyncClient, url: str
    ) -> tuple[RawDocument | None, list[str]]:
        if not await self._allowed(client, url):
            return None, []
        try:
            response = await client.get(url)
        except httpx.HTTPError as exc:
            logger.warning("Fetching %s failed: %s", url, exc)
            self.complete = False
            return None, []
        if response.status_code != 200:
            logger.info("Skipping %s: HTTP %s", url, response.status_code)
            if response.status_code >= 500 or response.status_code == 429:
                self.complete = False
            return None, []
        mime = response.headers.get("content-type", "text/html").split(";")[0].strip()
        final_url = _normalize(str(response.url))
        links = _links(final_url, response.content) if mime == "text/html" else []
        document = RawDocument(
            source=self.name,
            uri=final_url,
            data=response.content,
            mime_type=mime,
            metadata={"url": final_url},
        )
        return document, links

    def _keep(self, url: str) -> bool:
        if self.options.include and not glob_match(url, self.options.include):
            return False
        return not glob_match(url, self.options.exclude)

    def _in_scope(self, url: str) -> bool:
        parsed = urlparse(url)
        return any(
            parsed.netloc == host and parsed.path.startswith(prefix)
            for host, prefix in self._scopes
        )

    async def _allowed(self, client: httpx.AsyncClient, url: str) -> bool:
        if not self.options.respect_robots:
            return True
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots:
            self._robots[origin] = await _load_robots(client, origin)
        robots = self._robots[origin]
        return robots is None or robots.can_fetch(USER_AGENT, url)


async def _load_robots(client: httpx.AsyncClient, origin: str) -> RobotFileParser | None:
    try:
        response = await client.get(f"{origin}/robots.txt")
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    parser = RobotFileParser()
    parser.parse(response.text.splitlines())
    return parser


def _links(base: str, html: bytes) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for anchor in soup.find_all("a", href=True):
        url = _normalize(urljoin(base, str(anchor["href"])))
        if urlparse(url).scheme in ("http", "https"):
            links.add(url)
    return sorted(links)


def _normalize(url: str) -> str:
    return urldefrag(url)[0]


def _scope(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    path = parsed.path if parsed.path.endswith("/") else parsed.path.rsplit("/", 1)[0] + "/"
    return parsed.netloc, path


registry.register("connector", "web", WebConnector)
