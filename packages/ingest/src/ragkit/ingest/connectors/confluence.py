"""Pages from Confluence Cloud spaces, through the v2 REST API.

Authentication: an Atlassian account email (CONFLUENCE_EMAIL) and API token
(CONFLUENCE_API_TOKEN) by default; set `email_env` to an unset variable to send
the token as a bearer token instead. Page bodies arrive in Confluence's storage
format (XHTML) and are parsed as HTML.

Page restrictions are not synced: give the collection an `acl` if the spaces
are not readable by everyone who can query it.
"""

import html
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Self
from urllib.parse import urljoin, urlparse

import httpx
from pydantic import Field

from ragkit.core.errors import ConfigError
from ragkit.core.registry import registry
from ragkit.core.types import RawDocument
from ragkit.ingest.connectors.base import ConnectorOptions, request_with_retry, secret


class ConfluenceOptions(ConnectorOptions):
    base_url: str = Field(
        description="Site URL ending in /wiki, e.g. https://acme.atlassian.net/wiki"
    )
    spaces: list[str] = Field(min_length=1, description="Space keys to ingest.")
    email_env: str = "CONFLUENCE_EMAIL"
    """Variable with the Atlassian account email; when unset, the token is sent as a bearer token.
    """
    token_env: str = "CONFLUENCE_API_TOKEN"
    """Variable with the API token."""
    page_size: int = Field(default=100, ge=1, le=250)
    """Pages per API request."""


class ConfluenceConnector:
    def __init__(
        self, options: ConfluenceOptions, *, client: httpx.AsyncClient | None = None
    ) -> None:
        self.options = options
        self.base_url = options.base_url.rstrip("/")
        self.name = options.name or f"confluence:{self.base_url}:{','.join(options.spaces)}"
        self._client = client

    @classmethod
    def from_options(cls, options: dict[str, Any], *, root: Path) -> Self:
        return cls(ConfluenceOptions.model_validate(options))

    async def fetch(self) -> AsyncIterator[RawDocument]:
        client = self._client or httpx.AsyncClient(timeout=30, **self._auth())
        try:
            for key in self.options.spaces:
                async for document in self._space(client, key):
                    yield document
        finally:
            if self._client is None:
                await client.aclose()

    def _auth(self) -> dict[str, Any]:
        token = secret(self.options.token_env, purpose="read Confluence")
        if email := os.environ.get(self.options.email_env):
            return {"auth": httpx.BasicAuth(email, token)}
        return {"headers": {"Authorization": f"Bearer {token}"}}

    async def _space(self, client: httpx.AsyncClient, key: str) -> AsyncIterator[RawDocument]:
        spaces = await self._get(client, f"{self.base_url}/api/v2/spaces", params={"keys": key})
        if not spaces["results"]:
            raise ConfigError(f"Confluence space {key!r} was not found or is not visible.")
        space_id = spaces["results"][0]["id"]
        origin = "{0.scheme}://{0.netloc}".format(urlparse(self.base_url))
        url: str | None = f"{self.base_url}/api/v2/spaces/{space_id}/pages"
        params: dict[str, Any] | None = {
            "body-format": "storage",
            "status": "current",
            "limit": self.options.page_size,
        }
        while url:
            data = await self._get(client, url, params=params)
            for page in data["results"]:
                yield self._document(key, page)
            next_link = data.get("_links", {}).get("next")
            url, params = (urljoin(origin, next_link), None) if next_link else (None, None)

    def _document(self, space: str, page: dict[str, Any]) -> RawDocument:
        title = page.get("title") or f"Page {page['id']}"
        body = page.get("body", {}).get("storage", {}).get("value", "")
        markup = (
            f"<html><head><title>{html.escape(title)}</title></head>"
            f"<body><h1>{html.escape(title)}</h1>{body}</body></html>"
        )
        version = page.get("version", {})
        return RawDocument(
            source=self.name,
            uri=self.base_url + page.get("_links", {}).get("webui", f"/pages/{page['id']}"),
            data=markup.encode(),
            mime_type="text/html",
            metadata={
                "space": space,
                "page_id": page["id"],
                "version": version.get("number"),
                "updated": version.get("createdAt"),
            },
        )

    async def _get(self, client: httpx.AsyncClient, url: str, **kwargs: Any) -> dict[str, Any]:
        response = await request_with_retry(client, "GET", url, **kwargs)
        if response.status_code in (401, 403):
            raise ConfigError(
                f"Confluence refused access ({response.status_code}); check the token."
            )
        response.raise_for_status()
        return response.json()


registry.register("connector", "confluence", ConfluenceConnector)
