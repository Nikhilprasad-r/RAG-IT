"""Files from a GitHub repository, read through the GitHub API.

Public repositories work without a token (subject to GitHub's anonymous rate
limits). Set GITHUB_TOKEN, or the variable named by `token_env`, for private
repositories and higher limits.
"""

import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Self
from urllib.parse import quote

import httpx
from pydantic import Field

from ragkit.core.errors import ConfigError
from ragkit.core.registry import registry
from ragkit.core.types import RawDocument
from ragkit.ingest.connectors.base import (
    ConnectorOptions,
    bounded_map,
    glob_match,
    guess_mime_type,
)

logger = logging.getLogger(__name__)


class GitHubOptions(ConnectorOptions):
    repo: str = Field(pattern=r"^[\w.-]+/[\w.-]+$", description="owner/name")
    branch: str | None = Field(default=None, description="Defaults to the repository's default.")
    include: list[str] = Field(
        default_factory=lambda: ["**/*.md", "**/*.mdx", "**/*.rst", "**/*.txt"]
    )
    """Globs of files to read."""
    exclude: list[str] = Field(default_factory=list)
    """Globs of files to skip."""
    token_env: str = "GITHUB_TOKEN"
    """Variable with a GitHub token; optional for public repositories."""
    max_file_mb: float = Field(default=5, gt=0)
    """Skip files larger than this."""
    concurrency: int = Field(default=8, ge=1)
    """Files downloaded at once."""
    api_url: str = "https://api.github.com"
    """GitHub API URL (change for GitHub Enterprise)."""
    raw_url: str = "https://raw.githubusercontent.com"
    """Host that serves raw file contents."""


class GitHubConnector:
    def __init__(self, options: GitHubOptions, *, client: httpx.AsyncClient | None = None) -> None:
        self.options = options
        self.name = options.name or f"github:{options.repo}"
        self._client = client
        # False when a file failed to download, so ingestion keeps its earlier copy.
        self.complete = True

    @classmethod
    def from_options(cls, options: dict[str, Any], *, root: Path) -> Self:
        return cls(GitHubOptions.model_validate(options))

    async def fetch(self) -> AsyncIterator[RawDocument]:
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "ragkit"}
        if token := os.environ.get(self.options.token_env):
            headers["Authorization"] = f"Bearer {token}"
        client = self._client or httpx.AsyncClient(
            base_url=self.options.api_url, headers=headers, timeout=30, follow_redirects=True
        )
        try:
            async for document in self._fetch(client):
                yield document
        finally:
            if self._client is None:
                await client.aclose()

    async def _fetch(self, client: httpx.AsyncClient) -> AsyncIterator[RawDocument]:
        repo = self.options.repo
        branch = (
            self.options.branch or (await self._get(client, f"/repos/{repo}"))["default_branch"]
        )
        tree = await self._get(
            client, f"/repos/{repo}/git/trees/{quote(branch, safe='')}", recursive="1"
        )
        if tree.get("truncated"):
            logger.warning("GitHub truncated the file list of %s; some files are missing.", repo)
        max_bytes = self.options.max_file_mb * 1024 * 1024
        paths = [
            item["path"]
            for item in tree["tree"]
            if item["type"] == "blob"
            and item.get("size", 0) <= max_bytes
            and glob_match(item["path"], self.options.include)
            and not glob_match(item["path"], self.options.exclude)
        ]

        async def download(path: str) -> RawDocument | None:
            # raw.githubusercontent.com does not count against the REST API rate limit.
            try:
                response = await client.get(
                    f"{self.options.raw_url}/{repo}/{quote(branch, safe='')}/{quote(path)}"
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning("Downloading %s from %s failed: %s", path, repo, exc)
                self.complete = False
                return None
            return RawDocument(
                source=self.name,
                uri=f"https://github.com/{repo}/blob/{branch}/{path}",
                data=response.content,
                mime_type=guess_mime_type(path),
                metadata={"repo": repo, "branch": branch, "path": path},
            )

        async for document in bounded_map(sorted(paths), download, self.options.concurrency):
            if document is not None:
                yield document

    async def _get(self, client: httpx.AsyncClient, url: str, **params: str) -> dict[str, Any]:
        response = await client.get(url, params=params)
        if response.status_code == 404:
            raise ConfigError(f"GitHub returned 404 for {url}; check the repo, branch and token.")
        response.raise_for_status()
        return response.json()


registry.register("connector", "github", GitHubConnector)
