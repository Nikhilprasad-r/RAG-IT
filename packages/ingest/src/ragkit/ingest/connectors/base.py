import asyncio
import importlib
import mimetypes
import os
import re
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from functools import cache
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from ragkit.core.config import SourceConfig
from ragkit.core.errors import ConfigError
from ragkit.core.interfaces import Connector
from ragkit.core.registry import registry

_EXTRA_TYPES = {
    ".md": "text/markdown",
    ".mdx": "text/markdown",
    ".markdown": "text/markdown",
    ".rst": "text/x-rst",
    ".py": "text/x-python",
    ".ts": "application/typescript",
    ".tsx": "application/typescript",
    ".go": "text/x-go",
    ".rs": "text/x-rust",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".toml": "application/toml",
    ".sql": "application/sql",
}


class ConnectorOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    """Overrides the connector's default identity. Changing it re-ingests the source."""


def guess_mime_type(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return _EXTRA_TYPES.get(suffix) or mimetypes.guess_type(path)[0] or "application/octet-stream"


@cache
def _glob_regex(pattern: str) -> re.Pattern[str]:
    parts, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            parts.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            parts.append(".*")
            i += 2
        elif pattern[i] == "*":
            parts.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            parts.append("[^/]")
            i += 1
        else:
            parts.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(parts) + r"\Z")


def glob_match(path: str, patterns: Iterable[str]) -> bool:
    """Match a POSIX path against globs where `**` spans directories."""
    return any(_glob_regex(p).match(path) for p in patterns)


async def bounded_map[T, R](
    items: Iterable[T], fn: Callable[[T], Awaitable[R]], concurrency: int
) -> AsyncIterator[R]:
    """Apply `fn` to items with at most `concurrency` running, yielding results as they finish."""
    semaphore = asyncio.Semaphore(concurrency)

    async def run(item: T) -> R:
        async with semaphore:
            return await fn(item)

    tasks = [asyncio.ensure_future(run(item)) for item in items]
    try:
        for task in asyncio.as_completed(tasks):
            yield await task
    finally:
        for task in tasks:
            task.cancel()


async def request_with_retry(
    client: httpx.AsyncClient, method: str, url: str, *, attempts: int = 5, **kwargs: Any
) -> httpx.Response:
    """Send a request, waiting and retrying on 429 and 5xx (honouring Retry-After)."""
    delay = 1.0
    for attempt in range(1, attempts + 1):
        response = await client.request(method, url, **kwargs)
        if response.status_code != 429 and response.status_code < 500:
            return response
        if attempt == attempts:
            break
        retry_after = response.headers.get("retry-after", "")
        wait = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else delay
        await asyncio.sleep(min(wait, 60))
        delay *= 2
    return response


def secret(env: str, *, purpose: str) -> str:
    value = os.environ.get(env)
    if not value:
        raise ConfigError(f"Set {env} to {purpose}.")
    return value


def create_connector(source: SourceConfig, *, root: Path) -> Connector:
    """Build a connector from a source entry: a registered name or 'module:Class'."""
    if ":" in source.type:
        factory = _import(source.type, root)
    else:
        factory = registry.get("connector", source.type)
    try:
        if hasattr(factory, "from_options"):
            return factory.from_options(source.options, root=root)
        return factory(**source.options)
    except ValidationError as exc:
        raise ConfigError(f"Invalid options for {source.type} source:\n{exc}") from exc


def _import(target: str, root: Path) -> Any:
    module_name, _, attribute = target.partition(":")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        return getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise ConfigError(f"Cannot import connector {target!r}: {exc}") from exc
