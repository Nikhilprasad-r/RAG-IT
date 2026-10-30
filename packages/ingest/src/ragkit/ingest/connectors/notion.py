"""Pages from Notion, converted to Markdown.

Create an internal integration, share the pages or databases with it, and set
NOTION_TOKEN. Sources:

- `database_ids`: every page in those databases,
- `page_ids`: those pages and, with `include_child_pages`, the pages beneath them,
- neither: every page shared with the integration (Notion's search API).
"""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Self

import httpx
from pydantic import Field

from ragkit.core.errors import ConfigError
from ragkit.core.registry import registry
from ragkit.core.types import RawDocument
from ragkit.ingest.connectors.base import ConnectorOptions, request_with_retry, secret

MAX_DEPTH = 8


class NotionOptions(ConnectorOptions):
    database_ids: list[str] = Field(default_factory=list)
    page_ids: list[str] = Field(default_factory=list)
    include_child_pages: bool = True
    token_env: str = "NOTION_TOKEN"
    api_url: str = "https://api.notion.com"
    notion_version: str = "2022-06-28"
    concurrency: int = Field(default=3, ge=1, description="Notion allows about 3 requests/s.")


class NotionConnector:
    def __init__(self, options: NotionOptions, *, client: httpx.AsyncClient | None = None) -> None:
        self.options = options
        roots = options.database_ids + options.page_ids
        self.name = options.name or ("notion:" + ",".join(roots) if roots else "notion:shared")
        self._client = client

    @classmethod
    def from_options(cls, options: dict[str, Any], *, root: Path) -> Self:
        return cls(NotionOptions.model_validate(options))

    async def fetch(self) -> AsyncIterator[RawDocument]:
        client = self._client or httpx.AsyncClient(
            base_url=self.options.api_url,
            timeout=30,
            headers={
                "Authorization": f"Bearer {secret(self.options.token_env, purpose='read Notion')}",
                "Notion-Version": self.options.notion_version,
            },
        )
        try:
            async for document in self._crawl(client):
                yield document
        finally:
            if self._client is None:
                await client.aclose()

    async def _crawl(self, client: httpx.AsyncClient) -> AsyncIterator[RawDocument]:
        queue: list[dict[str, Any]] = await self._roots(client)
        seen: set[str] = set()
        semaphore = asyncio.Semaphore(self.options.concurrency)
        while queue:
            batch = [page for page in queue if page["id"] not in seen]
            seen.update(page["id"] for page in batch)
            queue = []
            results = await asyncio.gather(*(self._page(client, p, semaphore) for p in batch))
            for document, children in results:
                yield document
                if self.options.include_child_pages:
                    queue += [{"id": child} for child in children if child not in seen]

    async def _roots(self, client: httpx.AsyncClient) -> list[dict[str, Any]]:
        pages: list[dict[str, Any]] = []
        for database_id in self.options.database_ids:
            pages += await self._paginate(client, "POST", f"/v1/databases/{database_id}/query")
        for page_id in self.options.page_ids:
            pages.append({"id": page_id})
        if not (self.options.database_ids or self.options.page_ids):
            pages += await self._paginate(
                client, "POST", "/v1/search", {"filter": {"property": "object", "value": "page"}}
            )
        return pages

    async def _page(
        self, client: httpx.AsyncClient, page: dict[str, Any], semaphore: asyncio.Semaphore
    ) -> tuple[RawDocument, list[str]]:
        async with semaphore:
            if "properties" not in page:
                page = await self._json(client, "GET", f"/v1/pages/{page['id']}")
            children: list[str] = []
            lines = await self._blocks(client, page["id"], 0, children)
        title = _title(page)
        markdown = f"# {title}\n\n" + "\n".join(lines)
        return (
            RawDocument(
                source=self.name,
                uri=page.get("url") or f"https://www.notion.so/{page['id'].replace('-', '')}",
                data=markdown.encode(),
                mime_type="text/markdown",
                metadata={"page_id": page["id"], "updated": page.get("last_edited_time")},
            ),
            children,
        )

    async def _blocks(
        self, client: httpx.AsyncClient, block_id: str, depth: int, child_pages: list[str]
    ) -> list[str]:
        lines: list[str] = []
        blocks = await self._paginate(client, "GET", f"/v1/blocks/{block_id}/children")
        for block in blocks:
            kind = block["type"]
            if kind == "child_page":
                child_pages.append(block["id"])
                continue
            if kind == "table":
                rows = await self._paginate(client, "GET", f"/v1/blocks/{block['id']}/children")
                lines += ["", *_table(rows, block["table"].get("has_column_header", True)), ""]
                continue
            lines.append(_line(block, depth))
            if block.get("has_children") and kind != "child_database" and depth < MAX_DEPTH:
                lines += await self._blocks(client, block["id"], depth + 1, child_pages)
        return lines

    async def _paginate(
        self, client: httpx.AsyncClient, method: str, url: str, body: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        cursor = None
        while True:
            if method == "GET":
                params = {"page_size": 100, **({"start_cursor": cursor} if cursor else {})}
                data = await self._json(client, "GET", url, params=params)
            else:
                payload = {**(body or {}), "page_size": 100}
                if cursor:
                    payload["start_cursor"] = cursor
                data = await self._json(client, "POST", url, json=payload)
            results += data.get("results", [])
            if not data.get("has_more"):
                return results
            cursor = data.get("next_cursor")

    async def _json(
        self, client: httpx.AsyncClient, method: str, url: str, **kwargs: Any
    ) -> dict[str, Any]:
        response = await request_with_retry(client, method, url, **kwargs)
        if response.status_code in (401, 403):
            raise ConfigError(
                f"Notion refused access ({response.status_code}); check NOTION_TOKEN."
            )
        if response.status_code == 404:
            raise ConfigError(f"Notion returned 404 for {url}; share it with the integration.")
        response.raise_for_status()
        return response.json()


def _text(rich_text: list[dict[str, Any]]) -> str:
    return "".join(part.get("plain_text", "") for part in rich_text)


def _title(page: dict[str, Any]) -> str:
    for prop in page.get("properties", {}).values():
        if prop.get("type") == "title":
            return _text(prop["title"]) or "Untitled"
    return "Untitled"


def _line(block: dict[str, Any], depth: int) -> str:
    kind = block["type"]
    value = block.get(kind, {})
    text = _text(value.get("rich_text", []))
    indent = "  " * depth
    if kind.startswith("heading_"):
        return f"\n{'#' * (int(kind[-1]) + 1)} {text}\n"  # the page title is the only H1
    if kind == "bulleted_list_item":
        return f"{indent}- {text}"
    if kind == "numbered_list_item":
        return f"{indent}1. {text}"
    if kind == "to_do":
        return f"{indent}- [{'x' if value.get('checked') else ' '}] {text}"
    if kind in ("quote", "callout"):
        return f"\n> {text}\n"
    if kind == "code":
        return f"\n```{value.get('language', '')}\n{text}\n```\n"
    if kind == "divider":
        return "\n---\n"
    if kind == "equation":
        return f"\n$$ {value.get('expression', '')} $$\n"
    if kind in ("bookmark", "embed", "link_preview"):
        return f"{indent}{value.get('url', '')}"
    if kind in ("image", "file", "pdf", "video"):
        caption = _text(value.get("caption", []))
        return f"{indent}[{kind}: {caption}]" if caption else ""
    return f"\n{indent}{text}\n" if text else ""


def _table(rows: list[dict[str, Any]], header: bool) -> list[str]:
    cells = [[_text(cell) for cell in row["table_row"]["cells"]] for row in rows]
    if not cells:
        return []
    width = max(len(row) for row in cells)
    cells = [row + [""] * (width - len(row)) for row in cells]
    head = cells[0] if header else [""] * width
    body = cells[1:] if header else cells
    return [
        "| " + " | ".join(head) + " |",
        "|" + " --- |" * width,
        *("| " + " | ".join(row) + " |" for row in body),
    ]


registry.register("connector", "notion", NotionConnector)
