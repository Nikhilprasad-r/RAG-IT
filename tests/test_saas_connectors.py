import base64
import json

import httpx
import pytest

from ragkit.core.errors import ConfigError
from ragkit.ingest.connectors import ConfluenceConnector, NotionConnector
from ragkit.ingest.connectors.confluence import ConfluenceOptions
from ragkit.ingest.connectors.notion import NotionOptions
from ragkit.ingest.parsers import Parsers

BASE = "https://acme.atlassian.net/wiki"


def _page(page_id: str, title: str, body: str) -> dict:
    return {
        "id": page_id,
        "title": title,
        "version": {"number": 3, "createdAt": "2026-10-01T10:00:00Z"},
        "body": {"storage": {"value": body}},
        "_links": {"webui": f"/spaces/ENG/pages/{page_id}"},
    }


async def test_confluence_pages_across_pagination(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONFLUENCE_EMAIL", "bot@acme.com")
    monkeypatch.setenv("CONFLUENCE_API_TOKEN", "secret")
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        assert (
            request.headers["authorization"]
            == "Basic " + base64.b64encode(b"bot@acme.com:secret").decode()
        )
        if request.url.path == "/wiki/api/v2/spaces":
            assert request.url.params["keys"] == "ENG"
            return httpx.Response(200, json={"results": [{"id": "77", "key": "ENG"}]})
        if (
            request.url.path == "/wiki/api/v2/spaces/77/pages"
            and "cursor" not in request.url.params
        ):
            assert request.url.params["body-format"] == "storage"
            return httpx.Response(
                200,
                json={
                    "results": [_page("1", "Runbook", "<h2>Deploy</h2><p>Run make deploy.</p>")],
                    "_links": {"next": "/wiki/api/v2/spaces/77/pages?cursor=abc"},
                },
            )
        if request.url.params.get("cursor") == "abc":
            if len([c for c in calls if "cursor=abc" in c]) == 1:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(
                200, json={"results": [_page("2", "On-call", "<p>Page the SRE.</p>")], "_links": {}}
            )
        return httpx.Response(404)

    options = ConfluenceOptions(base_url=BASE, spaces=["ENG"])
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = ConfluenceConnector(options, client=client)
    connector._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), **connector._auth()
    )

    items = [item async for item in connector.fetch()]

    assert [i.uri for i in items] == [f"{BASE}/spaces/ENG/pages/1", f"{BASE}/spaces/ENG/pages/2"]
    assert items[0].metadata == {
        "space": "ENG",
        "page_id": "1",
        "version": 3,
        "updated": "2026-10-01T10:00:00Z",
    }
    parsed = await Parsers().parse(items[0])
    assert parsed.title == "Runbook"
    assert "## Deploy" in parsed.text and "Run make deploy." in parsed.text


async def test_confluence_needs_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONFLUENCE_API_TOKEN", raising=False)
    connector = ConfluenceConnector(ConfluenceOptions(base_url=BASE, spaces=["ENG"]))

    with pytest.raises(ConfigError, match="CONFLUENCE_API_TOKEN"):
        [item async for item in connector.fetch()]


def _rt(text: str) -> list[dict]:
    return [{"plain_text": text}]


def _block(block_id: str, kind: str, has_children: bool = False, **value: object) -> dict:
    return {"id": block_id, "type": kind, "has_children": has_children, kind: value}


NOTION_BLOCKS = {
    "p1": [
        _block("b1", "heading_1", rich_text=_rt("Setup")),
        _block("b2", "paragraph", rich_text=_rt("Install the agent.")),
        _block("b3", "bulleted_list_item", True, rich_text=_rt("macOS")),
        _block("b4", "code", rich_text=_rt("brew install nimbus"), language="bash"),
        _block("b5", "table", True, has_column_header=True),
        _block("p2", "child_page", title="Troubleshooting"),
    ],
    "b3": [_block("b31", "bulleted_list_item", rich_text=_rt("Apple silicon"))],
    "b5": [
        {"id": "r1", "type": "table_row", "table_row": {"cells": [_rt("OS"), _rt("Supported")]}},
        {"id": "r2", "type": "table_row", "table_row": {"cells": [_rt("Linux"), _rt("yes")]}},
    ],
    "p2": [_block("c1", "to_do", rich_text=_rt("Restart"), checked=True)],
}


def _notion_page(page_id: str, title: str) -> dict:
    return {
        "id": page_id,
        "url": f"https://www.notion.so/{page_id}",
        "last_edited_time": "2026-10-02T09:00:00.000Z",
        "properties": {"Name": {"type": "title", "title": _rt(title)}},
    }


async def test_notion_database_pages_children_and_blocks() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["notion-version"] == "2022-06-28"
        path = request.url.path
        if path == "/v1/databases/db1/query":
            body = json.loads(request.content)
            if "start_cursor" not in body:
                return httpx.Response(
                    200,
                    json={
                        "results": [_notion_page("p1", "Install guide")],
                        "has_more": True,
                        "next_cursor": "c2",
                    },
                )
            return httpx.Response(200, json={"results": [], "has_more": False})
        if path == "/v1/pages/p2":
            return httpx.Response(200, json=_notion_page("p2", "Troubleshooting"))
        if path.startswith("/v1/blocks/") and path.endswith("/children"):
            block_id = path.split("/")[3]
            return httpx.Response(
                200, json={"results": NOTION_BLOCKS.get(block_id, []), "has_more": False}
            )
        return httpx.Response(404)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.notion.com",
        headers={"Notion-Version": "2022-06-28"},
    )
    connector = NotionConnector(NotionOptions(database_ids=["db1"]), client=client)

    items = [item async for item in connector.fetch()]

    assert [i.uri for i in items] == ["https://www.notion.so/p1", "https://www.notion.so/p2"]
    guide = items[0].data.decode()
    assert guide.startswith("# Install guide\n")
    assert "## Setup" in guide
    assert "- macOS\n  - Apple silicon" in guide
    assert "```bash\nbrew install nimbus\n```" in guide
    assert "| OS | Supported |\n| --- | --- |\n| Linux | yes |" in guide
    assert "Troubleshooting" not in guide
    assert "- [x] Restart" in items[1].data.decode()
    assert items[0].metadata["page_id"] == "p1"
