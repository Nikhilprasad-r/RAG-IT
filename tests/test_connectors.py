import json
from pathlib import Path

import httpx
import pytest

from ragkit.core.config import SourceConfig
from ragkit.core.errors import ConfigError
from ragkit.ingest.connectors import (
    FilesConnector,
    GitHubConnector,
    WebConnector,
    create_connector,
    glob_match,
)
from ragkit.ingest.connectors.github import GitHubOptions
from ragkit.ingest.connectors.web import WebOptions


@pytest.mark.parametrize(
    ("path", "pattern", "expected"),
    [
        ("README.md", "**/*.md", True),
        ("docs/a/b.md", "**/*.md", True),
        ("docs/a/b.md", "docs/*.md", False),
        ("docs/b.md", "docs/*.md", True),
        ("docs/b.mdx", "**/*.md", False),
        ("src/x.py", "src/**", True),
    ],
)
def test_glob_match(path: str, pattern: str, expected: bool) -> None:
    assert glob_match(path, [pattern]) is expected


async def test_files_connector_filters_and_describes_files(tmp_path: Path) -> None:
    for name in ("data/a.md", "data/sub/b.txt", "data/skip.log", "data/.hidden/c.md"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(name)
    source = SourceConfig(type="files", path="data", include=["**/*.md", "**/*.txt"])

    connector = create_connector(source, root=tmp_path)
    items = [item async for item in connector.fetch()]

    assert connector.name == "files:data"
    assert [i.uri for i in items] == ["a.md", "sub/b.txt"]
    assert items[0].mime_type == "text/markdown"
    assert items[0].data == b"data/a.md"
    assert items[1].metadata["path"] == "sub/b.txt"


async def test_files_connector_reports_missing_directory(tmp_path: Path) -> None:
    connector = FilesConnector.from_options({"path": "nope"}, root=tmp_path)

    with pytest.raises(ConfigError, match="does not exist"):
        [item async for item in connector.fetch()]


def test_unknown_source_options_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="pathh"):
        create_connector(SourceConfig(type="files", pathh="data"), root=tmp_path)


def _site(pages: dict[str, tuple[int, str]], robots: str = "") -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/robots.txt"):
            return httpx.Response(200, text=robots) if robots else httpx.Response(404)
        if url not in pages:
            return httpx.Response(404)
        status, body = pages[url]
        if status == 0:
            raise httpx.ConnectTimeout("timed out", request=request)
        return httpx.Response(status, text=body, headers={"content-type": "text/html"})

    return httpx.MockTransport(handler)


async def test_web_crawler_follows_links_within_scope_and_depth() -> None:
    base = "https://docs.example.com/guide/"
    pages = {
        base: (
            200,
            '<a href="install">i</a><a href="/blog/x">b</a><a href="https://other.com">o</a>',
        ),
        base + "install": (200, '<a href="install#linux">l</a><a href="deep">d</a>'),
        base + "deep": (200, '<a href="deeper">x</a>'),
        base + "deeper": (200, "too deep"),
        "https://docs.example.com/blog/x": (200, "out of scope"),
    }
    client = httpx.AsyncClient(transport=_site(pages))
    connector = WebConnector(WebOptions(urls=[base], max_depth=2), client=client)

    uris = sorted([item.uri async for item in connector.fetch()])

    assert uris == [base, base + "deep", base + "install"]
    assert connector.complete


async def test_web_crawler_honours_robots_and_page_limit() -> None:
    base = "https://site.test/"
    pages = {base: (200, '<a href="/a">a</a><a href="/private/b">b</a><a href="/c">c</a>')}
    pages |= {f"{base}{p}": (200, p) for p in ("a", "private/b", "c")}
    client = httpx.AsyncClient(transport=_site(pages, robots="User-agent: *\nDisallow: /private/"))

    uris = [
        i.uri
        async for i in WebConnector(WebOptions(urls=[base], max_pages=3), client=client).fetch()
    ]

    assert base + "private/b" not in uris
    assert len(uris) <= 3


async def test_web_crawler_marks_listing_incomplete_on_network_errors() -> None:
    base = "https://site.test/"
    pages = {base: (200, '<a href="/ok">ok</a><a href="/down">down</a>'), base + "ok": (200, "ok")}
    pages[base + "down"] = (0, "")
    connector = WebConnector(
        WebOptions(urls=[base]), client=httpx.AsyncClient(transport=_site(pages))
    )

    uris = [item.uri async for item in connector.fetch()]

    assert base + "ok" in uris
    assert connector.complete is False


async def test_github_connector_lists_tree_and_downloads_matching_files() -> None:
    tree = {
        "tree": [
            {"path": "README.md", "type": "blob", "size": 10},
            {"path": "docs/guide.md", "type": "blob", "size": 10},
            {"path": "src/main.py", "type": "blob", "size": 10},
            {"path": "docs", "type": "tree"},
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://api.github.com/repos/acme/kb":
            return httpx.Response(200, json={"default_branch": "main"})
        if url.startswith("https://api.github.com/repos/acme/kb/git/trees/main"):
            assert request.url.params["recursive"] == "1"
            return httpx.Response(200, content=json.dumps(tree))
        if url.startswith("https://raw.githubusercontent.com/acme/kb/main/"):
            return httpx.Response(200, text=f"content of {url.rsplit('/main/', 1)[1]}")
        return httpx.Response(404)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://api.github.com"
    )
    connector = GitHubConnector(GitHubOptions(repo="acme/kb"), client=client)

    items = sorted([i async for i in connector.fetch()], key=lambda i: i.uri)

    assert [i.metadata["path"] for i in items] == ["README.md", "docs/guide.md"]
    assert items[1].uri == "https://github.com/acme/kb/blob/main/docs/guide.md"
    assert items[1].data == b"content of docs/guide.md"
    assert connector.complete


async def test_custom_connector_by_import_path(tmp_path: Path) -> None:
    (tmp_path / "my_source.py").write_text(
        "from ragkit.core.types import RawDocument\n"
        "class Notes:\n"
        "    def __init__(self, items):\n"
        "        self.name = 'notes'\n"
        "        self.items = items\n"
        "    async def fetch(self):\n"
        "        for i, text in enumerate(self.items):\n"
        "            yield RawDocument(source=self.name, uri=f'note/{i}', data=text.encode(),\n"
        "                              mime_type='text/plain')\n"
    )
    source = SourceConfig(type="my_source:Notes", items=["a", "b"])

    connector = create_connector(source, root=tmp_path)

    assert [i.uri async for i in connector.fetch()] == ["note/0", "note/1"]
