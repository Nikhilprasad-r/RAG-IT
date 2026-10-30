"""Turn raw bytes into Markdown-flavoured text.

The basic parsers cover text, Markdown, code, HTML, PDF, DOCX, CSV and JSON
with light dependencies. Set `parser: docling` on a collection to use Docling's
layout-aware conversion (tables, multi-column PDFs, PPTX, XLSX, scanned pages)
for the formats it supports; it needs `ragkit-ingest[docling]`.
"""

import asyncio
import csv
import io
import json
import re
from collections.abc import Awaitable, Callable
from pathlib import PurePosixPath

from bs4 import BeautifulSoup, Tag

from ragkit.core.errors import ConfigError, RagkitError
from ragkit.core.types import Modality, ParsedContent, RawDocument


class UnsupportedFormatError(RagkitError):
    """The item's format has no parser; ingestion skips it."""


ParseFn = Callable[[RawDocument], Awaitable[ParsedContent]]

_CODE_TYPES = {
    "text/x-python",
    "text/x-script.python",
    "application/javascript",
    "text/javascript",
    "application/typescript",
    "text/x-java",
    "text/x-go",
    "text/x-rust",
    "text/x-c",
    "text/x-sh",
    "application/x-sh",
    "application/sql",
    "application/x-yaml",
    "application/yaml",
    "text/yaml",
    "application/toml",
}
_MARKDOWN_HEADING = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_ANY_HEADING = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)


async def parse_text(raw: RawDocument) -> ParsedContent:
    text = _decode(raw.data)
    title = ""
    if raw.mime_type in ("text/markdown", "text/x-markdown"):
        match = _MARKDOWN_HEADING.search(text)
        title = match.group(1).strip() if match else ""
    modality = Modality.CODE if raw.mime_type in _CODE_TYPES else Modality.TEXT
    return ParsedContent(text=text, title=title or _filename(raw), modality=modality)


async def parse_html(raw: RawDocument) -> ParsedContent:
    soup = BeautifulSoup(raw.data, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    lines: list[str] = []
    for element in root.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "blockquote", "td", "th"]
    ):
        if not isinstance(element, Tag) or element.find_parent(
            ["li", "pre", "td", "th", "blockquote"]
        ):
            continue  # nested inside an element we already emit
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        name = element.name
        if name and name[0] == "h" and name[1:].isdigit():
            lines.append(f"\n{'#' * int(name[1:])} {text}\n")
        elif name == "li":
            lines.append(f"- {text}")
        elif name == "pre":
            lines.append(f"\n```\n{element.get_text()}\n```\n")
        else:
            lines.append(f"\n{text}\n")
    body = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if not title:
        h1 = root.find("h1")
        title = h1.get_text(strip=True) if h1 else _filename(raw)
    return ParsedContent(text=body, title=title)


async def parse_pdf(raw: RawDocument) -> ParsedContent:
    def extract() -> ParsedContent:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw.data))
        pages, offsets, position = [], [], 0
        for page in reader.pages:
            text = (page.extract_text() or "").strip()
            offsets.append(position)
            pages.append(text)
            position += len(text) + 2
        title = (reader.metadata.title if reader.metadata else None) or _filename(raw)
        return ParsedContent(
            text="\n\n".join(pages),
            title=str(title),
            metadata={"page_offsets": offsets, "pages": len(pages)},
        )

    return await asyncio.to_thread(extract)


async def parse_docx(raw: RawDocument) -> ParsedContent:
    def extract() -> ParsedContent:
        import docx

        document = docx.Document(io.BytesIO(raw.data))
        lines = []
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            style = paragraph.style.name if paragraph.style else ""
            level = style.removeprefix("Heading ").strip()
            if style.startswith("Heading") and level.isdigit():
                lines.append(f"\n{'#' * min(int(level), 6)} {text}\n")
            elif style == "Title":
                lines.append(f"\n# {text}\n")
            else:
                lines.append(f"\n{text}\n")
        for table in document.tables:
            rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
            lines.append("\n" + _markdown_table(rows) + "\n")
        title = document.core_properties.title or _filename(raw)
        return ParsedContent(text="\n".join(lines).strip(), title=title)

    return await asyncio.to_thread(extract)


async def parse_csv(raw: RawDocument) -> ParsedContent:
    """One block per row as `column: value` lines, so each row is retrievable on its own."""
    reader = csv.DictReader(io.StringIO(_decode(raw.data)))
    blocks = [
        "\n".join(f"{key}: {value}" for key, value in row.items() if key and value)
        for row in reader
    ]
    return ParsedContent(
        text="\n\n".join(b for b in blocks if b), title=_filename(raw), modality=Modality.TABLE
    )


async def parse_json(raw: RawDocument) -> ParsedContent:
    try:
        data = json.loads(raw.data)
    except json.JSONDecodeError as exc:
        raise UnsupportedFormatError(f"Invalid JSON: {exc}") from exc
    return ParsedContent(text=json.dumps(data, indent=2, ensure_ascii=False), title=_filename(raw))


BASIC_PARSERS: dict[str, ParseFn] = {
    "text/plain": parse_text,
    "text/markdown": parse_text,
    "text/x-markdown": parse_text,
    "text/x-rst": parse_text,
    **dict.fromkeys(_CODE_TYPES, parse_text),
    "text/html": parse_html,
    "application/xhtml+xml": parse_html,
    "application/pdf": parse_pdf,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": parse_docx,
    "text/csv": parse_csv,
    "application/json": parse_json,
}

DOCLING_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "image/png",
    "image/jpeg",
    "image/tiff",
}


class Parsers:
    """Chooses a parser by MIME type."""

    def __init__(self, engine: str = "basic") -> None:
        self._docling = DoclingParser() if engine == "docling" else None

    async def parse(self, raw: RawDocument) -> ParsedContent:
        mime = raw.mime_type.split(";")[0].strip().lower()
        raw = raw.model_copy(update={"mime_type": mime})
        if self._docling and mime in DOCLING_TYPES:
            return await self._docling.parse(raw)
        if parser := BASIC_PARSERS.get(mime):
            return await parser(raw)
        if mime.startswith("text/") or _looks_like_text(raw.data):
            return await parse_text(raw)
        raise UnsupportedFormatError(f"No parser for {mime}.")


class DoclingParser:
    """Layout-aware conversion with Docling's models.

    Loading the models takes several seconds, so one converter is reused, and
    conversions run one at a time because the converter is not documented as
    thread-safe (the models already use every core or the GPU).
    """

    mime_types = tuple(sorted(DOCLING_TYPES))

    def __init__(self) -> None:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise ConfigError("parser: docling needs `ragkit-ingest[docling]` installed.") from exc
        self._converter = DocumentConverter()
        self._lock = asyncio.Lock()

    async def parse(self, raw: RawDocument) -> ParsedContent:
        async with self._lock:
            return await asyncio.to_thread(self._convert, raw)

    def _convert(self, raw: RawDocument) -> ParsedContent:
        from docling.datamodel.base_models import DocumentStream

        name = PurePosixPath(raw.uri).name or "document"
        document = self._converter.convert(
            DocumentStream(name=name, stream=io.BytesIO(raw.data))
        ).document
        pages = sorted(document.pages)
        metadata: dict[str, object] = {}
        if len(pages) > 1:
            # Export page by page so chunks can cite the page they came from.
            texts = [document.export_to_markdown(page_no=page).strip() for page in pages]
            offsets, position = [], 0
            for text in texts:
                offsets.append(position)
                position += len(text) + 2
            text = "\n\n".join(texts)
            metadata = {"page_offsets": offsets, "pages": len(pages)}
        else:
            text = document.export_to_markdown()
        return ParsedContent(
            text=text, title=_first_heading(text) or _filename(raw), metadata=metadata
        )


def _first_heading(text: str) -> str:
    match = _ANY_HEADING.search(text)
    return match.group(1).strip() if match else ""


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _looks_like_text(data: bytes) -> bool:
    sample = data[:4096]
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError as exc:
        return exc.start > len(sample) - 4  # a multi-byte character cut by the sample
    return True


def _filename(raw: RawDocument) -> str:
    return PurePosixPath(raw.uri.rstrip("/")).name or raw.uri


def _markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join(lines)
