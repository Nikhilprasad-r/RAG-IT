"""Structure-aware chunking.

Text is split along Markdown headings first, then packed paragraph by
paragraph up to `max_tokens`. Paragraphs that are too long are split at
sentence and then word boundaries. Each chunk records its heading path and,
for paged formats, its page, and gets a short context header (document title
and section path) that is indexed with it.

Token counts are estimated at four characters per token, which is close
enough for sizing chunks without loading a tokenizer.
"""

import bisect
import math
import re
from dataclasses import dataclass

from ragkit.core.config import ChunkingConfig
from ragkit.core.types import Chunk, Document

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CHUNK_METADATA_SKIP = {"page_offsets"}


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


@dataclass
class _Block:
    text: str
    offset: int


@dataclass
class _Section:
    path: list[str]
    blocks: list[_Block]


class MarkdownChunker:
    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(self, document: Document) -> list[Chunk]:
        page_offsets: list[int] = document.metadata.get("page_offsets") or []
        shared = {k: v for k, v in document.metadata.items() if k not in _CHUNK_METADATA_SKIP}
        chunks: list[Chunk] = []
        for section in _sections(document.content):
            for text, offset in self._pack(section.blocks):
                metadata = {**shared, "heading_path": section.path}
                if page_offsets:
                    metadata["page"] = bisect.bisect_right(page_offsets, offset)
                chunks.append(
                    Chunk(
                        id=f"{document.id}:{len(chunks)}",
                        document_id=document.id,
                        collection=document.collection,
                        tenant_id=document.tenant_id,
                        index=len(chunks),
                        text=text,
                        context=_context(document.title, section.path),
                        metadata=metadata,
                        acl=document.acl,
                    )
                )
        return chunks

    def _pack(self, blocks: list[_Block]) -> list[tuple[str, int]]:
        limit, overlap = self.config.max_tokens, self.config.overlap_tokens
        pieces = [
            _Block(part, block.offset)
            for block in blocks
            for part in (
                [block.text] if estimate_tokens(block.text) <= limit else _split(block.text, limit)
            )
        ]
        packed: list[tuple[str, int]] = []
        current: list[str] = []
        start = 0
        for piece in pieces:
            candidate = "\n\n".join([*current, piece.text])
            if current and estimate_tokens(candidate) > limit:
                packed.append(("\n\n".join(current), start))
                tail = _tail(packed[-1][0], overlap)
                current = [tail] if tail and estimate_tokens(tail + piece.text) <= limit else []
                start = piece.offset
            if not current:
                start = piece.offset
            current.append(piece.text)
        if current:
            packed.append(("\n\n".join(current), start))
        return packed


def _sections(content: str) -> list[_Section]:
    sections = [_Section([], [])]
    headings: list[tuple[int, str]] = []
    paragraph: list[str] = []
    paragraph_start = 0
    in_fence = False
    offset = 0

    def flush() -> None:
        text = "\n".join(paragraph).strip()
        if text:
            sections[-1].blocks.append(_Block(text, paragraph_start))
        paragraph.clear()

    for line in content.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        if _FENCE.match(stripped):
            in_fence = not in_fence
        heading = None if in_fence else _HEADING.match(stripped)
        if heading:
            flush()
            level, title = len(heading.group(1)), heading.group(2)
            headings = [(lvl, t) for lvl, t in headings if lvl < level] + [(level, title)]
            sections.append(_Section([t for _, t in headings], []))
        elif not stripped.strip() and not in_fence:
            flush()
        else:
            if not paragraph:
                paragraph_start = offset
            paragraph.append(stripped)
        offset += len(line)
    flush()
    return [s for s in sections if s.blocks]


def _split(text: str, limit: int) -> list[str]:
    """Split an oversized block at sentence boundaries, then at word boundaries."""
    parts: list[str] = []
    current = ""
    for unit in _SENTENCE_END.split(text):
        if estimate_tokens(unit) > limit:
            if current:
                parts.append(current)
                current = ""
            parts.extend(_split_words(unit, limit))
            continue
        candidate = f"{current} {unit}".strip()
        if current and estimate_tokens(candidate) > limit:
            parts.append(current)
            candidate = unit
        current = candidate
    if current:
        parts.append(current)
    return parts


def _split_words(text: str, limit: int) -> list[str]:
    parts, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and estimate_tokens(candidate) > limit:
            parts.append(current)
            candidate = word
        current = candidate
    if current:
        parts.append(current)
    return parts


def _tail(text: str, tokens: int) -> str:
    if tokens <= 0:
        return ""
    words = text.split()
    tail: list[str] = []
    for word in reversed(words):
        if estimate_tokens(" ".join([word, *tail])) > tokens:
            break
        tail.insert(0, word)
    return " ".join(tail)


def _context(title: str, path: list[str]) -> str:
    parts = [title] if title else []
    parts += [p for p in path if p != title]
    return " > ".join(parts)
