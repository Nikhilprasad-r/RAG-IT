from ragkit.core.config import ChunkingConfig
from ragkit.core.types import Document
from ragkit.ingest.chunking import MarkdownChunker, estimate_tokens


def _doc(content: str, **kwargs: object) -> Document:
    return Document(id="d1", source="s", uri="u", content=content, title="Guide", **kwargs)


def test_sections_follow_heading_hierarchy() -> None:
    content = (
        "# Guide\nIntro.\n\n## Install\nRun it.\n\n### Linux\nUse apt.\n\n## Usage\nCall it.\n"
    )

    chunks = MarkdownChunker().chunk(_doc(content))

    assert [c.metadata["heading_path"] for c in chunks] == [
        ["Guide"],
        ["Guide", "Install"],
        ["Guide", "Install", "Linux"],
        ["Guide", "Usage"],
    ]
    assert chunks[2].context == "Guide > Install > Linux"
    assert chunks[2].text == "Use apt."
    assert [c.index for c in chunks] == [0, 1, 2, 3]
    assert chunks[0].id == "d1:0"


def test_paragraphs_are_packed_up_to_the_limit() -> None:
    paragraphs = [f"Paragraph {i} " + "word " * 30 for i in range(6)]
    config = ChunkingConfig(max_tokens=100, overlap_tokens=0)

    chunks = MarkdownChunker(config).chunk(_doc("\n\n".join(paragraphs)))

    assert 1 < len(chunks) < len(paragraphs)
    assert all(estimate_tokens(c.text) <= 100 for c in chunks)
    joined = " ".join(c.text for c in chunks)
    assert all(f"Paragraph {i}" in joined for i in range(6))


def test_oversized_paragraph_is_split_at_sentences() -> None:
    long = " ".join(f"Sentence number {i} is here." for i in range(80))
    config = ChunkingConfig(max_tokens=60, overlap_tokens=0)

    chunks = MarkdownChunker(config).chunk(_doc(long))

    assert len(chunks) > 3
    assert all(estimate_tokens(c.text) <= 60 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)


def test_overlap_repeats_the_tail_of_the_previous_chunk() -> None:
    paragraphs = [f"Block {i} " + " ".join(f"w{i}x{j}" for j in range(40)) for i in range(4)]
    config = ChunkingConfig(max_tokens=100, overlap_tokens=15)

    chunks = MarkdownChunker(config).chunk(_doc("\n\n".join(paragraphs)))

    assert len(chunks) > 1
    last_word = chunks[0].text.split()[-1]
    assert chunks[1].text.startswith(chunks[1].text.split()[0])
    assert last_word in chunks[1].text


def test_headings_inside_code_fences_are_not_sections() -> None:
    content = "# Doc\n\n```bash\n# not a heading\necho hi\n```\n"

    chunks = MarkdownChunker().chunk(_doc(content))

    assert len(chunks) == 1
    assert "# not a heading" in chunks[0].text
    assert chunks[0].metadata["heading_path"] == ["Doc"]


def test_pages_are_mapped_from_page_offsets() -> None:
    pages = ["Page one text. " * 20, "Page two text. " * 20, "Page three text. " * 20]
    content = "\n\n".join(p.strip() for p in pages)
    offsets = [0, len(pages[0].strip()) + 2, len(pages[0].strip()) + len(pages[1].strip()) + 4]
    config = ChunkingConfig(max_tokens=100, overlap_tokens=0)

    chunks = MarkdownChunker(config).chunk(_doc(content, metadata={"page_offsets": offsets}))

    assert [c.metadata["page"] for c in chunks] == [1, 2, 3]
    assert all("page_offsets" not in c.metadata for c in chunks)


def test_document_acl_and_collection_are_copied_to_chunks() -> None:
    doc = _doc("Hello there.", acl=["team-a"], collection="kb", metadata={"path": "a.md"})

    (chunk,) = MarkdownChunker().chunk(doc)

    assert chunk.acl == ["team-a"]
    assert chunk.collection == "kb"
    assert chunk.metadata["path"] == "a.md"


def test_empty_document_has_no_chunks() -> None:
    assert MarkdownChunker().chunk(_doc("   \n\n# Only a heading\n")) == []
