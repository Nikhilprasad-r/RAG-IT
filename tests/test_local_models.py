"""Docling parsing and local cross-encoder reranking.

These need the optional model dependencies (`uv sync --group local-models`) and
are skipped without them. The first run downloads model weights.
"""

import io

import pytest
from test_parsers import _raw, make_pdf

from ragkit.core.config import ChunkingConfig
from ragkit.core.types import Chunk, Document, Query, ScoredChunk
from ragkit.ingest.chunking import MarkdownChunker
from ragkit.ingest.parsers import Parsers

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@pytest.fixture(scope="module")
def docling() -> Parsers:
    pytest.importorskip("docling")
    return Parsers("docling")


async def test_docling_office_formats_keep_headings_and_tables(docling: Parsers) -> None:
    import docx
    import openpyxl
    from pptx import Presentation

    document = docx.Document()
    document.add_heading("Leave policy", 1)
    document.add_paragraph("Employees receive 25 days of paid leave.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Tenure", "Extra days"
    table.cell(1, 0).text, table.cell(1, 1).text = "5 years", "2"
    docx_bytes = io.BytesIO()
    document.save(docx_bytes)

    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[1])
    slide.shapes.title.text = "Q3 Roadmap"
    slide.placeholders[1].text = "Launch SSO"
    pptx_bytes = io.BytesIO()
    deck.save(pptx_bytes)

    workbook = openpyxl.Workbook()
    for row in [("Plan", "Storage"), ("Pro", "2 TB")]:
        workbook.active.append(row)
    xlsx_bytes = io.BytesIO()
    workbook.save(xlsx_bytes)

    word = await docling.parse(_raw(docx_bytes.getvalue(), DOCX, "handbook.docx"))
    slides = await docling.parse(_raw(pptx_bytes.getvalue(), PPTX, "roadmap.pptx"))
    sheet = await docling.parse(_raw(xlsx_bytes.getvalue(), XLSX, "pricing.xlsx"))

    assert word.title == "Leave policy"
    assert "| 5 years" in word.text
    assert slides.title == "Q3 Roadmap" and "Launch SSO" in slides.text
    assert "| Pro" in sheet.text and "2 TB" in sheet.text


async def test_docling_pdf_pages_map_to_chunks(docling: Parsers) -> None:
    pdf = make_pdf(
        [
            ("Introduction", "This guide explains the platform."),
            ("Security", "All data is encrypted at rest."),
            ("Billing", "Invoices are issued monthly."),
        ]
    )

    parsed = await docling.parse(_raw(pdf, "application/pdf", "guide.pdf"))
    document = Document(
        id="d",
        source="t",
        uri="guide.pdf",
        content=parsed.text,
        title=parsed.title,
        metadata=parsed.metadata,
    )
    chunks = MarkdownChunker(ChunkingConfig(max_tokens=100, overlap_tokens=0)).chunk(document)

    assert parsed.metadata["pages"] == 3
    assert parsed.title == "Introduction"
    assert [(c.metadata["heading_path"], c.metadata["page"]) for c in chunks] == [
        (["Introduction"], 1),
        (["Security"], 2),
        (["Billing"], 3),
    ]


async def test_cross_encoder_puts_the_answer_first() -> None:
    pytest.importorskip("sentence_transformers")
    from ragkit.retrieval import CrossEncoderReranker

    texts = [
        "The Pro plan includes 2 TB of storage.",
        "A reset link is emailed to you and expires after 30 minutes.",
        "The REST API allows 600 requests per minute.",
    ]
    candidates = [
        ScoredChunk(
            chunk=Chunk(id=str(i), document_id="d", text=t, index=i), score=0.0, retriever="hybrid"
        )
        for i, t in enumerate(texts)
    ]
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L6-v2")

    ranked = await reranker.rerank(Query(text="How long is a reset link valid?"), candidates, 2)

    assert len(ranked) == 2
    assert ranked[0].chunk.id == "1"
    assert ranked[0].score > ranked[1].score
    assert ranked[0].retriever == "cross-encoder"
