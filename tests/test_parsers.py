import io

import docx
import pytest

from ragkit.core.types import Modality, RawDocument
from ragkit.ingest.parsers import Parsers, UnsupportedFormatError


def _raw(data: bytes | str, mime: str, uri: str = "docs/file") -> RawDocument:
    return RawDocument(
        source="test",
        uri=uri,
        data=data.encode() if isinstance(data, str) else data,
        mime_type=mime,
    )


def make_pdf(pages: list[str | tuple[str, str]]) -> bytes:
    """A minimal valid PDF. Each page is a line of text, or a (bold heading, body) pair."""
    page_ids = [5 + 2 * i for i in range(len(pages))]
    objects = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] "
        f"/Count {len(pages)} >>",
        3: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        4: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
    }
    for page_id, page in zip(page_ids, pages, strict=True):
        if isinstance(page, tuple):
            heading, body = page
            stream = (
                f"BT /F2 18 Tf 72 720 Td ({heading}) Tj ET\nBT /F1 11 Tf 72 690 Td ({body}) Tj ET"
            )
        else:
            stream = f"BT /F1 12 Tf 72 720 Td ({page}) Tj ET"
        objects[page_id] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {page_id + 1} 0 R >>"
        )
        objects[page_id + 1] = f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"
    out, offsets = b"%PDF-1.4\n", {}
    for number in sorted(objects):
        offsets[number] = len(out)
        out += f"{number} 0 obj\n{objects[number]}\nendobj\n".encode()
    xref, size = len(out), max(objects) + 1
    out += f"xref\n0 {size}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offsets[n]:010d} 00000 n \n".encode() for n in range(1, size))
    out += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


async def test_markdown_title_comes_from_first_heading() -> None:
    parsed = await Parsers().parse(_raw("Intro\n\n# Real Title\n\nBody", "text/markdown"))

    assert parsed.title == "Real Title"
    assert "Body" in parsed.text


async def test_html_keeps_structure_and_drops_chrome() -> None:
    html = """<html><head><title>Docs | Nimbus</title><script>var x=1;</script></head>
    <body><nav>Home | Pricing</nav>
    <main><h1>Getting started</h1><p>Install the CLI.</p>
    <h2>Configure</h2><ul><li>Set the <b>token</b></li><li>Pick a region</li></ul>
    <pre>nimbus init</pre></main><footer>© Nimbus</footer></body></html>"""

    parsed = await Parsers().parse(_raw(html, "text/html; charset=utf-8"))

    assert parsed.title == "Docs | Nimbus"
    assert "# Getting started" in parsed.text
    assert "## Configure" in parsed.text
    assert "- Set the token" in parsed.text
    assert "nimbus init" in parsed.text
    for chrome in ("Home | Pricing", "var x", "© Nimbus"):
        assert chrome not in parsed.text


async def test_pdf_text_and_page_offsets() -> None:
    parsed = await Parsers().parse(_raw(make_pdf(["First page", "Second page"]), "application/pdf"))

    assert "First page" in parsed.text and "Second page" in parsed.text
    offsets = parsed.metadata["page_offsets"]
    assert parsed.metadata["pages"] == 2
    assert parsed.text[offsets[1] :].startswith("Second page")


async def test_docx_headings_paragraphs_and_tables() -> None:
    document = docx.Document()
    document.add_heading("Handbook", 0)
    document.add_heading("Leave policy", 1)
    document.add_paragraph("Employees get 25 days of leave.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Plan", "Days"
    table.cell(1, 0).text, table.cell(1, 1).text = "Standard", "25"
    buffer = io.BytesIO()
    document.save(buffer)

    parsed = await Parsers().parse(
        _raw(
            buffer.getvalue(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    )

    assert "# Handbook" in parsed.text
    assert "# Leave policy" in parsed.text
    assert "Employees get 25 days of leave." in parsed.text
    assert "| Standard | 25 |" in parsed.text


async def test_csv_rows_become_blocks() -> None:
    parsed = await Parsers().parse(_raw("sku,name,price\nA1,Lamp,20\nB2,Desk,150\n", "text/csv"))

    assert parsed.modality is Modality.TABLE
    assert parsed.text.split("\n\n") == [
        "sku: A1\nname: Lamp\nprice: 20",
        "sku: B2\nname: Desk\nprice: 150",
    ]


async def test_json_is_pretty_printed() -> None:
    parsed = await Parsers().parse(_raw('{"a": {"b": 1}}', "application/json"))

    assert parsed.text == '{\n  "a": {\n    "b": 1\n  }\n}'


async def test_unknown_text_is_read_and_binary_is_rejected() -> None:
    text = await Parsers().parse(_raw("plain words", "application/octet-stream"))
    assert text.text == "plain words"

    with pytest.raises(UnsupportedFormatError):
        await Parsers().parse(_raw(b"\x89PNG\r\n\x1a\n\x00\x00", "image/png"))


async def test_code_files_are_tagged_as_code() -> None:
    parsed = await Parsers().parse(_raw("def f():\n    return 1\n", "text/x-python", "a/f.py"))

    assert parsed.modality is Modality.CODE
    assert parsed.title == "f.py"
