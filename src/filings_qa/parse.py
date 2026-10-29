"""Turn a 10-K HTML document into item sections and citable chunks."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from filings_qa.models import Chunk

# Items we index by default: Business, Risk Factors, MD&A, Market Risk, Financial Statements.
DEFAULT_ITEMS = ("1", "1A", "7", "7A", "8")

_ITEM_HEADING = re.compile(
    r"^\s*item\s*(1a|1b|1c|1|2|3|4|5|6|7a|7|8|9a|9b|9c|9|10|11|12|13|14|15|16)\b[\s.:\-—–]*",
    re.IGNORECASE | re.MULTILINE,
)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    # Inline XBRL filings carry a hidden header with machine-readable data; drop it.
    for tag in soup.find_all(["ix:header"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = text.replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    # Collapse runs of blank lines into single paragraph breaks.
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out).strip()


def split_items(text: str, wanted: tuple[str, ...] = DEFAULT_ITEMS) -> dict[str, str]:
    """Split filing text into {item: section text}.

    Item headings appear twice: once in the table of contents (short sections) and once
    where the section really starts. For each item we keep the occurrence followed by the
    longest span, which is the real section.
    """
    matches = list(_ITEM_HEADING.finditer(text))
    best: dict[str, tuple[int, int]] = {}
    for i, m in enumerate(matches):
        item = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        if item not in best or (end - start) > (best[item][1] - best[item][0]):
            best[item] = (start, end)
    return {
        item: text[start:end].strip()
        for item, (start, end) in best.items()
        if item in wanted and text[start:end].strip()
    }


def chunk_section(text: str, max_chars: int = 1500, overlap_paragraphs: int = 1) -> list[str]:
    """Pack paragraphs into chunks of up to `max_chars`, repeating the last paragraph(s)
    of each chunk at the start of the next so a fact split across chunks stays findable."""
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 1]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for p in paragraphs:
        if len(p) > max_chars:  # very long paragraph: hard-split it
            pieces = [p[i : i + max_chars] for i in range(0, len(p), max_chars)]
        else:
            pieces = [p]
        for piece in pieces:
            if current and size + len(piece) > max_chars:
                chunks.append("\n".join(current))
                current = current[-overlap_paragraphs:] if overlap_paragraphs else []
                size = sum(len(x) for x in current)
            current.append(piece)
            size += len(piece)
    if current:
        chunks.append("\n".join(current))
    return chunks


def filing_to_chunks(
    html: str, *, ticker: str, fiscal_year: int, accession: str, max_chars: int = 1500
) -> list[Chunk]:
    sections = split_items(html_to_text(html))
    chunks: list[Chunk] = []
    for item, section in sections.items():
        for idx, text in enumerate(chunk_section(section, max_chars=max_chars)):
            chunks.append(
                Chunk(
                    ticker=ticker,
                    fiscal_year=fiscal_year,
                    accession=accession,
                    item=item,
                    index=idx,
                    text=text,
                )
            )
    return chunks
