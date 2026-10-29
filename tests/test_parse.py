from pathlib import Path

from filings_qa.parse import chunk_section, filing_to_chunks, html_to_text, split_items

HTML = (Path(__file__).parent / "fixtures" / "mini_10k.html").read_text()


def test_html_to_text_drops_hidden_xbrl_header_and_styles():
    text = html_to_text(HTML)
    assert "hidden xbrl header" not in text
    assert "p{}" not in text
    assert "single supplier in Taiwan" in text


def test_split_items_skips_table_of_contents():
    items = split_items(html_to_text(HTML))
    assert set(items) == {"1", "1A", "7", "8"}
    # The real section, not the one-line table-of-contents entry.
    assert "single supplier in Taiwan" in items["1A"]
    assert "Risk Factors 5" not in items["1A"]
    assert "$12,345 million" in items["7"]


def test_chunk_section_respects_size_and_overlaps():
    section = "\n".join(f"Paragraph number {i} " + "x" * 80 for i in range(30))
    chunks = chunk_section(section, max_chars=500)
    assert len(chunks) > 1
    assert all(len(c) <= 500 + 120 for c in chunks)  # one overlapping paragraph allowed
    first_last_para = chunks[0].split("\n")[-1]
    assert chunks[1].startswith(first_last_para)


def test_chunk_section_hard_splits_huge_paragraph():
    chunks = chunk_section("y" * 4000, max_chars=1000, overlap_paragraphs=0)
    assert len(chunks) == 4


def test_filing_to_chunks_ids_are_stable_and_citable():
    chunks = filing_to_chunks(HTML, ticker="EXMP", fiscal_year=2024, accession="0000-24-1")
    ids = [c.source_id for c in chunks]
    assert "EXMP-2024-item1A-0" in ids
    assert len(ids) == len(set(ids))
