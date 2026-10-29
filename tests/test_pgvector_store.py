"""Integration test against a real PostgreSQL + pgvector.

Runs when TEST_DATABASE_URL is set (CI starts a pgvector service); skipped otherwise.
    TEST_DATABASE_URL=postgresql://filings:filings@localhost:5432/filings pytest
"""

import os
from pathlib import Path

import pytest

from filings_qa.embed import HashEmbedder
from filings_qa.models import Fact
from filings_qa.parse import filing_to_chunks

DB_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")

HTML = (Path(__file__).parent / "fixtures" / "mini_10k.html").read_text()


@pytest.fixture()
def store():
    import psycopg

    from filings_qa.store import PgVectorStore

    with psycopg.connect(DB_URL, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks, facts")
    return PgVectorStore(DB_URL, dim=HashEmbedder().dim)


def test_round_trip_search_and_facts(store):
    embedder = HashEmbedder()
    chunks = filing_to_chunks(HTML, ticker="EXMP", fiscal_year=2024, accession="0000-24-1")
    store.add_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))
    store.add_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))  # idempotent

    hits = store.search(embedder.embed_query("single supplier in Taiwan"), ticker="EXMP", k=3)
    assert hits and hits[0][0].item == "1A"
    assert store.search(embedder.embed_query("supplier"), ticker="OTHER") == []

    fact = Fact("EXMP", "Revenues", 2024, 12345000000.0, "USD", "0000-24-1", "10-K", "2024-12-31")
    store.add_facts([fact])
    assert store.get_fact("EXMP", "revenue", 2024) == fact
    assert store.get_fact("EXMP", "revenue", 2023) is None
    assert store.known_tickers() == ["EXMP"]
