"""Ingestion: download 10-Ks and their reported facts, chunk, embed and store them."""

from __future__ import annotations

from filings_qa.edgar import EdgarClient
from filings_qa.embed import Embedder
from filings_qa.parse import filing_to_chunks
from filings_qa.store import Store
from filings_qa.xbrl import COMPANYFACTS_URL, extract_metrics


def ingest_ticker(
    client: EdgarClient, store: Store, embedder: Embedder, ticker: str, years: int = 2
) -> dict:
    """Ingest the last `years` annual reports for one company. Returns a small summary."""
    refs = client.annual_filings(ticker, limit=years)
    if not refs:
        return {"ticker": ticker, "filings": 0, "chunks": 0, "facts": 0}
    companyfacts = client.get_json(
        COMPANYFACTS_URL.format(cik=refs[0].cik), f"companyfacts_{refs[0].cik}.json"
    )
    n_chunks = n_facts = 0
    for ref in refs:
        chunks = filing_to_chunks(
            client.download(ref),
            ticker=ref.ticker,
            fiscal_year=ref.fiscal_year,
            accession=ref.accession,
        )
        if chunks:
            store.add_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))
        facts = extract_metrics(
            companyfacts,
            ticker=ref.ticker,
            fiscal_year=ref.fiscal_year,
            accession=ref.accession,
            report_date=ref.report_date,
        )
        store.add_facts(list(facts.values()))
        n_chunks += len(chunks)
        n_facts += len(facts)
    return {"ticker": ticker, "filings": len(refs), "chunks": n_chunks, "facts": n_facts}
