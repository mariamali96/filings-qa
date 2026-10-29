"""Storage for chunks (with embeddings) and reported facts.

`PgVectorStore` is the real backend (PostgreSQL + pgvector). `InMemoryStore` has the
same interface and is used by the offline tests and small demos.
"""

from __future__ import annotations

import math
from typing import Protocol

from filings_qa.models import Chunk, Fact
from filings_qa.xbrl import METRICS


class Store(Protocol):
    def add_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def add_facts(self, facts: list[Fact]) -> None: ...

    def search(
        self,
        query_vector: list[float],
        *,
        ticker: str | None = None,
        fiscal_year: int | None = None,
        k: int = 6,
    ) -> list[tuple[Chunk, float]]: ...

    def get_fact(self, ticker: str, metric: str, fiscal_year: int) -> Fact | None: ...

    def known_tickers(self) -> list[str]: ...


def _metric_concepts(metric: str) -> list[str]:
    return METRICS.get(metric, [metric])


class InMemoryStore:
    def __init__(self) -> None:
        self._chunks: list[tuple[Chunk, list[float]]] = []
        self._facts: dict[tuple[str, str, int], Fact] = {}

    def add_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self._chunks.extend(zip(chunks, vectors, strict=True))

    def add_facts(self, facts: list[Fact]) -> None:
        for f in facts:
            self._facts[(f.ticker, f.concept, f.fiscal_year)] = f

    def search(self, query_vector, *, ticker=None, fiscal_year=None, k=6):
        def cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b, strict=True))
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(y * y for y in b)) or 1.0
            return dot / (na * nb)

        scored = [
            (chunk, cosine(query_vector, vec))
            for chunk, vec in self._chunks
            if (ticker is None or chunk.ticker == ticker)
            and (fiscal_year is None or chunk.fiscal_year == fiscal_year)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def get_fact(self, ticker: str, metric: str, fiscal_year: int) -> Fact | None:
        for concept in _metric_concepts(metric):
            fact = self._facts.get((ticker, concept, fiscal_year))
            if fact:
                return fact
        return None

    def known_tickers(self) -> list[str]:
        return sorted(
            {c.ticker for c, _ in self._chunks} | {f.ticker for f in self._facts.values()}
        )


SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    source_id   TEXT PRIMARY KEY,
    ticker      TEXT NOT NULL,
    fiscal_year INT  NOT NULL,
    accession   TEXT NOT NULL,
    item        TEXT NOT NULL,
    idx         INT  NOT NULL,
    text        TEXT NOT NULL,
    embedding   vector({dim}) NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_filter_idx ON chunks (ticker, fiscal_year);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE TABLE IF NOT EXISTS facts (
    ticker      TEXT NOT NULL,
    concept     TEXT NOT NULL,
    fiscal_year INT  NOT NULL,
    value       DOUBLE PRECISION NOT NULL,
    unit        TEXT NOT NULL,
    accession   TEXT NOT NULL,
    form        TEXT NOT NULL,
    period_end  TEXT NOT NULL,
    PRIMARY KEY (ticker, concept, fiscal_year)
);
"""


class PgVectorStore:  # pragma: no cover - needs a running database
    def __init__(self, database_url: str, dim: int):
        import psycopg
        from pgvector.psycopg import register_vector

        self._conn = psycopg.connect(database_url, autocommit=True)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)
        self._conn.execute(SCHEMA.format(dim=dim))

    def add_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        import numpy as np

        with self._conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (source_id, ticker, fiscal_year, accession, item, idx, text,
                                    embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source_id) DO UPDATE
                    SET text = EXCLUDED.text, embedding = EXCLUDED.embedding
                """,
                [
                    (
                        c.source_id,
                        c.ticker,
                        c.fiscal_year,
                        c.accession,
                        c.item,
                        c.index,
                        c.text,
                        np.array(v, dtype=np.float32),
                    )
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )

    def add_facts(self, facts: list[Fact]) -> None:
        with self._conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO facts (ticker, concept, fiscal_year, value, unit, accession, form,
                                   period_end)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, concept, fiscal_year) DO UPDATE SET value = EXCLUDED.value
                """,
                [
                    (
                        f.ticker,
                        f.concept,
                        f.fiscal_year,
                        f.value,
                        f.unit,
                        f.accession,
                        f.form,
                        f.period_end,
                    )
                    for f in facts
                ],
            )

    def search(self, query_vector, *, ticker=None, fiscal_year=None, k=6):
        import numpy as np

        where, params = [], []
        if ticker:
            where.append("ticker = %s")
            params.append(ticker)
        if fiscal_year:
            where.append("fiscal_year = %s")
            params.append(fiscal_year)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        q = np.array(query_vector, dtype=np.float32)
        rows = self._conn.execute(
            f"""
            SELECT ticker, fiscal_year, accession, item, idx, text, 1 - (embedding <=> %s) AS score
            FROM chunks {clause}
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            [q, *params, q, k],
        ).fetchall()
        return [
            (
                Chunk(
                    ticker=r[0], fiscal_year=r[1], accession=r[2], item=r[3], index=r[4], text=r[5]
                ),
                float(r[6]),
            )
            for r in rows
        ]

    def get_fact(self, ticker: str, metric: str, fiscal_year: int) -> Fact | None:
        for concept in _metric_concepts(metric):
            row = self._conn.execute(
                "SELECT ticker, concept, fiscal_year, value, unit, accession, form, period_end "
                "FROM facts WHERE ticker = %s AND concept = %s AND fiscal_year = %s",
                [ticker, concept, fiscal_year],
            ).fetchone()
            if row:
                return Fact(*row)
        return None

    def known_tickers(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT ticker FROM chunks UNION SELECT DISTINCT ticker FROM facts"
        ).fetchall()
        return sorted(r[0] for r in rows)
