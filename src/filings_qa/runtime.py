"""Wire the configured components together."""

from __future__ import annotations

from filings_qa.agent import FilingsAgent
from filings_qa.config import Settings, get_settings
from filings_qa.embed import make_embedder
from filings_qa.llm import make_llm
from filings_qa.store import PgVectorStore


def build_agent(settings: Settings | None = None) -> FilingsAgent:  # pragma: no cover
    s = settings or get_settings()
    embedder = make_embedder(s.embeddings, s.embedding_model)
    store = PgVectorStore(s.database_url, dim=embedder.dim)
    llm = make_llm(s.llm_provider, s.anthropic_model)
    return FilingsAgent(llm=llm, store=store, embedder=embedder)
