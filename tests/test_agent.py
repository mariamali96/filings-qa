"""The agent end to end, offline: scripted LLM, in-memory store, hashing embedder."""

import json
from pathlib import Path

from filings_qa.agent import ANSWER_SYSTEM, FilingsAgent
from filings_qa.embed import HashEmbedder
from filings_qa.llm import FakeLLM
from filings_qa.models import Fact
from filings_qa.parse import filing_to_chunks
from filings_qa.store import InMemoryStore

HTML = (Path(__file__).parent / "fixtures" / "mini_10k.html").read_text()
REVENUE = Fact("EXMP", "Revenues", 2024, 12345000000.0, "USD", "0000-24-1", "10-K", "2024-12-31")


def make_store() -> tuple[InMemoryStore, HashEmbedder]:
    store, embedder = InMemoryStore(), HashEmbedder()
    chunks = filing_to_chunks(HTML, ticker="EXMP", fiscal_year=2024, accession="0000-24-1")
    store.add_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))
    store.add_facts([REVENUE])
    return store, embedder


def test_numeric_question_uses_the_xbrl_tool_and_passes_verification():
    store, embedder = make_store()

    def respond(system, user):
        if system != ANSWER_SYSTEM:
            return {
                "ticker": "EXMP",
                "fiscal_year": 2024,
                "metrics": ["revenue"],
                "search_query": None,
            }
        evidence = {e["id"] for e in json.loads(user)["evidence"]}
        assert evidence == {"EXMP-2024-xbrl-Revenues"}
        return {
            "refused": False,
            "claims": [
                {"text": "Revenue was $12,345 million.", "citations": ["EXMP-2024-xbrl-Revenues"]}
            ],
        }

    answer, evidence, plan = FilingsAgent(FakeLLM(respond), store, embedder).ask("Revenue 2024?")
    assert not answer.issues and not answer.refused
    assert "12,345" in answer.text


def test_text_question_retrieves_risk_factor_passage():
    store, embedder = make_store()

    def respond(system, user):
        if system != ANSWER_SYSTEM:
            return {
                "ticker": "EXMP",
                "fiscal_year": 2024,
                "metrics": [],
                "search_query": "dependence on a single supplier in Taiwan",
            }
        ids = [e["id"] for e in json.loads(user)["evidence"]]
        assert any("item1A" in i for i in ids)
        cite = next(i for i in ids if "item1A" in i)
        return {
            "refused": False,
            "claims": [{"text": "It depends on a single supplier in Taiwan.", "citations": [cite]}],
        }

    answer, _, _ = FilingsAgent(FakeLLM(respond), store, embedder).ask("Supply risk?")
    assert not answer.issues


def test_hallucinated_number_triggers_one_retry_then_is_reported():
    store, embedder = make_store()
    llm_calls = {"answer": 0}

    def respond(system, user):
        if system != ANSWER_SYSTEM:
            return {
                "ticker": "EXMP",
                "fiscal_year": 2024,
                "metrics": ["revenue"],
                "search_query": None,
            }
        llm_calls["answer"] += 1
        if llm_calls["answer"] == 2:
            assert "problems_with_previous_answer" in json.loads(user)
        return {
            "refused": False,
            "claims": [
                {"text": "Revenue was $14.2 billion.", "citations": ["EXMP-2024-xbrl-Revenues"]}
            ],
        }

    answer, _, _ = FilingsAgent(FakeLLM(respond), store, embedder).ask("Revenue 2024?")
    assert llm_calls["answer"] == 2  # exactly one retry
    assert answer.issues and "does not appear" in answer.issues[0]


def test_retry_can_fix_the_answer():
    store, embedder = make_store()
    llm_calls = {"answer": 0}

    def respond(system, user):
        if system != ANSWER_SYSTEM:
            return {
                "ticker": "EXMP",
                "fiscal_year": 2024,
                "metrics": ["revenue"],
                "search_query": None,
            }
        llm_calls["answer"] += 1
        value = "$14.2 billion" if llm_calls["answer"] == 1 else "$12,345 million"
        return {
            "refused": False,
            "claims": [{"text": f"Revenue was {value}.", "citations": ["EXMP-2024-xbrl-Revenues"]}],
        }

    answer, _, _ = FilingsAgent(FakeLLM(respond), store, embedder).ask("Revenue 2024?")
    assert answer.issues == [] and "12,345" in answer.text


def test_unknown_company_gets_no_evidence_and_refusal():
    store, embedder = make_store()

    def respond(system, user):
        if system != ANSWER_SYSTEM:
            return {
                "ticker": "RACE",
                "fiscal_year": 2024,
                "metrics": ["net_income"],
                "search_query": "net income",
            }
        assert json.loads(user)["evidence"] == []
        return {"refused": True, "refusal_reason": "No filings for this company.", "claims": []}

    answer, evidence, plan = FilingsAgent(FakeLLM(respond), store, embedder).ask("Ferrari?")
    assert plan["ticker"] is None  # unknown tickers are dropped, not guessed
    assert answer.refused and evidence == {}
