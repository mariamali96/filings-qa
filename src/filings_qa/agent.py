"""The question-answering agent, as a LangGraph state machine.

    plan ──► gather ──► answer ──► verify ──► done
                          ▲          │
                          └─ retry ──┘   (once, with the problems found)

* plan:   the LLM reads the question and decides which company and year it is about,
          which reported metrics it needs, and what to search for in the filing text.
* gather: deterministic tools fetch evidence: XBRL facts and filing passages from pgvector.
* answer: the LLM answers using only that evidence, as claims that each cite source ids,
          or refuses if the evidence does not support an answer.
* verify: deterministic checks (see verify.py). If they fail, the LLM gets one retry with
          the list of problems; if it still fails, the answer is returned with its issues
          attached instead of being silently "fixed".
"""

from __future__ import annotations

import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from filings_qa.embed import Embedder
from filings_qa.llm import LLM
from filings_qa.models import Answer, Claim
from filings_qa.store import Store
from filings_qa.verify import verify_answer
from filings_qa.xbrl import METRICS

PLAN_SYSTEM = """You plan how to answer questions about US public companies' 10-K filings.
Return ONLY a JSON object with these keys:
  "ticker": one of the known tickers, or null if the question names no known company
  "fiscal_year": integer fiscal year, or null if not stated
  "metrics": list of metric names needed, chosen from the known metrics (may be empty)
  "search_query": a short query for searching the filing text, or null if numbers suffice
Do not answer the question."""

ANSWER_SYSTEM = """You answer questions about 10-K filings using ONLY the evidence provided.
Rules:
- Split the answer into short factual claims. Every claim must cite one or more evidence ids
  exactly as given, e.g. "AAPL-2024-xbrl-Revenues" or "AAPL-2024-item1A-3".
- Copy numbers exactly from the evidence. Do not compute new numbers unless the claim shows
  the inputs, and never round a number in a way that changes its meaning.
- If the evidence does not contain what is needed, refuse instead of guessing.
Return ONLY a JSON object:
  {"refused": false, "refusal_reason": "", "claims": [{"text": "...", "citations": ["id"]}]}
or {"refused": true, "refusal_reason": "why the evidence is insufficient", "claims": []}"""


class AgentState(TypedDict, total=False):
    question: str
    plan: dict
    evidence: dict[str, str]
    answer: Answer
    issues: list[str]
    attempts: int


class FilingsAgent:
    def __init__(
        self, llm: LLM, store: Store, embedder: Embedder, k: int = 6, max_retries: int = 1
    ):
        self.llm = llm
        self.store = store
        self.embedder = embedder
        self.k = k
        self.max_retries = max_retries
        self.graph = self._build()

    # ---- nodes -------------------------------------------------------------

    def _plan(self, state: AgentState) -> AgentState:
        user = json.dumps(
            {
                "question": state["question"],
                "known_tickers": self.store.known_tickers(),
                "known_metrics": sorted(METRICS),
            }
        )
        plan = self.llm.complete_json(PLAN_SYSTEM, user)
        plan["metrics"] = [m for m in plan.get("metrics") or [] if m in METRICS]
        if plan.get("ticker") not in self.store.known_tickers():
            plan["ticker"] = None
        return {"plan": plan, "attempts": 0}

    def _gather(self, state: AgentState) -> AgentState:
        plan = state["plan"]
        ticker, year = plan.get("ticker"), plan.get("fiscal_year")
        evidence: dict[str, str] = {}
        if ticker and year:
            for metric in plan["metrics"]:
                fact = self.store.get_fact(ticker, metric, int(year))
                if fact:
                    evidence[fact.source_id] = fact.as_text()
        query = plan.get("search_query")
        if query and ticker:  # unknown company: no search, so the answer step will refuse
            hits = self.store.search(
                self.embedder.embed_query(query),
                ticker=ticker,
                fiscal_year=int(year) if year else None,
                k=self.k,
            )
            for chunk, _score in hits:
                evidence[chunk.source_id] = chunk.text
        return {"evidence": evidence}

    def _answer(self, state: AgentState) -> AgentState:
        payload = {
            "question": state["question"],
            "evidence": [{"id": sid, "text": text} for sid, text in state["evidence"].items()],
        }
        if state.get("issues"):
            payload["problems_with_previous_answer"] = state["issues"]
        raw = self.llm.complete_json(ANSWER_SYSTEM, json.dumps(payload))
        answer = Answer(
            question=state["question"],
            refused=bool(raw.get("refused")),
            refusal_reason=str(raw.get("refusal_reason") or ""),
            claims=[
                Claim(text=str(c.get("text", "")), citations=list(c.get("citations") or []))
                for c in raw.get("claims") or []
            ],
        )
        return {"answer": answer, "attempts": state.get("attempts", 0) + 1}

    def _verify(self, state: AgentState) -> AgentState:
        issues = verify_answer(state["answer"], state["evidence"])
        state["answer"].issues = issues
        return {"issues": issues}

    def _after_verify(self, state: AgentState) -> str:
        if state["issues"] and state["attempts"] <= self.max_retries:
            return "answer"
        return END

    # ---- graph -------------------------------------------------------------

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("plan", self._plan)
        g.add_node("gather", self._gather)
        g.add_node("answer", self._answer)
        g.add_node("verify", self._verify)
        g.add_edge(START, "plan")
        g.add_edge("plan", "gather")
        g.add_edge("gather", "answer")
        g.add_edge("answer", "verify")
        g.add_conditional_edges("verify", self._after_verify, ["answer", END])
        return g.compile()

    def ask(self, question: str) -> tuple[Answer, dict[str, str], dict]:
        """Answer a question. Returns (answer, evidence shown to the model, plan)."""
        final = self.graph.invoke({"question": question})
        return final["answer"], final["evidence"], final["plan"]
