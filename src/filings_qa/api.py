"""HTTP API: POST /ask {"question": "..."}.

Run with:  uvicorn filings_qa.api:app --reload
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel

from filings_qa.agent import FilingsAgent

app = FastAPI(title="filings-qa", version="0.1.0")


class AskRequest(BaseModel):
    question: str


class ClaimOut(BaseModel):
    text: str
    citations: list[str]


class AskResponse(BaseModel):
    answer: str
    refused: bool
    claims: list[ClaimOut]
    sources: dict[str, str]
    issues: list[str]


@lru_cache(maxsize=1)
def get_agent() -> FilingsAgent:  # pragma: no cover - needs DB and API key
    from filings_qa.runtime import build_agent

    return build_agent()


def answer_to_response(answer, evidence: dict[str, str]) -> AskResponse:
    cited = {c for claim in answer.claims for c in claim.citations}
    return AskResponse(
        answer=answer.text,
        refused=answer.refused,
        claims=[ClaimOut(text=c.text, citations=c.citations) for c in answer.claims],
        sources={sid: text for sid, text in evidence.items() if sid in cited},
        issues=answer.issues,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:  # pragma: no cover - thin wrapper
    answer, evidence, _plan = get_agent().ask(request.question)
    return answer_to_response(answer, evidence)
