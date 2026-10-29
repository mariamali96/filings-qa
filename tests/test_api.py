from fastapi.testclient import TestClient

from filings_qa.api import answer_to_response, app
from filings_qa.models import Answer, Claim


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_response_only_includes_cited_sources():
    answer = Answer("q", [Claim("Revenue was $12,345 million.", ["a"])])
    resp = answer_to_response(answer, {"a": "source a", "b": "source b"})
    assert resp.sources == {"a": "source a"}
    assert resp.claims[0].citations == ["a"]
