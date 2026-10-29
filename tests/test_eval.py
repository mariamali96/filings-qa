import pytest

from eval.build_numeric_gold import build_questions
from eval.metrics import wilson
from eval.report import review_sheet, summarize, to_markdown
from eval.scoring import score
from filings_qa.edgar import FilingRef
from filings_qa.models import Answer, Claim
from tests.test_xbrl import ACCN, COMPANYFACTS, REPORT_DATE


def test_wilson_interval_known_values():
    r = wilson(45, 49)
    assert r.value == pytest.approx(0.918, abs=1e-3)
    assert r.low == pytest.approx(0.809, abs=2e-3)
    assert r.high == pytest.approx(0.968, abs=2e-3)
    assert wilson(0, 0).fmt() == "n/a"


NUMERIC_Q = {
    "id": "num-1",
    "type": "numeric",
    "question": "Revenue?",
    "expected": {"value": 12345000000.0, "unit": "USD", "source_id": "EXMP-2024-xbrl-Revenues"},
}


def test_numeric_scoring_correct_and_grounded():
    a = Answer("Revenue?", [Claim("Revenue was $12.345 billion.", ["EXMP-2024-xbrl-Revenues"])])
    r = score(NUMERIC_Q, a)
    assert r["correct"] and r["grounded"] and not r["false_refusal"]


def test_numeric_scoring_right_number_wrong_source_is_not_grounded():
    a = Answer("Revenue?", [Claim("Revenue was $12,345 million.", ["EXMP-2024-item7-0"])])
    r = score(NUMERIC_Q, a)
    assert r["correct"] and not r["grounded"]


def test_refusal_scoring():
    refusal = Answer("q", [], refused=True, refusal_reason="no data")
    assert score({"id": "u", "type": "unanswerable", "question": "q"}, refusal)["correct"]
    r = score(NUMERIC_Q, refusal)
    assert r["false_refusal"] and not r["correct"]


def test_text_scoring_checks_item_and_points():
    q = {
        "id": "t",
        "type": "text",
        "question": "Supply risk?",
        "expected_items": ["1A"],
        "expected_points": ["single supplier", "taiwan"],
    }
    a = Answer("q", [Claim("Relies on a single supplier.", ["EXMP-2024-item1A-0"])])
    r = score(q, a)
    assert r["cited_expected_item"] is True
    assert r["key_points_found"] == "1/2"
    assert r["correct"] is None  # left for a human


def test_summary_uses_human_grades_and_renders():
    q = {"id": "t", "type": "text", "question": "Supply risk?", "expected_items": ["1A"]}
    a = Answer("q", [Claim("Relies on a single supplier.", ["EXMP-2024-item1A-0"])])
    results = [score(q, a), score(NUMERIC_Q, Answer("q", [], refused=True))]
    summary = summarize(results, {"t": "correct"})
    assert summary["text accuracy (human graded)"].successes == 1
    assert summary["false refusals on answerable questions"].successes == 1
    assert "| numeric accuracy |" in to_markdown(summary)
    assert "## t" in review_sheet(results)


def test_build_questions_from_xbrl():
    ref = FilingRef("AAPL", 320193, ACCN, 2024, REPORT_DATE, "2024-11-01", "aapl.htm")
    qs = build_questions(COMPANYFACTS, [ref], "Apple Inc.")
    by_id = {q["id"]: q for q in qs}
    assert by_id["num-AAPL-2024-revenue"]["expected"]["value"] == 391035000000
    assert "fiscal year 2024" in by_id["num-AAPL-2024-revenue"]["question"]
