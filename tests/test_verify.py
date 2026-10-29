import pytest

from filings_qa.models import Answer, Claim
from filings_qa.verify import extract_numbers, number_supported, verify_answer


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Net sales were $391.0 billion.", [391.0e9]),
        ("Net sales 391,035 (in millions)", [391035.0]),
        ("Revenue grew 8% in fiscal 2024", [8.0]),  # 2024 is a year, not a quantity
        ("EPS of $6.08", [6.08]),
        ("a loss of $(1,234) million", [1234.0e6]),
        ("The board has 10 members", [10.0]),  # "m" in "members" is not "million"
    ],
)
def test_extract_numbers(text, expected):
    assert extract_numbers(text) == pytest.approx(expected)


def test_number_supported_handles_table_units():
    # Claim says 12.3 billion, the table says 12,345 (in millions).
    assert number_supported(12.345e9, [12345.0])
    assert not number_supported(13.0e9, [12345.0])


EVIDENCE = {
    "EXMP-2024-xbrl-Revenues": "EXMP Revenues for fiscal year 2024: 12,345,000,000 USD",
    "EXMP-2024-item7-0": "Total net sales increased 8% to $12,345 million in fiscal 2024.",
}


def test_grounded_answer_passes():
    answer = Answer(
        question="q",
        claims=[
            Claim("Revenue was $12,345 million in fiscal 2024.", ["EXMP-2024-xbrl-Revenues"]),
            Claim("Sales grew 8%.", ["EXMP-2024-item7-0"]),
        ],
    )
    assert verify_answer(answer, EVIDENCE) == []


def test_invented_number_is_flagged():
    answer = Answer(
        question="q",
        claims=[Claim("Revenue was $14.2 billion.", ["EXMP-2024-xbrl-Revenues"])],
    )
    issues = verify_answer(answer, EVIDENCE)
    assert len(issues) == 1 and "does not appear" in issues[0]


def test_number_must_be_in_the_cited_source_not_just_anywhere():
    # 8% is in the MD&A passage, but this claim only cites the XBRL fact.
    answer = Answer(question="q", claims=[Claim("Sales grew 8%.", ["EXMP-2024-xbrl-Revenues"])])
    assert any("does not appear" in i for i in verify_answer(answer, EVIDENCE))


def test_uncited_and_unknown_citations_are_flagged():
    answer = Answer(
        question="q",
        claims=[Claim("Revenue grew.", []), Claim("Margins rose.", ["EXMP-2023-item7-9"])],
    )
    issues = verify_answer(answer, EVIDENCE)
    assert any("no citation" in i for i in issues)
    assert any("not provided" in i for i in issues)


def test_refusal_passes_and_empty_answer_fails():
    assert verify_answer(Answer("q", [], refused=True, refusal_reason="no data"), {}) == []
    assert verify_answer(Answer("q", []), {})
