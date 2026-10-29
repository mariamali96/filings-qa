"""Score one answer against one labelled question.

Question types (see eval/questions_*.yaml):

* numeric:      a reported number with ground truth from XBRL.
                Correct = some claim states the expected value (within 0.5%), and
                grounded = that claim cites the expected source.
* text:         a qualitative question. Automatic checks: did it answer, did it cite the
                expected 10-K item, how many of the hand-written key points appear.
                The final verdict comes from a human grade in the review sheet.
* unanswerable: the evidence cannot support an answer (future values, companies not
                ingested, facts not in a 10-K). Correct = the agent refuses.
"""

from __future__ import annotations

from filings_qa.models import Answer
from filings_qa.verify import extract_numbers, number_supported

HALLUCINATION_MARKER = "does not appear in its cited sources"


def score(question: dict, answer: Answer) -> dict:
    qtype = question["type"]
    result: dict = {
        "id": question["id"],
        "type": qtype,
        "question": question["question"],
        "answer": answer.text,
        "refused": answer.refused,
        "citations": sorted({c for claim in answer.claims for c in claim.citations}),
        "issues": answer.issues,
        "passed_verification": not answer.issues,
        "unsupported_number": any(HALLUCINATION_MARKER in i for i in answer.issues),
    }

    if qtype == "unanswerable":
        result["correct"] = answer.refused
        return result

    result["false_refusal"] = answer.refused

    if qtype == "numeric":
        expected = float(question["expected"]["value"])
        source = question["expected"].get("source_id")
        correct = grounded = False
        for claim in answer.claims:
            if number_supported(expected, extract_numbers(claim.text)):
                correct = True
                if source and source in claim.citations:
                    grounded = True
        result["correct"] = correct and not answer.refused
        result["grounded"] = grounded
        return result

    if qtype == "text":
        expected_items = {str(i).upper() for i in question.get("expected_items", [])}
        cited_items = {
            c.split("-item", 1)[1].rsplit("-", 1)[0].upper()
            for c in result["citations"]
            if "-item" in c
        }
        points = [p.lower() for p in question.get("expected_points", [])]
        text = answer.text.lower()
        hit = [p for p in points if p in text]
        result["cited_expected_item"] = (
            bool(expected_items & cited_items) if expected_items else None
        )
        result["key_points_found"] = f"{len(hit)}/{len(points)}" if points else None
        # Human grade: "correct" | "partial" | "wrong" | "invented" (filled in review).
        result["correct"] = None
        return result

    raise ValueError(f"Unknown question type: {qtype}")
