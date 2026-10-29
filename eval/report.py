"""Aggregate scored results into the metrics shown in the README."""

from __future__ import annotations

from eval.metrics import Rate, wilson


def _rate(rows: list[dict], key: str) -> Rate:
    usable = [r for r in rows if r.get(key) is not None]
    return wilson(sum(bool(r[key]) for r in usable), len(usable))


def summarize(results: list[dict], human_grades: dict[str, str] | None = None) -> dict[str, Rate]:
    human_grades = human_grades or {}
    for r in results:
        if r["type"] == "text" and r["id"] in human_grades:
            r["human_grade"] = human_grades[r["id"]]
            r["correct"] = human_grades[r["id"]] == "correct"

    numeric = [r for r in results if r["type"] == "numeric"]
    text = [r for r in results if r["type"] == "text"]
    answerable = numeric + text
    unanswerable = [r for r in results if r["type"] == "unanswerable"]
    answered = [r for r in results if not r["refused"]]

    return {
        "numeric accuracy": _rate(numeric, "correct"),
        "numeric answers citing the right source": _rate(numeric, "grounded"),
        "text accuracy (human graded)": _rate(text, "correct"),
        "text answers citing the expected 10-K item": _rate(text, "cited_expected_item"),
        "correct refusals on unanswerable questions": _rate(unanswerable, "correct"),
        "false refusals on answerable questions": _rate(answerable, "false_refusal"),
        "answers passing all verification checks": _rate(answered, "passed_verification"),
        "answers with an unsupported number": _rate(answered, "unsupported_number"),
    }


def to_markdown(summary: dict[str, Rate], title: str = "Evaluation results") -> str:
    lines = [f"## {title}", "", "| Metric | Result |", "|---|---|"]
    lines += [f"| {name} | {rate.fmt()} |" for name, rate in summary.items()]
    return "\n".join(lines) + "\n"


def review_sheet(results: list[dict]) -> str:
    """A markdown sheet for grading text answers by hand."""
    lines = [
        "# Review sheet",
        "",
        "Grade each text answer by reading its cited passages: "
        "`correct`, `partial`, `wrong` or `invented` (states something no source says).",
        "Put grades in eval/human_grades.yaml as `<id>: <grade>`.",
        "",
    ]
    for r in results:
        if r["type"] != "text":
            continue
        lines += [
            f"## {r['id']}",
            f"**Q:** {r['question']}",
            "",
            f"**A:** {r['answer']}",
            "",
            f"Citations: {', '.join(r['citations']) or 'none'}",
            f"Cited expected item: {r['cited_expected_item']} | "
            f"key points found: {r['key_points_found']} | issues: {len(r['issues'])}",
            "",
        ]
    return "\n".join(lines)
