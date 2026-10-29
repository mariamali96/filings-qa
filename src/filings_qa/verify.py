"""Deterministic checks on an answer, before anyone trusts it.

These do not use an LLM, so they cannot be talked out of a problem:

1. every claim cites at least one source;
2. every cited source was actually given to the model;
3. every number in a claim appears in one of the sources that claim cites
   (allowing for "391.0 billion" vs "391,035" in a table reported in millions).

Check 3 is the one that catches the most dangerous failure in finance: a fluent,
confident, wrong number.
"""

from __future__ import annotations

import re

from filings_qa.models import Answer

_NUMBER = re.compile(
    r"(?<![\w.])\$?\(?(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)\)?"
    r"\s*(?:(billion|million|thousand|bn|mn|m|b)\b|(%))?",
    re.IGNORECASE,
)
_SCALE = {
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "million": 1e6,
    "mn": 1e6,
    "m": 1e6,
    "thousand": 1e3,
}
# Tables in 10-Ks are often "in millions" or "in thousands"; a source number may
# need scaling before it can be compared with a claim.
_SOURCE_SCALES = (1.0, 1e3, 1e6, 1e9)


def extract_numbers(text: str) -> list[float]:
    values: list[float] = []
    for m in _NUMBER.finditer(text):
        raw = m.group(1)
        suffix = (m.group(2) or m.group(3) or "").lower()
        value = float(raw.replace(",", ""))
        if suffix == "%":
            values.append(value)
            continue
        if not suffix and "." not in raw and "," not in raw and 1990 <= value <= 2100:
            continue  # a year, not a quantity
        values.append(value * _SCALE.get(suffix, 1.0))
    return values


def _close(a: float, b: float, rel_tol: float) -> bool:
    if a == b:
        return True
    return abs(a - b) <= rel_tol * max(abs(a), abs(b))


def number_supported(value: float, source_numbers: list[float], rel_tol: float = 0.005) -> bool:
    return any(
        _close(value, s * scale, rel_tol) or _close(abs(value), abs(s * scale), rel_tol)
        for s in source_numbers
        for scale in _SOURCE_SCALES
    )


def verify_answer(answer: Answer, evidence: dict[str, str], rel_tol: float = 0.005) -> list[str]:
    """Return a list of human-readable problems (empty list means the checks passed)."""
    if answer.refused:
        return []
    issues: list[str] = []
    if not answer.claims:
        issues.append("Answer has no claims and is not a refusal.")
    for i, claim in enumerate(answer.claims, start=1):
        if not claim.citations:
            issues.append(f"Claim {i} has no citation: {claim.text!r}")
            continue
        unknown = [c for c in claim.citations if c not in evidence]
        if unknown:
            issues.append(f"Claim {i} cites sources that were not provided: {unknown}")
        cited_text = " ".join(evidence[c] for c in claim.citations if c in evidence)
        source_numbers = extract_numbers(cited_text)
        for value in extract_numbers(claim.text):
            if not number_supported(value, source_numbers, rel_tol):
                issues.append(
                    f"Claim {i} states {value:g}, which does not appear in its cited sources."
                )
    return issues
