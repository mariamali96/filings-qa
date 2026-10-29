"""Reported financial facts from the SEC XBRL "companyfacts" API.

These are the numbers the company itself tagged in its filing, so they work as
ground truth for numeric questions, and as a tool the agent can call instead of
reading numbers out of text.

A 10-K reports the current year and comparatives for earlier years, all inside the
same filing. To get the value *for* a fiscal year we keep only facts from that
year's 10-K whose period ends on the report date, and for flow concepts (revenue,
income) whose period is about one year long.
"""

from __future__ import annotations

from datetime import date

from filings_qa.models import Fact

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# Friendly metric name -> candidate us-gaap concepts, in order of preference.
# Companies tag the same idea with different concepts, so we try several.
METRICS: dict[str, list[str]] = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ],
    "net_income": ["NetIncomeLoss"],
    "operating_income": ["OperatingIncomeLoss"],
    "research_and_development": ["ResearchAndDevelopmentExpense"],
    "total_assets": ["Assets"],
    "cash_and_equivalents": ["CashAndCashEquivalentsAtCarryingValue"],
    "eps_diluted": ["EarningsPerShareDiluted"],
}

INSTANT_CONCEPTS = {"Assets", "CashAndCashEquivalentsAtCarryingValue"}


def _days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def select_annual_fact(
    companyfacts: dict,
    *,
    ticker: str,
    concept: str,
    fiscal_year: int,
    accession: str,
    report_date: str,
) -> Fact | None:
    """Pick the value of `concept` for the given fiscal year's 10-K, or None."""
    concept_data = companyfacts.get("facts", {}).get("us-gaap", {}).get(concept)
    if not concept_data:
        return None
    for unit, entries in concept_data.get("units", {}).items():
        for e in entries:
            if e.get("accn") != accession or e.get("form") != "10-K":
                continue
            if e.get("end") != report_date:
                continue
            if concept not in INSTANT_CONCEPTS:
                start = e.get("start")
                if not start or not 350 <= _days(start, e["end"]) <= 380:
                    continue  # skip quarterly values reported inside the 10-K
            return Fact(
                ticker=ticker,
                concept=concept,
                fiscal_year=fiscal_year,
                value=float(e["val"]),
                unit=unit,
                accession=accession,
                form="10-K",
                period_end=e["end"],
            )
    return None


def extract_metrics(
    companyfacts: dict, *, ticker: str, fiscal_year: int, accession: str, report_date: str
) -> dict[str, Fact]:
    """All known metrics for one filing, keyed by friendly metric name."""
    found: dict[str, Fact] = {}
    for metric, concepts in METRICS.items():
        for concept in concepts:
            fact = select_annual_fact(
                companyfacts,
                ticker=ticker,
                concept=concept,
                fiscal_year=fiscal_year,
                accession=accession,
                report_date=report_date,
            )
            if fact:
                found[metric] = fact
                break
    return found
