"""Generate numeric evaluation questions with ground truth taken from XBRL.

The expected answer is the value the company itself reported, so no hand labelling is
needed for these. (Text questions are labelled by hand: see eval/questions_text.yaml.)
"""

from __future__ import annotations

from pathlib import Path

import yaml

from filings_qa.xbrl import COMPANYFACTS_URL, extract_metrics

QUESTION_TEMPLATES = {
    "revenue": "What was {name}'s total revenue for fiscal year {year}?",
    "net_income": "What net income did {name} report for fiscal year {year}?",
    "operating_income": "What was {name}'s operating income in fiscal year {year}?",
    "research_and_development": "How much did {name} spend on research and development in fiscal year {year}?",
    "total_assets": "What were {name}'s total assets at the end of fiscal year {year}?",
    "cash_and_equivalents": "How much cash and cash equivalents did {name} hold at the end of fiscal year {year}?",
    "eps_diluted": "What was {name}'s diluted earnings per share for fiscal year {year}?",
}


def build_questions(companyfacts: dict, refs: list, name: str) -> list[dict]:
    questions = []
    for ref in refs:
        facts = extract_metrics(
            companyfacts,
            ticker=ref.ticker,
            fiscal_year=ref.fiscal_year,
            accession=ref.accession,
            report_date=ref.report_date,
        )
        for metric, fact in facts.items():
            questions.append(
                {
                    "id": f"num-{ref.ticker}-{ref.fiscal_year}-{metric}",
                    "type": "numeric",
                    "question": QUESTION_TEMPLATES[metric].format(name=name, year=ref.fiscal_year),
                    "expected": {
                        "value": fact.value,
                        "unit": fact.unit,
                        "source_id": fact.source_id,
                    },
                }
            )
    return questions


def main(tickers: list[str], years: int, out: str) -> None:  # pragma: no cover - network
    from filings_qa.config import get_settings
    from filings_qa.edgar import SUBMISSIONS_URL, EdgarClient

    s = get_settings()
    client = EdgarClient(s.sec_user_agent, s.data_dir / "edgar")
    all_questions: list[dict] = []
    for ticker in tickers:
        refs = client.annual_filings(ticker.upper(), limit=years)
        if not refs:
            continue
        cik = refs[0].cik
        name = client.get_json(SUBMISSIONS_URL.format(cik=cik), f"submissions_{cik}.json")["name"]
        facts = client.get_json(COMPANYFACTS_URL.format(cik=cik), f"companyfacts_{cik}.json")
        all_questions += build_questions(facts, refs, name.title())
    Path(out).write_text(yaml.safe_dump({"questions": all_questions}, sort_keys=False))
    print(f"Wrote {len(all_questions)} numeric questions to {out}")
