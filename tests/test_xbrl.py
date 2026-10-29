from filings_qa.xbrl import extract_metrics, select_annual_fact

ACCN = "0000320193-24-000123"
REPORT_DATE = "2024-09-28"

COMPANYFACTS = {
    "facts": {
        "us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "units": {
                    "USD": [
                        # prior-year comparative inside this year's 10-K: must be ignored
                        {
                            "start": "2022-09-25",
                            "end": "2023-09-30",
                            "val": 383285000000,
                            "accn": ACCN,
                            "form": "10-K",
                            "fy": 2024,
                            "fp": "FY",
                        },
                        # Q4 value inside the 10-K: ignored (not a full year)
                        {
                            "start": "2024-06-30",
                            "end": "2024-09-28",
                            "val": 94930000000,
                            "accn": ACCN,
                            "form": "10-K",
                            "fy": 2024,
                            "fp": "FY",
                        },
                        # the one we want
                        {
                            "start": "2023-10-01",
                            "end": "2024-09-28",
                            "val": 391035000000,
                            "accn": ACCN,
                            "form": "10-K",
                            "fy": 2024,
                            "fp": "FY",
                        },
                    ]
                }
            },
            "Assets": {
                "units": {
                    "USD": [
                        {"end": "2024-09-28", "val": 364980000000, "accn": ACCN, "form": "10-K"},
                        {"end": "2023-09-30", "val": 352583000000, "accn": ACCN, "form": "10-K"},
                    ]
                }
            },
        }
    }
}


def test_select_annual_fact_ignores_comparatives_and_quarters():
    fact = select_annual_fact(
        COMPANYFACTS,
        ticker="AAPL",
        concept="RevenueFromContractWithCustomerExcludingAssessedTax",
        fiscal_year=2024,
        accession=ACCN,
        report_date=REPORT_DATE,
    )
    assert fact is not None and fact.value == 391035000000


def test_extract_metrics_falls_back_across_concepts_and_handles_instants():
    metrics = extract_metrics(
        COMPANYFACTS, ticker="AAPL", fiscal_year=2024, accession=ACCN, report_date=REPORT_DATE
    )
    assert metrics["revenue"].value == 391035000000
    assert metrics["total_assets"].value == 364980000000
    assert "net_income" not in metrics
    assert metrics["revenue"].source_id == (
        "AAPL-2024-xbrl-RevenueFromContractWithCustomerExcludingAssessedTax"
    )
