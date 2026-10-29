"""Download 10-K filings from SEC EDGAR.

SEC's fair-access rules: identify yourself with a User-Agent containing a contact email,
and stay under 10 requests per second. https://www.sec.gov/os/accessing-edgar-data
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}"


@dataclass(frozen=True)
class FilingRef:
    ticker: str
    cik: int
    accession: str
    fiscal_year: int  # simplification: the calendar year of the report period end
    report_date: str  # period of report, ISO date
    filing_date: str
    primary_document: str

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(
            cik=self.cik,
            accession_nodash=self.accession.replace("-", ""),
            document=self.primary_document,
        )


class EdgarClient:
    def __init__(self, user_agent: str, cache_dir: Path, min_interval_s: float = 0.15):
        if "@" not in user_agent:
            raise ValueError(
                "SEC requires a User-Agent with a contact email, e.g. 'Jane Doe jane@example.com'. "
                "Set SEC_USER_AGENT in your .env."
            )
        self._http = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=60,
            follow_redirects=True,
        )
        self._cache = cache_dir
        self._cache.mkdir(parents=True, exist_ok=True)
        self._min_interval = min_interval_s
        self._last_request = 0.0

    def _get(self, url: str) -> bytes:
        wait = self._min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        response = self._http.get(url)
        self._last_request = time.monotonic()
        response.raise_for_status()
        return response.content

    def _get_cached(self, url: str, name: str) -> bytes:
        path = self._cache / name
        if path.exists():
            return path.read_bytes()
        content = self._get(url)
        path.write_bytes(content)
        return content

    def cik_for(self, ticker: str) -> int:
        table = json.loads(self._get_cached(TICKERS_URL, "company_tickers.json"))
        for row in table.values():
            if row["ticker"].upper() == ticker.upper():
                return int(row["cik_str"])
        raise KeyError(f"Unknown ticker: {ticker}")

    def annual_filings(self, ticker: str, limit: int = 3) -> list[FilingRef]:
        """Most recent original 10-K filings (amendments excluded), newest first."""
        cik = self.cik_for(ticker)
        data = json.loads(
            self._get_cached(SUBMISSIONS_URL.format(cik=cik), f"submissions_{cik}.json")
        )
        recent = data["filings"]["recent"]
        refs: list[FilingRef] = []
        for i, form in enumerate(recent["form"]):
            if form != "10-K":
                continue
            report_date = recent["reportDate"][i]
            refs.append(
                FilingRef(
                    ticker=ticker.upper(),
                    cik=cik,
                    accession=recent["accessionNumber"][i],
                    fiscal_year=int(report_date[:4]),
                    report_date=report_date,
                    filing_date=recent["filingDate"][i],
                    primary_document=recent["primaryDocument"][i],
                )
            )
            if len(refs) >= limit:
                break
        return refs

    def download(self, ref: FilingRef) -> str:
        name = f"{ref.ticker}_{ref.fiscal_year}_{ref.accession}.htm"
        return self._get_cached(ref.url, name).decode("utf-8", errors="replace")

    def get_json(self, url: str, cache_name: str) -> dict:
        return json.loads(self._get_cached(url, cache_name))
