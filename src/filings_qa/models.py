"""Shared data types.

Every piece of evidence the agent can cite has a stable `source_id`, so a citation
can always be traced back to an exact passage of a filing or an exact reported fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    """A passage of a 10-K filing."""

    ticker: str
    fiscal_year: int
    accession: str  # SEC accession number of the filing
    item: str  # e.g. "1A" (Risk Factors), "7" (MD&A)
    index: int  # position of the chunk within its item
    text: str

    @property
    def source_id(self) -> str:
        return f"{self.ticker}-{self.fiscal_year}-item{self.item}-{self.index}"


@dataclass(frozen=True)
class Fact:
    """A value reported in a filing's XBRL data (e.g. revenue for fiscal year 2024)."""

    ticker: str
    concept: str  # us-gaap concept name, e.g. "Revenues"
    fiscal_year: int
    value: float
    unit: str  # e.g. "USD"
    accession: str
    form: str  # e.g. "10-K"
    period_end: str  # ISO date

    @property
    def source_id(self) -> str:
        return f"{self.ticker}-{self.fiscal_year}-xbrl-{self.concept}"

    def as_text(self) -> str:
        return (
            f"{self.ticker} {self.concept} for fiscal year {self.fiscal_year} "
            f"(period ending {self.period_end}): {self.value:,.0f} {self.unit} "
            f"[{self.form}, accession {self.accession}]"
        )


@dataclass
class Claim:
    text: str
    citations: list[str]


@dataclass
class Answer:
    question: str
    claims: list[Claim]
    refused: bool = False  # the agent said it cannot answer from the evidence
    refusal_reason: str = ""
    issues: list[str] = field(default_factory=list)  # problems found by verification

    @property
    def text(self) -> str:
        if self.refused:
            return f"I can't answer this from the filings: {self.refusal_reason}"
        return " ".join(c.text for c in self.claims)
