"""Proportions with honest uncertainty.

With 50 questions, "92% accurate" could easily be 81-97%. Every rate in the report is
shown with a Wilson 95% interval so small samples are not over-read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Rate:
    successes: int
    total: int
    low: float
    high: float

    @property
    def value(self) -> float:
        return self.successes / self.total if self.total else float("nan")

    def fmt(self) -> str:
        if not self.total:
            return "n/a"
        return f"{self.value:.0%} ({self.successes}/{self.total}, 95% CI {self.low:.0%}-{self.high:.0%})"


def wilson(successes: int, total: int, z: float = 1.96) -> Rate:
    if total == 0:
        return Rate(0, 0, float("nan"), float("nan"))
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return Rate(successes, total, max(0.0, centre - half), min(1.0, centre + half))
