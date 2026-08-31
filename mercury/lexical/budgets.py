"""Shared, bounded price parsing and constraint semantics for catalog currency."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import re


# Never accept the numeric prefix of a malformed amount, exponent or measurement.
AMOUNT = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w|,\d|\.\d)"
MONEY = rf"(?P<currency>\$|USD\s+)?(?P<amount>{AMOUNT})(?:\s+(?P<unit>dollars?|USD)\b)?"
NEGATIVE_SPEND = (r"(?:i\s+)?(?:can't|cannot|don't\s+want\s+to|do\s+not\s+want\s+to|won't|will\s+not)"
                  r"\s+(?:spend|pay)\s+(?:more\s+than|less\s+than|over|above|under|below)")
MODE = (NEGATIVE_SPEND + r"|not\s+under|not\s+below|not\s+over|not\s+above|no\s+more\s+than|not\s+more\s+than|"
        r"no\s+less\s+than|not\s+less\s+than|less\s+than|more\s+than|at\s+most|at\s+least|"
        r"up\s+to|under|below|over|above|maximum|max|minimum|min|<=|>=|"
        r"budget\s+around|around|about|approximately|roughly|budget(?:\s+(?:of|is))?")
PRICE_RE = re.compile(rf"(?<![\w.,$-])(?:(?P<mode>{MODE})\s*)?{MONEY}", re.I)
RANGE_RE = re.compile(
    rf"(?<![\w.,$-])(?:(?P<intro>between|from)\s+)?(?P<c1>\$|USD\s+)?(?P<lo>{AMOUNT})"
    rf"\s*(?P<link>\band\b|\bto\b|[-–])\s*(?P<c2>\$|USD\s+)?(?P<hi>{AMOUNT})(?:\s+(?P<unit>dollars?|USD)\b)?", re.I,
)


@dataclass(frozen=True, slots=True)
class BudgetPreference:
    mode: str
    amount: float
    weight: float
    group: str = ""
    alternative: int = 0

    @property
    def hard(self) -> bool:
        return self.mode != "around"

    def allows(self, price: float) -> bool:
        if self.mode in {"under", "below"}:
            return price < self.amount
        if self.mode in {"maximum", "max"}:
            return price <= self.amount
        if self.mode == "over":
            return price > self.amount
        if self.mode == "minimum":
            return price >= self.amount
        return abs(price - self.amount) <= max(self.amount, 10.0) * .35

    def relative_violation(self, price: float) -> float:
        if self.allows(price):
            return 0.0
        distance = abs(price - self.amount) / max(self.amount, 10.0)
        return max(distance, .001) if self.hard else max(0.0, distance - .35)


def _mode(value: str | None) -> str:
    value = " ".join((value or "around").lower().split())
    if re.fullmatch(NEGATIVE_SPEND, value, re.I):
        return "minimum" if re.search(r"(?:less than|under|below)$", value) else "maximum"
    if value in {"under", "below", "less than"}:
        return "under"
    if value in {"over", "above", "more than"}:
        return "over"
    if value in {"minimum", "min", ">=", "at least", "no less than", "not less than", "not under", "not below"}:
        return "minimum"
    if value in {"maximum", "max", "<=", "at most", "up to", "no more than", "not more than",
                 "not over", "not above", "budget", "budget of", "budget is"}:
        return "maximum"
    return "around"


@lru_cache(maxsize=4096)
def _limits(text: str) -> tuple[tuple[str, float, int, int, int], ...]:
    text = text[:8000]
    values, ranges = [], []
    for match in RANGE_RE.finditer(text):
        if match.group("link").lower() == "and" and not match.group("intro"):
            continue
        if not (match.group("c1") or match.group("c2") or match.group("unit")):
            continue
        lo, hi = (float(match.group(name).replace(",", "")) for name in ("lo", "hi"))
        ranges.append(match.span())
        if math.isfinite(lo) and math.isfinite(hi) and lo <= hi:
            values.extend((("minimum", lo, *match.span()), ("maximum", hi, *match.span())))
    for match in PRICE_RE.finditer(text):
        if any(start <= match.start() < end for start, end in ranges):
            continue
        prefix = text[max(0, match.start() - 40):match.start()]
        if not (match.group("currency") or match.group("unit") or
                match.group("mode") and re.search(r"\b(?:budget|price|cost|spend|pay)\b", prefix, re.I) or
                (match.group("mode") or "").lower().startswith("budget")):
            continue
        # A bare negated amount is not an asserted budget.
        if re.search(r"\b(?:not|avoid)\s*$", prefix, re.I):
            continue
        amount = float(match.group("amount").replace(",", ""))
        if math.isfinite(amount):
            values.append((_mode(match.group("mode")), amount, *match.span()))
    values.sort(key=lambda row: row[2])
    result, branch, last_end = [], 0, 0
    for mode, amount, start, end in values:
        if result and re.fullmatch(r"\s+or\s+", text[last_end:start], re.I):
            branch += 1
        result.append((mode, amount, start, end, branch))
        last_end = end
    return tuple(dict.fromkeys(result))


def parse_budgets(text: str, weight: float = 1.0) -> tuple[BudgetPreference, ...]:
    limits = _limits(text)
    group = text if any(branch for _, _, _, _, branch in limits) else ""
    return tuple(BudgetPreference(mode, amount, weight, group, branch) for mode, amount, _, _, branch in limits)


def budget_groups(budgets):
    """Conjunction of requirements, each with one or more alternative intervals."""
    grouped = {}
    for index, budget in enumerate(budgets):
        grouped.setdefault(budget.group or index, {}).setdefault(budget.alternative, []).append(budget)
    return tuple(tuple(tuple(branch) for branch in alternatives.values()) for alternatives in grouped.values())


def budgets_allow(price: float, budgets, *, hard_only: bool = False) -> bool:
    return all(any(all((hard_only and not budget.hard) or budget.allows(price) for budget in branch)
                   for branch in group) for group in budget_groups(budgets))


def separate_budget(text: str) -> tuple[str, ...]:
    """Keep price and non-price intent independently retractable in dialogue."""
    limits = _limits(text)
    spans = sorted({(start, end) for _, _, start, end, _ in limits})
    if not spans:
        return (text,)
    # Keep an OR expression together, including its connective.
    if any(branch for _, _, _, _, branch in limits):
        spans = [(spans[0][0], spans[-1][1])]
    remainder = text
    for start, end in reversed(spans):
        remainder = remainder[:start] + " " + remainder[end:]
    remainder = re.sub(r"\s+", " ", remainder).strip(" ,;.")
    remainder = re.sub(r"^(?:and|with)\s+|\s+(?:and|with)$", "", remainder, flags=re.I)
    substantive = re.sub(r"\b(?:i|a|an|my|the|need|want|prefer|require|have|budget|price|cost|of|is|to|spend|pay|and|with)\b",
                         "", remainder, flags=re.I).strip(" ,;.")
    if not substantive:
        return (text,)
    return (remainder, *(text[start:end] for start, end in spans))
