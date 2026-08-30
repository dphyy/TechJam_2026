from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class FacetEvidence:
    attribute: str
    value: str
    source: str
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class Product:
    parent_asin: str
    title: str
    fields: dict[str, str]
    facets: dict[str, tuple[str, ...]] = field(default_factory=dict)
    evidence: tuple[FacetEvidence, ...] = ()
    price: float | None = None
    price_lower_bound: bool = False

    @property
    def text(self) -> str:
        return " ".join(self.fields.values())


@dataclass(slots=True)
class Preference:
    attribute: str
    value: str
    source_turn: int
    source_text: str
    hard: bool = False
    polarity: int = 1
    active: bool = True
    confidence: float = 1.0
    depends_on: tuple[str, str] | None = None
    alternative_group: str | None = None


@dataclass(slots=True)
class Candidate:
    product: Product
    score: float
    route_scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    ask_attribute: str | None
    message: str
    slate_size: int
    diagnostics: dict = field(default_factory=dict)
