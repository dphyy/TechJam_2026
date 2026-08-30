from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


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
    scope: str | None = None
    source_kind: str = "explicit"


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
    question_goal: str | None = None


@dataclass(frozen=True, slots=True)
class IntentDecision:
    """Explainable routing decision derived only from live conversational state."""

    mode: Literal["buying", "browsing", "mixed"]
    specificity: float
    confidence: float
    hard_constraint_count: int
    over_general: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalSufficiencyDecision:
    """Target-independent decision about how much retrieval work to perform."""

    action: Literal["retrieve", "minimal_probe", "clarify_first"]
    sufficient: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ComputeCascadeDecision:
    """Bounded pre-rerank decision; one turn runs either D30 or D60 once."""

    escalate: bool
    rerank_limit: int
    uncertainty: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalHypothesis:
    query: str
    object_types: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class FeedbackDecision:
    scope: Literal["none", "item", "product_type", "attribute_value", "attribute_unknown"]
    attribute: str | None = None
    reason: str = "none"


@dataclass(frozen=True, slots=True)
class StateDelta:
    """Observable semantic change produced by the latest user message."""

    kind: Literal["none", "additive", "refinement", "replacement", "polarity_change", "category_change"]
    added: tuple[tuple[str, str, int], ...] = ()
    removed: tuple[tuple[str, str, int], ...] = ()
    explicit_replacement: bool = False


@dataclass(frozen=True, slots=True)
class PlanSignal:
    attribute: str
    value: str
    polarity: int
    hard: bool
    confidence: float
    source_turn: int
    scope: str | None = None
    alternative_group: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    mode: Literal["buying", "browsing", "mixed"]
    object_types: tuple[str, ...]
    category_terms: tuple[str, ...]
    positive_terms: tuple[str, ...]
    negative_terms: tuple[str, ...]
    hard_constraints: tuple[PlanSignal, ...]
    soft_preferences: tuple[PlanSignal, ...]
    use_case: tuple[str, ...]
    scoped_features: tuple[PlanSignal, ...]
    semantic_queries: tuple[str, ...]
    lexical_query: str
    rerank_context: str
    hypotheses: tuple[RetrievalHypothesis, ...] = ()
