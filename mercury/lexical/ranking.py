from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .dialogue import EXPLORING_RE, SessionState


class RankingMode(str, Enum):
    """Runtime retrieval strategy inferred only from observable customer state."""

    BUYING = "buying"
    BROWSING = "browsing"


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    mode: RankingMode
    reasons: tuple[str, ...]


class IntentRouter:
    """Route explicit requirements to precision and open intent to discovery."""

    _HARD_SOURCES = frozenset({"hard_constraint", "override"})

    def route(self, state: SessionState) -> RoutingDecision:
        sources = {item.source for item in state.evidence}
        if sources & self._HARD_SOURCES:
            return RoutingDecision(RankingMode.BUYING, ("explicit_requirement",))

        started_open_ended = bool(
            state.messages and EXPLORING_RE.search(state.messages[0])
        )
        if started_open_ended:
            return RoutingDecision(RankingMode.BROWSING, ("open_ended_start",))

        if "initial_preference" in sources:
            return RoutingDecision(RankingMode.BUYING, ("stated_preference",))

        has_specific_evidence = any(
            item.source != "category" for item in state.evidence
        )
        if has_specific_evidence:
            return RoutingDecision(RankingMode.BUYING, ("specific_evidence",))
        return RoutingDecision(RankingMode.BROWSING, ("category_only",))


@dataclass(frozen=True, slots=True)
class RankingPolicy:
    """Mode-specific weights applied to general, data-derived score features."""

    rrf_scale: float = 1.0
    constraint_scale: float = 1.0
    price_scale: float = 1.0
    quality_scale: float = 1.0
    vector_scale: float = 1.0
    hard_coverage_bonus: float = 0.0
    hard_exact_bonus: float = 0.0
    hard_missing_penalty: float = 0.0
    contradiction_penalty: float = 0.0
    soft_coverage_bonus: float = 0.0
    soft_exact_bonus: float = 0.0
    budget_violation_penalty: float = 0.0
    profile_rating_scale: float = 0.0


@dataclass(frozen=True, slots=True)
class RankingPolicies:
    buying: RankingPolicy
    browsing: RankingPolicy

    def for_mode(self, mode: RankingMode) -> RankingPolicy:
        return self.buying if mode is RankingMode.BUYING else self.browsing


LEGACY_POLICY = RankingPolicy()
LEGACY_RANKING_POLICIES = RankingPolicies(
    buying=LEGACY_POLICY,
    browsing=LEGACY_POLICY,
)


DEFAULT_RANKING_POLICIES = RankingPolicies(
    buying=RankingPolicy(
        vector_scale=0.0,
        hard_coverage_bonus=0.30,
        hard_exact_bonus=0.30,
        hard_missing_penalty=0.25,
        contradiction_penalty=0.50,
        budget_violation_penalty=0.50,
        profile_rating_scale=0.75,
    ),
    browsing=RankingPolicy(),
)
