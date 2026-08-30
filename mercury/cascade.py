"""Target-independent compute escalation under a hard neural-pair ceiling."""

from __future__ import annotations

from mercury.config import Config
from mercury.types import ComputeCascadeDecision, IntentDecision, RetrievalPlan


def decide_compute_cascade(intent: IntentDecision, plan: RetrievalPlan,
                           route_overlap: dict[str, float], candidate_count: int,
                           config: Config, escalations_used: int,
                           reranker_available: bool,
                           previous_neural_margin: float | None = None) -> ComputeCascadeDecision:
    base = config.rerank_limit
    if not config.compute_cascade:
        return ComputeCascadeDecision(False, base, 0.0, ("cascade_disabled",))
    if not reranker_available:
        return ComputeCascadeDecision(False, base, 0.0, ("reranker_unavailable",))
    if escalations_used >= config.cascade_max_turns:
        return ComputeCascadeDecision(False, base, 0.0, ("session_budget_exhausted",))

    uncertainty = 0.0
    reasons = []
    if intent.confidence < config.cascade_low_confidence:
        uncertainty += 0.30
        reasons.append("low_intent_confidence")
    if route_overlap and min(route_overlap.values()) < config.cascade_low_overlap:
        uncertainty += 0.30
        reasons.append("route_disagreement")
    if candidate_count >= config.cascade_candidate_threshold:
        uncertainty += 0.20
        reasons.append("candidate_pressure")
    if not plan.object_types:
        uncertainty += 0.20
        reasons.append("unresolved_object")
    if (config.cascade_previous_margin_threshold > 0 and previous_neural_margin is not None
            and previous_neural_margin < config.cascade_previous_margin_threshold):
        uncertainty += 0.30
        reasons.append("low_previous_neural_margin")
    uncertainty = min(1.0, uncertainty)
    escalate = uncertainty >= config.cascade_threshold
    reasons.append("threshold_passed" if escalate else "threshold_not_met")
    return ComputeCascadeDecision(
        escalate,
        config.cascade_max_rerank_limit if escalate else base,
        uncertainty,
        tuple(reasons),
    )
