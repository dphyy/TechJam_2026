"""Cheap, deterministic gating before catalog and neural work."""

from __future__ import annotations

from mercury.config import Config
from mercury.state import SessionState
from mercury.types import IntentDecision, RetrievalPlan, RetrievalSufficiencyDecision


def decide_retrieval_sufficiency(state: SessionState, intent: IntentDecision,
                                 plan: RetrievalPlan, config: Config, turn: int,
                                 deferred_turns: int) -> RetrievalSufficiencyDecision:
    """Choose a bounded action without inspecting candidates or target labels."""
    if not config.retrieval_sufficiency_gate:
        return RetrievalSufficiencyDecision("retrieve", True, ("gate_disabled",))
    if turn >= 10:
        return RetrievalSufficiencyDecision("retrieve", True, ("final_turn",))
    if config.question_policy == "none":
        return RetrievalSufficiencyDecision("retrieve", True, ("question_policy_disabled",))
    if deferred_turns >= config.max_deferred_turns:
        return RetrievalSufficiencyDecision("retrieve", True, ("defer_budget_exhausted",))
    if turn > 1 and state.last_update_informative:
        return RetrievalSufficiencyDecision("retrieve", True, ("productive_answer",))
    if not intent.over_general or intent.specificity >= config.minimum_retrieval_specificity:
        return RetrievalSufficiencyDecision("retrieve", True, ("sufficient_specificity",))
    if plan.object_types or intent.hard_constraint_count:
        return RetrievalSufficiencyDecision("retrieve", True, ("object_or_hard_constraint",))
    known = {preference.attribute for preference in state.active_preferences()}
    named_available = any(
        attribute not in known and attribute not in state.unproductive_attributes
        and not state.asked_counts.get(attribute, 0)
        for attribute in ("category", "use_case", "material", "style", "feature")
    )
    other_available = (
        "other" not in state.unproductive_attributes
        and state.asked_counts.get("other", 0) < config.other_question_limit
    )
    if not named_available and not other_available:
        return RetrievalSufficiencyDecision("retrieve", True, ("no_productive_question",))
    return RetrievalSufficiencyDecision(
        config.insufficient_action,
        False,
        ("over_general", "low_specificity", "no_object", "bounded_deferral"),
    )
