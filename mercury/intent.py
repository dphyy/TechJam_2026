from __future__ import annotations

import re
from dataclasses import dataclass

from mercury.state import SessionState
from mercury.types import IntentDecision


_BUYING = re.compile(
    r"\b(?:must|need|require|exactly|under|below|at most|without|avoid|only|replacement)\b",
    re.I,
)
_BROWSING = re.compile(
    r"\b(?:ideas?|explor(?:e|ing)|browse|browsing|inspiration|suggestions?|"
    r"something for|gift(?:s|ing)?|occasion|not sure|figuring out|open to)\b",
    re.I,
)
_OVERRIDE = re.compile(
    r"\b(?:actually|instead|switch|change (?:that|it)|no longer|rather than|on second thought)\b",
    re.I,
)


@dataclass(frozen=True, slots=True)
class IntentWeights:
    """Interpretable rule weights fitted without evaluator targets.

    The defaults are the deterministic grouped-CV choice from the independently
    authored intent-v1 training split. Validation is used only as an acceptance
    check; the consumed sealed split is not used to select these values.
    """

    object: float = 0.20
    slots: float = 0.25
    hard: float = 0.00
    buying_language: float = 0.20
    browsing_language: float = 0.50
    use_case_without_object: float = 0.25
    unresolved: float = 0.25
    sparse_request: float = 0.50


def decide_intent(state: SessionState, message: str, buying_threshold: float = 0.50,
                  browsing_threshold: float = 0.50, over_general_threshold: float = 0.35,
                  weights: IntentWeights | None = None) -> IntentDecision:
    """Classify a turn without evaluator labels or stale raw-history features."""
    weights = weights or IntentWeights()
    active = [preference for preference in state.active_preferences() if preference.polarity != 0]
    positives = [preference for preference in active if preference.polarity == 1]
    hard_count = sum(preference.hard or preference.polarity == -1 for preference in active)
    has_object = any(preference.attribute == "category" for preference in positives)
    has_use_case = any(preference.attribute == "use_case" for preference in positives)
    distinct_slots = len({preference.attribute for preference in positives if preference.attribute != "other"})
    unresolved = any(preference.attribute == "other" for preference in positives)

    buying_score = (weights.object * has_object + weights.slots * min(distinct_slots, 3)
                    + weights.hard * bool(hard_count))
    browsing_score = (weights.browsing_language * bool(_BROWSING.search(message))
                      + weights.use_case_without_object * (has_use_case and not has_object)
                      + weights.unresolved * unresolved
                      + weights.sparse_request * (not has_object and distinct_slots <= 1))
    if _BUYING.search(message):
        buying_score += weights.buying_language

    # A normalized structural diagnostic is easier to interpret and transfer
    # than another fitted score. It is not treated as a target probability.
    specificity = min(1.0, (
        float(has_object) + min(distinct_slots, 3) / 3.0
        + float(bool(hard_count)) + float(not unresolved)
    ) / 4.0)
    reasons = []
    if has_object:
        reasons.append("explicit_object")
    if hard_count:
        reasons.append("hard_constraints")
    if _BUYING.search(message):
        reasons.append("buying_language")
    if _BROWSING.search(message):
        reasons.append("browsing_language")
    if has_use_case and not has_object:
        reasons.append("use_case_without_object")
    if unresolved:
        reasons.append("unresolved_open_vocabulary")
    if _OVERRIDE.search(message):
        reasons.append("intent_override")

    if buying_score >= buying_threshold and browsing_score < browsing_threshold:
        mode = "buying"
    elif browsing_score >= browsing_threshold and buying_score < buying_threshold:
        mode = "browsing"
    else:
        mode = "mixed"
    # Relative separation, not a calibrated probability. Zero evidence or a tie
    # yields zero; a one-sided decision approaches one.
    confidence = abs(buying_score - browsing_score) / max(
        buying_score, browsing_score, buying_threshold, browsing_threshold, 1e-12,
    )
    over_general = specificity < over_general_threshold and (browsing_score > 0 or not has_object)
    if over_general:
        reasons.append("over_general")
    return IntentDecision(mode, specificity, confidence, hard_count, over_general, tuple(reasons))
