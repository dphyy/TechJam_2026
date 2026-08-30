from __future__ import annotations

import re
from dataclasses import dataclass

from mercury.state import SessionState
from mercury.types import IntentDecision


_COMMITTED = re.compile(
    r"\b(?:must|require|exactly|under|below|at most|without|avoid|"
    r"non[- ]negotiable|deal[- ]?breaker)\b",
    re.I,
)
_DIRECT_REQUEST = re.compile(
    r"\b(?:i\s+(?:need|want)|i(?:'m| am)\s+(?:looking for|after)|"
    r"find me|show me|please find)\b",
    re.I,
)
_EXPLORATORY = re.compile(
    r"\b(?:ideas?|explor(?:e|ing)|browse|browsing|inspiration|suggestions?|"
    r"not sure|figuring out|open to|maybe|might|possibly|considering)\b",
    re.I,
)
_SCENARIO_BROWSING = re.compile(
    r"\b(?:something for|gift(?:s|ing)?|occasion)\b",
    re.I,
)
_NONCOMMITMENT = re.compile(
    r"\bwithout\s+(?:choosing|committing|settling|deciding)\b",
    re.I,
)
_VAGUE_REQUEST = re.compile(r"\b(?:something|anything|ideas?|suggestions?|help)\b", re.I)
_MIXED = re.compile(
    r"\b(?:surprise me|flexible|torn between|somewhere between|don't know which|"
    r"do not know which|other (?:options|ideas|solutions|alternatives)|show alternatives|"
    r"better options even if|if (?:that|one) exists|one .* for both|"
    r"(?:heavy|substantial).*(?:weightless|no heft))\b",
    re.I,
)
_RELAXATION = re.compile(
    r"\b(?:forget|drop|remove|release|does(?:n't| not) matter|no preference|"
    r"anything is fine|whatever works|any (?:color|material|style|brand|size))\b",
    re.I,
)
_CORRECTION = re.compile(
    r"\b(?:not that|I meant|correction|scratch that|don't want|do not want)\b",
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
                  weights: IntentWeights | None = None,
                  routing_confidence_threshold: float = 0.65) -> IntentDecision:
    """Classify a turn without evaluator labels or stale raw-history features."""
    if not 0 <= routing_confidence_threshold <= 1:
        raise ValueError("routing_confidence_threshold must be in [0, 1]")
    weights = weights or IntentWeights()
    active = [preference for preference in state.active_preferences() if preference.polarity != 0]
    positives = [preference for preference in active if preference.polarity == 1]
    hard_count = sum(preference.hard or preference.polarity == -1 for preference in active)
    has_object = any(preference.attribute == "category" for preference in positives)
    has_use_case = any(preference.attribute == "use_case" for preference in positives)
    distinct_slots = len({preference.attribute for preference in positives if preference.attribute != "other"})
    unresolved = any(preference.attribute == "other" for preference in positives)
    exploratory_language = bool(_EXPLORATORY.search(message))
    scenario_language = bool(_SCENARIO_BROWSING.search(message))
    browsing_language = exploratory_language or (scenario_language and not has_object)
    committed_language = bool(_COMMITTED.search(message)) and not _NONCOMMITMENT.search(message)
    direct_request_phrase = bool(_DIRECT_REQUEST.search(message))
    direct_request = (direct_request_phrase and not browsing_language
                      and (has_object or not _VAGUE_REQUEST.search(message)))
    mixed_language = bool(_MIXED.search(message))
    buying_language = committed_language or direct_request

    buying_score = (weights.object * has_object + weights.slots * min(distinct_slots, 3)
                    + weights.hard * bool(hard_count))
    browsing_score = (weights.browsing_language * browsing_language
                      + weights.use_case_without_object * (has_use_case and not has_object)
                      + weights.unresolved * unresolved
                      + weights.sparse_request * (not has_object and distinct_slots <= 1))
    if buying_language:
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
    if buying_language:
        reasons.append("buying_language")
    if committed_language:
        reasons.append("committed_language")
    if direct_request:
        reasons.append("direct_request")
    if mixed_language:
        reasons.append("mixed_language")
    if browsing_language:
        reasons.append("browsing_language")
    if has_use_case and not has_object:
        reasons.append("use_case_without_object")
    if unresolved:
        reasons.append("unresolved_open_vocabulary")
    delta = state.last_state_delta
    removed_without_replacement = bool(delta.removed) and not delta.added
    if _RELAXATION.search(message) or removed_without_replacement:
        event = "relaxation"
    elif delta.kind == "polarity_change" or (
            _CORRECTION.search(message) and not delta.explicit_replacement):
        event = "correction"
    elif delta.explicit_replacement or delta.kind in {"replacement", "category_change"} \
            or _OVERRIDE.search(message):
        event = "override"
    else:
        event = "continue"
    if event != "continue":
        reasons.append(f"intent_{event}")

    # Phrase-level collisions are deliberate mixed intent, not a tie to be broken
    # by whichever independently weighted keyword happens to be stronger.
    if event == "relaxation" and browsing_language and not committed_language:
        mode = "browsing"
    elif mixed_language and (has_object or committed_language or direct_request_phrase or distinct_slots):
        mode = "mixed"
    elif browsing_language and committed_language and (has_object or hard_count):
        mode = "mixed"
    elif buying_score >= buying_threshold and browsing_score < browsing_threshold:
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
    action_mode = mode if confidence >= routing_confidence_threshold else "mixed"
    if action_mode != mode:
        reasons.append("low_confidence_safe_fallback")
    return IntentDecision(
        mode, specificity, confidence, hard_count, over_general, tuple(reasons), event, action_mode,
    )
