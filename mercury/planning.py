from __future__ import annotations

from mercury.state import SessionState
from mercury.types import IntentDecision, PlanSignal, Preference, RetrievalPlan


def _signal(preference: Preference) -> PlanSignal:
    return PlanSignal(
        preference.attribute,
        preference.value,
        preference.polarity,
        preference.hard,
        preference.confidence,
        preference.source_turn,
        preference.scope,
        preference.alternative_group,
    )


def _unique(values) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def build_retrieval_plan(state: SessionState, intent: IntentDecision) -> RetrievalPlan:
    """Distill the active ledger into route-specific, polarity-preserving contexts."""
    active = state.active_preferences()
    if state.canonical_state_semantics:
        active = sorted(
            active,
            key=lambda preference: (
                preference.attribute, preference.value, preference.polarity,
                preference.scope or "", preference.depends_on or ("", ""),
            ),
        )
    positive = [preference for preference in active if preference.polarity == 1]
    negative = [preference for preference in active if preference.polarity == -1]
    hard = tuple(_signal(preference) for preference in active
                 if preference.polarity != 0 and preference.hard)
    soft = tuple(_signal(preference) for preference in positive if not preference.hard)
    scoped = tuple(_signal(preference) for preference in active
                   if preference.polarity != 0 and preference.scope is not None)
    objects = _unique(preference.value for preference in positive if preference.attribute == "category")
    use_cases = _unique(preference.value for preference in positive if preference.attribute == "use_case")
    positive_terms = _unique(preference.value for preference in positive)
    negative_terms = _unique(preference.value for preference in negative)
    lexical_query = state.query()
    semantic_parts = _unique((*objects, *use_cases, *positive_terms))
    semantic_query = " ".join(semantic_parts)
    context_lines = [f"Mode: {intent.mode}"]
    if objects:
        context_lines.append("Object: " + "; ".join(objects))
    must = _unique(signal.value for signal in hard if signal.polarity == 1 and signal.attribute != "budget")
    must_not = _unique(signal.value for signal in hard if signal.polarity == -1)
    if must:
        context_lines.append("Must have: " + "; ".join(must))
    if must_not:
        context_lines.append("Must not have: " + "; ".join(must_not))
    preferred = _unique(signal.value for signal in soft if signal.attribute != "use_case")
    if preferred:
        context_lines.append("Prefer: " + "; ".join(preferred))
    if use_cases:
        context_lines.append("Preferred use: " + "; ".join(use_cases))
    budgets = _unique(preference.value for preference in active
                      if preference.active and preference.polarity == 1 and preference.attribute == "budget")
    if budgets:
        context_lines.append("Budget: " + "; ".join(budgets))
    for signal in scoped:
        context_lines.append(f"Scoped {signal.attribute}: {signal.value} -> {signal.scope}")
    return RetrievalPlan(
        intent.mode,
        objects,
        objects,
        positive_terms,
        negative_terms,
        hard,
        soft,
        use_cases,
        scoped,
        (semantic_query,) if semantic_query else (),
        lexical_query,
        "\n".join(context_lines),
        (),
    )
