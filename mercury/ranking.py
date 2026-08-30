from __future__ import annotations

import re

from mercury.catalog import negated_match
from mercury.types import Candidate, Preference, Product


WEIGHTS = {"category": 1.1, "material": 0.65, "color": 0.55, "use_case": 0.65,
           "style": 0.45, "feature": 0.65, "size": 0.3, "brand": 0.5, "budget": 0.5}


def _variants(value: str) -> set[str]:
    value = value.lower().replace("gray", "grey")
    variants = {value}
    if value.endswith("ies"):
        variants.add(value[:-3] + "y")
    elif value.endswith("s") and not value.endswith("ss"):
        variants.add(value[:-1])
        if value.endswith("ses"):
            variants.add(value[:-2])
    else:
        variants.add(value + "s")
    return variants


def _value_signals(needle: str, haystack: str) -> tuple[bool, bool]:
    haystack = haystack.lower().replace("gray", "grey")
    supported = contradicted = False
    for value in _variants(needle):
        for match in re.finditer(r"(?<!\w)" + re.escape(value) + r"(?!\w)", haystack):
            if negated_match(haystack, match.start(), match.end()):
                contradicted = True
            else:
                supported = True
            if supported and contradicted:
                return True, True
    return supported, contradicted


def value_matches(needle: str, haystack: str) -> bool:
    return _value_signals(needle, haystack)[0]


def _budget(product: Product, value: str) -> float:
    if product.price is None:
        return 0.0
    numbers = [float(number) for number in re.findall(r"\d+(?:\.\d+)?", value)]
    if not numbers:
        return 0.0
    lower = 0.0
    upper = float("inf")
    if len(numbers) == 2:
        lower, upper = numbers
    elif value.startswith(">="):
        lower = numbers[0]
    else:
        upper = numbers[0]
    if product.price > upper:
        return -1.0
    if product.price_lower_bound:
        # A lower bound only establishes a minimum, never a maximum.
        return 1.0 if upper == float("inf") and product.price >= lower else 0.0
    return 1.0 if lower <= product.price <= upper else -1.0


def preference_evidence(product: Product, preference: Preference) -> float:
    """Positive support, observed contradiction, or zero for unknown information."""
    if not preference.active or preference.polarity == 0:
        return 0.0
    if preference.attribute == "budget":
        return _budget(product, preference.value) * preference.polarity
    support = contradiction = 0.0
    sources = [(item.value, item.confidence) for item in product.evidence
               if item.attribute == preference.attribute]
    # Open-vocabulary values are useful when explicitly present. Span checks
    # retain qualified values such as faux leather without supporting leather.
    sources.extend((product.fields.get(source, ""), 0.6 if source in ("title", "categories") else 0.4)
                   for source in ("title", "categories", "features", "details", "description", "store"))
    for value, confidence in sources:
        positive, negative = _value_signals(preference.value, value)
        if positive:
            support = max(support, confidence)
        if negative:
            contradiction = max(contradiction, confidence)
    # Conflicting sources do not prove either side; absence is also unknown.
    if support and contradiction:
        return 0.0
    return (support - contradiction) * preference.polarity


def evidence_score(product: Product, preferences: list[Preference]) -> float:
    # Alternatives in a single facet should not require a product to satisfy
    # every value. Independent features and explicit exclusions remain additive.
    groups: dict[str | tuple[str, str], list[float]] = {}
    for preference in preferences:
        if preference.active and preference.polarity != 0:
            key: str | tuple[str, str] = preference.attribute
            if preference.polarity == 1 and preference.alternative_group is not None:
                key = (preference.attribute, preference.alternative_group)
            elif key == "feature" or preference.polarity < 0:
                key += ":" + preference.value + ":" + str(preference.polarity)
            strength = 1.25 if preference.hard else preference.confidence
            signal = preference_evidence(product, preference)
            if signal < 0:
                strength *= 2.0
            groups.setdefault(key, []).append(WEIGHTS.get(preference.attribute, 0.4) * signal * strength)
    return sum(max(signals) for signals in groups.values())


def rank_candidates(candidates: list[Candidate], preferences: list[Preference]) -> list[Candidate]:
    result = []
    for candidate in candidates:
        evidence = evidence_score(candidate.product, preferences)
        result.append(Candidate(candidate.product, candidate.score + evidence,
                                {**candidate.route_scores, "evidence": evidence}))
    return sorted(result, key=lambda item: (-item.score, item.product.parent_asin))


def _constraint_groups(preferences: list[Preference]) -> list[list[Preference]]:
    constraints: list[list[Preference]] = []
    alternatives: dict[tuple[str, str], list[Preference]] = {}
    for preference in preferences:
        if not preference.active or preference.polarity == 0:
            continue
        if preference.polarity == 1 and preference.alternative_group is not None:
            key = (preference.attribute, preference.alternative_group)
            alternatives.setdefault(key, []).append(preference)
        elif preference.hard or preference.polarity < 0:
            constraints.append([preference])
    # Hardness belongs to the group; even a soft member can satisfy its OR.
    constraints.extend(group for group in alternatives.values() if any(item.hard for item in group))
    return constraints


def rank_constraints(candidates: list[Candidate], preferences: list[Preference]) -> list[Candidate]:
    """Demote only observed hard/negative contradictions, retaining unknowns.

    The stored penalty is replaced, not accumulated. Additive rankers may keep
    it; rankers replacing scores must discard it with the old score scale.
    """
    constraints = _constraint_groups(preferences)
    if not candidates or (not constraints and not any(
            "constraint_penalty" in candidate.route_scores for candidate in candidates)):
        return list(candidates)
    contradictions = [any(max(preference_evidence(candidate.product, preference) for preference in group) < 0.0
                          for group in constraints) for candidate in candidates]
    marked = [candidate.score for candidate, contradicted in zip(candidates, contradictions) if contradicted]
    neutral = [candidate.score for candidate, contradicted in zip(candidates, contradictions) if not contradicted]
    # An already valid guard is a fixed point, including fractional scores.
    if all(("constraint_penalty" in candidate.route_scores) == contradicted
           for candidate, contradicted in zip(candidates, contradictions)) and (
            not marked or not neutral or max(marked) < min(neutral)):
        return sorted(candidates, key=lambda item: -item.score)
    scores = [candidate.score + candidate.route_scores.get("constraint_penalty", 0.0)
              for candidate in candidates]
    # A shared offset preserves score gaps and ties among contradicted items
    # while placing every one below the remaining pool on the same scale.
    penalty = max(scores) - min(scores) + 1.0
    result = []
    for candidate, score, contradicted in zip(candidates, scores, contradictions):
        parts = dict(candidate.route_scores)
        parts.pop("constraint_penalty", None)
        if contradicted:
            score -= penalty
            parts["constraint_penalty"] = penalty
        result.append(Candidate(candidate.product, score, parts))
    return sorted(result, key=lambda item: -item.score)
