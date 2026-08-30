from __future__ import annotations

import re
from dataclasses import replace

from mercury.catalog import negated_match
from mercury.composition_evidence import composition_evidence
from mercury.product_types import accessory_mismatch, classify_product, scoped_value_evidence
from mercury.role_evidence import role_evidence
from mercury.types import Candidate, PlanSignal, Preference, Product, RetrievalPlan


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


def _token_phrase_signals(needle: str, haystack: str) -> tuple[bool, bool]:
    tokens = [token for token in re.findall(r"[a-z0-9]+", needle.lower()) if len(token) > 2]
    if len(tokens) < 2:
        return False, False
    text = haystack.lower()
    token_set = set(tokens)
    occurrences = []
    for token in token_set:
        matches = list(re.finditer(r"(?<!\w)" + re.escape(token) + r"s?(?!\w)", text))
        if not matches:
            return False, False
        occurrences.extend((match.start(), token, match) for match in matches)
    occurrences.sort(key=lambda item: item[0])

    counts: dict[str, int] = {}
    left = 0
    best: tuple[int, int] | None = None
    for right, (_, token, _) in enumerate(occurrences):
        counts[token] = counts.get(token, 0) + 1
        while len(counts) == len(token_set):
            width = occurrences[right][0] - occurrences[left][0]
            candidate = (left, right)
            if best is None or (width, occurrences[left][0], occurrences[right][0]) < (
                    occurrences[best[1]][0] - occurrences[best[0]][0],
                    occurrences[best[0]][0], occurrences[best[1]][0]):
                best = candidate
            left_token = occurrences[left][1]
            counts[left_token] -= 1
            if counts[left_token] == 0:
                del counts[left_token]
            left += 1
    if best is None or occurrences[best[1]][0] - occurrences[best[0]][0] > 80:
        return False, False
    selected = {}
    for _, token, match in occurrences[best[0]:best[1] + 1]:
        selected.setdefault(token, match)
    contradicted = any(negated_match(text, match.start(), match.end()) for match in selected.values())
    return not contradicted, contradicted


def value_matches(needle: str, haystack: str) -> bool:
    return _value_signals(needle, haystack)[0] or _token_phrase_signals(needle, haystack)[0]


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
    if preference.scope is not None:
        return scoped_value_evidence(product, preference.value, preference.scope) * preference.polarity
    support = contradiction = 0.0
    sources = [(item.value, item.confidence) for item in product.evidence
               if item.attribute == preference.attribute]
    # Open-vocabulary values are useful when explicitly present. Span checks
    # retain qualified values such as faux leather without supporting leather.
    sources.extend((product.fields.get(source, ""), 0.6 if source in ("title", "categories") else 0.4)
                   for source in ("title", "categories", "features", "details", "description", "store"))
    for value, confidence in sources:
        positive, negative = _value_signals(preference.value, value)
        if preference.attribute == "other" and not positive and not negative:
            positive, negative = _token_phrase_signals(preference.value, value)
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


def _plan_preference(signal: PlanSignal) -> Preference:
    return Preference(
        signal.attribute, signal.value, signal.source_turn, "typed retrieval plan",
        hard=signal.hard, polarity=signal.polarity, confidence=signal.confidence,
        scope=signal.scope, alternative_group=signal.alternative_group,
    )


def typed_plan_evidence(product: Product, plan: RetrievalPlan) -> float:
    """Bounded catalog evidence for typed hard/soft signals, excluding price."""
    groups: dict[str | tuple[str, str], list[float]] = {}
    for signal in (*plan.hard_constraints, *plan.soft_preferences):
        if signal.attribute == "budget" or signal.polarity == 0:
            continue
        key: str | tuple[str, str] = signal.attribute
        if signal.polarity == 1 and signal.alternative_group is not None:
            key = (signal.attribute, signal.alternative_group)
        elif key == "feature" or signal.polarity < 0:
            key += ":" + signal.value + ":" + str(signal.polarity)
        evidence = preference_evidence(product, _plan_preference(signal))
        strength = 1.0 if signal.hard else 0.6 * signal.confidence
        if evidence < 0:
            strength *= 1.5
        groups.setdefault(key, []).append(WEIGHTS.get(signal.attribute, 0.4) * evidence * strength)
    return sum(max(signals) for signals in groups.values())


def rank_typed_plan(candidates: list[Candidate], plan: RetrievalPlan,
                    weight: float, active: bool) -> list[Candidate]:
    """Attach typed evidence in shadow mode or apply its weighted adjustment."""
    result = []
    for candidate in candidates:
        parts = dict(candidate.route_scores)
        score = candidate.score - parts.pop("typed_plan_adjustment", 0.0)
        evidence = typed_plan_evidence(candidate.product, plan)
        parts["typed_plan_evidence"] = evidence
        adjustment = weight * evidence if active else 0.0
        if adjustment:
            parts["typed_plan_adjustment"] = adjustment
        result.append(Candidate(candidate.product, score + adjustment, parts))
    if not active:
        return result
    return sorted(result, key=lambda item: (-item.score, item.product.parent_asin))


def hard_plan_evidence(product: Product, plan: RetrievalPlan) -> float:
    """Catalog evidence for hard non-price requirements only."""
    return typed_plan_evidence(product, replace(plan, soft_preferences=()))


def rank_hard_plan(candidates: list[Candidate], plan: RetrievalPlan,
                   weight: float) -> list[Candidate]:
    """Promote known hard-requirement support without turning unknown into failure."""
    result = []
    for candidate in candidates:
        parts = dict(candidate.route_scores)
        score = candidate.score - parts.pop("intent_hard_adjustment", 0.0)
        evidence = hard_plan_evidence(candidate.product, plan)
        parts["intent_hard_evidence"] = evidence
        adjustment = weight * evidence
        if adjustment:
            parts["intent_hard_adjustment"] = adjustment
        result.append(Candidate(candidate.product, score + adjustment, parts))
    return sorted(result, key=lambda item: (-item.score, item.product.parent_asin))


def rank_candidates(candidates: list[Candidate], preferences: list[Preference]) -> list[Candidate]:
    result = []
    for candidate in candidates:
        evidence = evidence_score(candidate.product, preferences)
        result.append(Candidate(candidate.product, candidate.score + evidence,
                                {**candidate.route_scores, "evidence": evidence}))
    return sorted(result, key=lambda item: (-item.score, item.product.parent_asin))


def rank_soft_prices(candidates: list[Candidate], preferences: list[Preference],
                     weight: float) -> list[Candidate]:
    """Apply a small structured price preference without filtering products.

    Firm limits retain binary fit evidence. Soft figures use continuous
    proximity to their target. Missing, malformed, and inconclusive lower-bound
    prices remain neutral. Even a confirmed mismatch receives only a bounded
    score adjustment, never removal.
    """
    budgets = [preference for preference in preferences
               if preference.active and preference.attribute == "budget"
               and preference.polarity == 1]
    if not candidates or not budgets or weight <= 0:
        return list(candidates)
    result = []
    for candidate in candidates:
        signals = [budget_preference_score(candidate.product, preference) * preference.confidence
                   for preference in budgets]
        signal = sum(signals) / len(signals)
        adjustment = weight * max(-1.0, min(1.0, signal))
        parts = dict(candidate.route_scores)
        score = candidate.score - parts.pop("price_preference", 0.0)
        if adjustment:
            parts["price_preference"] = adjustment
        result.append(Candidate(candidate.product, score + adjustment, parts))
    return sorted(result, key=lambda item: (-item.score, item.product.parent_asin))


def budget_preference_score(product: Product, preference: Preference) -> float:
    """Binary firm-budget evidence or continuous soft-target proximity."""
    if preference.hard:
        return _budget(product, preference.value)
    if product.price is None or product.price_lower_bound or product.price < 0:
        return 0.0
    numbers = [float(number) for number in re.findall(r"\d+(?:\.\d+)?", preference.value)]
    if not numbers:
        return 0.0
    if len(numbers) == 2:
        low, high = sorted(numbers[:2])
        target = (low + high) / 2.0
        scale = max((high - low) / 2.0, target, 1.0)
    else:
        target = numbers[0]
        scale = max(target, 1.0)
    distance = abs(product.price - target) / scale
    return max(-1.0, 1.0 - 2.0 * distance)


def rank_role_evidence(candidates: list[Candidate], preferences: list[Preference]) -> tuple[list[Candidate], dict[str, list[dict]]]:
    """Apply bounded source-span role support without turning unknowns into constraints."""
    result = []
    diagnostics: dict[str, list[dict]] = {}
    for candidate in candidates:
        evidence = role_evidence(candidate.product, preferences)
        parts = dict(candidate.route_scores)
        if evidence.score:
            parts["role_evidence"] = evidence.score
            diagnostics[candidate.product.parent_asin] = [
                {"preference": witness.preference, "material": witness.material, "role": witness.role,
                 "source": witness.source, "span": witness.span, "start": witness.start, "end": witness.end}
                for witness in evidence.witnesses
            ]
        result.append(Candidate(candidate.product, candidate.score + evidence.score, parts))
    return sorted(result, key=lambda item: (-item.score, item.product.parent_asin)), diagnostics


def rank_composition_evidence(candidates: list[Candidate], preferences: list[Preference]) -> tuple[list[Candidate], dict[str, list[dict]]]:
    """Apply one direct composition adjustment after bounded neural reranking."""
    result = []
    diagnostics: dict[str, list[dict]] = {}
    for candidate in candidates:
        evidence = composition_evidence(candidate.product, preferences)
        parts = dict(candidate.route_scores)
        if evidence.score:
            parts["composition_evidence"] = evidence.score
            diagnostics[candidate.product.parent_asin] = [
                {"preference": witness.preference, "material": witness.material, "source": witness.source,
                 "span": witness.span, "start": witness.start, "end": witness.end}
                for witness in evidence.witnesses
            ]
        result.append(Candidate(candidate.product, candidate.score + evidence.score, parts))
    return sorted(result, key=lambda item: (-item.score, item.product.parent_asin)), diagnostics


def _constraint_groups(preferences: list[Preference]) -> list[list[Preference]]:
    constraints: list[list[Preference]] = []
    alternatives: dict[tuple[str, str], list[Preference]] = {}
    for preference in preferences:
        if not preference.active or preference.polarity == 0:
            continue
        if preference.polarity == 1 and preference.alternative_group is not None:
            key = (preference.attribute, preference.alternative_group)
            alternatives.setdefault(key, []).append(preference)
        elif preference.hard:
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


def rank_soft_negatives(candidates: list[Candidate], preferences: list[Preference],
                        weight: float) -> list[Candidate]:
    """Apply a bounded demotion for explicitly soft exclusions.

    Matching a soft negative should lower priority without acting as a hard
    guard. Unknown evidence and explicit evidence that the value is absent are
    neutral. The stored adjustment is replaced so repeated application is a
    fixed point.
    """
    avoided = [preference for preference in preferences
               if preference.active and preference.polarity == -1 and not preference.hard]
    if not candidates or (not avoided and not any(
            "soft_negative_preference" in candidate.route_scores for candidate in candidates)):
        return list(candidates)
    result = []
    for candidate in candidates:
        parts = dict(candidate.route_scores)
        score = candidate.score - parts.pop("soft_negative_preference", 0.0)
        adjustment = weight * sum(
            min(0.0, preference_evidence(candidate.product, preference)) * preference.confidence
            for preference in avoided
        )
        if adjustment:
            parts["soft_negative_preference"] = adjustment
        result.append(Candidate(candidate.product, score + adjustment, parts))
    return sorted(result, key=lambda item: (-item.score, item.product.parent_asin))


def rank_product_compatibility(candidates: list[Candidate], preferences: list[Preference]) -> list[Candidate]:
    """Demote proven accessory/component mismatches while preserving unknown types."""
    requested = tuple(preference.value for preference in preferences
                      if preference.active and preference.polarity == 1 and preference.attribute == "category")
    mismatches = [accessory_mismatch(candidate.product, requested) for candidate in candidates]
    if not candidates or not any(mismatches):
        return list(candidates)
    if all(("object_penalty" in candidate.route_scores) == mismatch
           for candidate, mismatch in zip(candidates, mismatches)):
        return sorted(candidates, key=lambda item: -item.score)
    scores = [candidate.score + candidate.route_scores.get("object_penalty", 0.0)
              for candidate in candidates]
    penalty = max(scores) - min(scores) + 1.0
    result = []
    for candidate, score, mismatch in zip(candidates, scores, mismatches):
        parts = dict(candidate.route_scores)
        parts.pop("object_penalty", None)
        if mismatch:
            score -= penalty
            parts["object_penalty"] = penalty
        product_type = classify_product(candidate.product)
        parts["product_type_confidence"] = product_type.confidence
        result.append(Candidate(candidate.product, score, parts))
    return sorted(result, key=lambda item: -item.score)
