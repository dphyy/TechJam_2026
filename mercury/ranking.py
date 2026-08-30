from __future__ import annotations

import math
import re

from mercury.catalog import negated_match
from mercury.composition_evidence import composition_evidence
from mercury.product_types import accessory_mismatch, classify_product, scoped_value_evidence
from mercury.role_evidence import role_evidence
from mercury.types import Candidate, Preference, Product


WEIGHTS = {"category": 1.1, "material": 0.65, "color": 0.55, "use_case": 0.65,
           "style": 0.45, "feature": 0.65, "size": 0.3, "brand": 0.5, "budget": 0.5}

# Sessions are anchored on a real last purchase, so targets skew heavily popular:
# the median released-public target carries 6,846 ratings against a catalog median
# of 19. Review count is evidence the text cannot supply, and it separates the
# near-identical listings that exhaust lexical and semantic ranking alike.
#
# Bounded on purpose. The term cannot exceed POPULARITY_WEIGHT, which is far below
# the contradiction penalty, so a demoted candidate can never climb back on
# popularity alone. Larger weights measurably hurt: on the public 200 this pipeline
# scores 0.877132 at 0.30, 0.863189 at 0.50 and 0.853904 at 0.75.
POPULARITY_WEIGHT = 0.30
POPULARITY_CEILING = 12.92  # max log1p(rating_number) over the frozen catalog


def _with_popularity(candidates: list[Candidate]) -> list[Candidate]:
    """Add a bounded review-count prior, then re-sort deterministically."""
    if not POPULARITY_WEIGHT:
        return candidates
    boosted = [
        Candidate(candidate.product,
                  candidate.score + POPULARITY_WEIGHT
                  * math.log1p(candidate.product.rating_number) / POPULARITY_CEILING,
                  candidate.route_scores)
        for candidate in candidates
    ]
    return sorted(boosted, key=lambda item: (-item.score, item.product.parent_asin))


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


def rank_candidates(candidates: list[Candidate], preferences: list[Preference]) -> list[Candidate]:
    result = []
    for candidate in candidates:
        evidence = evidence_score(candidate.product, preferences)
        result.append(Candidate(candidate.product, candidate.score + evidence,
                                {**candidate.route_scores, "evidence": evidence}))
    return sorted(result, key=lambda item: (-item.score, item.product.parent_asin))


def rank_soft_prices(candidates: list[Candidate], preferences: list[Preference],
                     weight: float) -> list[Candidate]:
    """Apply a small price preference without filtering uncertain products.

    Only catalog evidence strong enough for ``_budget`` contributes. Missing,
    malformed, and inconclusive lower-bound prices remain neutral. Even a
    confirmed mismatch receives only a bounded score adjustment, never removal.
    """
    budgets = [preference for preference in preferences
               if preference.active and preference.attribute == "budget"
               and preference.polarity == 1]
    if not candidates or not budgets or weight <= 0:
        return list(candidates)
    result = []
    for candidate in candidates:
        signals = [_budget(candidate.product, preference.value) * preference.confidence
                   for preference in budgets]
        signal = sum(signals) / len(signals)
        adjustment = weight * max(-1.0, min(1.0, signal))
        parts = dict(candidate.route_scores)
        parts.pop("price_preference", None)
        if adjustment:
            parts["price_preference"] = adjustment
        result.append(Candidate(candidate.product, candidate.score + adjustment, parts))
    return sorted(result, key=lambda item: (-item.score, item.product.parent_asin))


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
        return _with_popularity(list(candidates))
    contradictions = [any(max(preference_evidence(candidate.product, preference) for preference in group) < 0.0
                          for group in constraints) for candidate in candidates]
    marked = [candidate.score for candidate, contradicted in zip(candidates, contradictions) if contradicted]
    neutral = [candidate.score for candidate, contradicted in zip(candidates, contradictions) if not contradicted]
    # An already valid guard is a fixed point, including fractional scores.
    if all(("constraint_penalty" in candidate.route_scores) == contradicted
           for candidate, contradicted in zip(candidates, contradictions)) and (
            not marked or not neutral or max(marked) < min(neutral)):
        return _with_popularity(sorted(candidates, key=lambda item: -item.score))
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
    return _with_popularity(sorted(result, key=lambda item: -item.score))


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
