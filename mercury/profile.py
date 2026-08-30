from __future__ import annotations

import re

from mercury.ranking import value_matches
from mercury.types import Candidate


def distill_profile(profile: object) -> tuple[str, ...]:
    """Extract bounded anonymous profile tags; malformed profiles become empty priors."""
    if not isinstance(profile, dict):
        return ()
    tags = profile.get("preference_tags", [])
    if not isinstance(tags, list):
        return ()
    result = []
    for tag in tags[:8]:
        if not isinstance(tag, str):
            continue
        normalized = re.sub(r"\s+", " ", tag.lower()).strip()
        if normalized and len(normalized) <= 40 and len(normalized.split()) <= 4:
            result.append(normalized)
    return tuple(dict.fromkeys(result))


def rank_profile_prior(candidates: list[Candidate], profile_terms: tuple[str, ...],
                       weight: float) -> list[Candidate]:
    if not candidates or not profile_terms or weight <= 0:
        return list(candidates)
    result = []
    for candidate in candidates:
        matches = sum(value_matches(term, candidate.product.text) for term in profile_terms)
        prior = weight * matches / len(profile_terms)
        result.append(Candidate(candidate.product, candidate.score + prior,
                                {**candidate.route_scores, "profile_prior": prior}))
    return sorted(result, key=lambda item: (-item.score, item.product.parent_asin))
