"""Catalog-only, bounded review priors; no targets, user IDs, or model calls."""

from __future__ import annotations

import math

from mercury.catalog import review_count, star_rating
from mercury.types import Candidate, Product


# Round, fixed saturation and pseudo-count, not fitted to public target outcomes.
COUNT_SATURATION = 500_000
STAR_PSEUDOCOUNT = 20
ADJUSTMENT_KEY = "review_prior_adjustment"


def review_signal(product: Product, mode: str, count_fraction: float = .5) -> float:
    if type(count_fraction) not in (int, float) or not 0 <= count_fraction <= 1:
        raise ValueError("Count fraction must be finite and in [0, 1]")
    count = review_count(product.rating_number)
    popularity = min(1.0, math.log1p(count) / math.log1p(COUNT_SATURATION))
    rating = star_rating(product.average_rating)
    raw_quality = (rating - 3.0) / 2.0 if rating is not None else 0.0
    quality = raw_quality * count / (count + STAR_PSEUDOCOUNT)
    signals = {"none": 0.0, "count": popularity, "raw_stars": raw_quality,
               "stars": quality, "mixed": count_fraction * popularity + (1 - count_fraction) * quality}
    if mode not in signals:
        raise ValueError(f"Unknown review prior mode: {mode}")
    return signals[mode]


def rank_review_prior(candidates: list[Candidate], mode: str, weight: float,
                      count_fraction: float = .5) -> list[Candidate]:
    """Replace one bonus on the current score scale, preserving guarded separation.

    Neural score replacement must drop ADJUSTMENT_KEY. A later, smaller prior
    may then operate on that new scale; reapplying on the same scale is idempotent.
    Missing metadata is neutral. This never retrieves or removes a candidate.
    """
    if type(weight) not in (int, float) or not 0 <= weight <= .30:
        raise ValueError("Review prior weight must be finite and in [0, .30]")
    if type(count_fraction) not in (int, float) or not 0 <= count_fraction <= 1:
        raise ValueError("Count fraction must be finite and in [0, 1]")
    if mode not in {"none", "count", "raw_stars", "stars", "mixed"}:
        raise ValueError(f"Unknown review prior mode: {mode}")
    if not candidates or ((not weight or mode == "none") and not any(
            ADJUSTMENT_KEY in item.route_scores for item in candidates)):
        return list(candidates)
    bases = [item.score - item.route_scores.get(ADJUSTMENT_KEY, 0.0) for item in candidates]
    for guard in ("constraint_penalty", "object_penalty"):
        clean = [score for score, item in zip(bases, candidates) if guard not in item.route_scores]
        contradicted = [score for score, item in zip(bases, candidates) if guard in item.route_scores]
        if clean and contradicted:
            # A prior ranges from -weight to +weight. Preserve even a tiny existing
            # clean/contradicted gap, without repeating semantic constraint checks.
            gap = min(clean) - max(contradicted)
            weight = min(weight, max(0.0, gap) * .499999)
    result = []
    for item, base in zip(candidates, bases):
        parts = dict(item.route_scores)
        parts.pop(ADJUSTMENT_KEY, None)
        adjustment = weight * review_signal(item.product, mode, count_fraction)
        if adjustment:
            parts[ADJUSTMENT_KEY] = adjustment
        result.append(Candidate(item.product, base + adjustment, parts))
    return sorted(result, key=lambda item: (-item.score, item.product.parent_asin))
