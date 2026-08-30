from __future__ import annotations

from mercury.types import Candidate, Product


FACET_ATTRIBUTES = ("category", "material", "color", "style")


def facet_signature(product: Product) -> frozenset[tuple[str, str]]:
    """Stable product facets suitable for a small browsing-diversity policy."""
    return frozenset(
        (attribute, value)
        for attribute in FACET_ATTRIBUTES
        for value in product.facets.get(attribute, ())
    )


def diversify_candidates(candidates: list[Candidate], strength: float,
                         pool_limit: int = 30) -> list[Candidate]:
    """Greedily balance rank prior and facet novelty within a bounded prefix.

    The top product is anchored. Candidate scores and identities are untouched;
    only order changes. Products with no supported facets receive no novelty
    bonus, preventing missing metadata from masquerading as diversity.
    """
    if len(candidates) < 2 or strength <= 0 or pool_limit < 2:
        return list(candidates)
    protected = next((index for index, candidate in enumerate(candidates)
                      if "constraint_penalty" in candidate.route_scores
                      or "object_penalty" in candidate.route_scores), len(candidates))
    head = list(candidates[:min(pool_limit, protected)])
    if len(head) < 2:
        return list(candidates)
    ranks = {candidate.product.parent_asin: index for index, candidate in enumerate(head)}
    signatures = {candidate.product.parent_asin: facet_signature(candidate.product) for candidate in head}
    selected = [head[0]]
    remaining = head[1:]
    while remaining:
        best = None
        best_key = None
        for candidate in remaining:
            identifier = candidate.product.parent_asin
            rank = ranks[identifier]
            relevance = 1.0 - rank / max(1, len(head) - 1)
            signature = signatures[identifier]
            similarity = max((
                len(signature & signatures[item.product.parent_asin])
                / len(signature | signatures[item.product.parent_asin])
                if signature and signatures[item.product.parent_asin] else 0.0
                for item in selected
            ), default=0.0)
            novelty = 1.0 - similarity if signature else 0.0
            score = (1.0 - strength) * relevance + strength * novelty
            key = (score, -rank, identifier)
            if best_key is None or key > best_key:
                best, best_key = candidate, key
        selected.append(best)
        remaining.remove(best)
    return selected + list(candidates[len(head):])
