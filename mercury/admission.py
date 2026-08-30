"""Deterministic fixed-budget admission for the neural reranker prefix."""

from __future__ import annotations

from mercury.ranking import preference_evidence
from mercury.types import Candidate, Preference


MODES = frozenset({"prefix", "stratified", "cover"})
RARE_SUPPORT_CEILING = 12


def _positive_group(preference: Preference) -> tuple[str, str, str]:
    if preference.alternative_group is not None:
        return preference.attribute, "alternative", preference.alternative_group
    return preference.attribute, "value", preference.value


def _prefix(candidates: list[Candidate], limit: int) -> list[Candidate]:
    return list(candidates[:limit])


def _stratified(candidates: list[Candidate], limit: int) -> list[Candidate]:
    """Keep the leading half and evenly sample the remaining ranked tail."""
    if len(candidates) <= limit:
        return list(candidates)
    anchors = min(limit, max(1, limit // 2))
    selected = list(range(anchors))
    remaining = limit - anchors
    if remaining:
        tail_start, tail_end = anchors, len(candidates) - 1
        if remaining == 1:
            selected.append(tail_end)
        else:
            selected.extend(
                tail_start + (tail_end - tail_start) * position // (remaining - 1)
                for position in range(remaining)
            )
    return [candidates[index] for index in selected]


def _cover(candidates: list[Candidate], preferences: list[Preference], limit: int) -> list[Candidate]:
    """Preserve leaders then cover positive hard or sparse evidence groups.

    This only reads active user preferences and ordinary catalog evidence. It is
    deliberately not allowed to inspect evaluator labels, targets, or future
    user turns.
    """
    if len(candidates) <= limit:
        return list(candidates)
    anchors = min(limit, max(1, limit // 2))
    active = [item for item in preferences if item.active and item.polarity == 1]
    grouped: dict[tuple[str, str, str], list[Preference]] = {}
    for preference in active:
        grouped.setdefault(_positive_group(preference), []).append(preference)
    if not grouped:
        return _prefix(candidates, limit)

    support: dict[tuple[str, str, str], set[int]] = {}
    weights: dict[tuple[str, str, str], int] = {}
    for key, group in grouped.items():
        matching = {
            index
            for index, candidate in enumerate(candidates)
            if any(preference_evidence(candidate.product, preference) > 0 for preference in group)
        }
        if not matching:
            continue
        hard = any(preference.hard for preference in group)
        if hard or len(matching) <= RARE_SUPPORT_CEILING:
            support[key] = matching
            weights[key] = 2 if hard else 1
    if not support:
        return _prefix(candidates, limit)

    selected = list(range(anchors))
    chosen = set(selected)
    covered = {key for key, members in support.items() if members & chosen}
    while len(selected) < limit:
        best_index = None
        best_benefit = 0
        for index, candidate in enumerate(candidates):
            if index in chosen:
                continue
            benefit = sum(weight for key, weight in weights.items() if key not in covered and index in support[key])
            if benefit > best_benefit or (
                benefit == best_benefit and benefit > 0 and best_index is not None
                and candidate.product.parent_asin < candidates[best_index].product.parent_asin
            ):
                best_index, best_benefit = index, benefit
        if best_index is None or best_benefit == 0:
            break
        selected.append(best_index)
        chosen.add(best_index)
        covered.update(key for key, members in support.items() if best_index in members)
    selected.extend(index for index in range(len(candidates)) if index not in chosen)
    return [candidates[index] for index in selected[:limit]]


def select_rerank_prefix(
    candidates: list[Candidate], preferences: list[Preference], limit: int, mode: str
) -> list[Candidate]:
    """Return exactly the legal reranker prefix without changing candidate IDs."""
    if mode not in MODES:
        raise ValueError(f"Unsupported rerank admission mode: {mode!r}")
    if type(limit) is not int or limit < 1:
        raise ValueError("Rerank admission limit must be a positive integer")
    if mode == "prefix":
        return _prefix(candidates, limit)
    if mode == "stratified":
        return _stratified(candidates, limit)
    return _cover(candidates, preferences, limit)
