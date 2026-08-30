"""Target-independent fixed-budget admission for the neural reranker prefix."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from mercury.product_types import accessory_mismatch, classify_product, requested_family
from mercury.ranking import preference_evidence
from mercury.retrieval import terms
from mercury.types import Candidate, Preference, RetrievalPlan


MODES = frozenset({"prefix", "stratified", "cover", "fusion", "linear"})
RARE_SUPPORT_CEILING = 12
MODEL_SCHEMA = "mercury-admission-linear-v1"
FEATURE_VERSION = "admission-features-v1"
FEATURE_NAMES = (
    "bm25_normalized", "rank_reciprocal", "route_agreement",
    "title_coverage", "category_coverage", "feature_coverage",
    "detail_coverage", "description_coverage", "all_field_coverage",
    "object_agreement", "positive_coverage", "negative_compatibility",
    "hard_compatibility", "metadata_completeness", "price_compatibility",
)
FUSION_WEIGHTS = {
    "bm25_normalized": 0.38,
    "rank_reciprocal": 0.12,
    "route_agreement": 0.05,
    "title_coverage": 0.12,
    "category_coverage": 0.10,
    "feature_coverage": 0.08,
    "detail_coverage": 0.03,
    "description_coverage": 0.02,
    "all_field_coverage": 0.05,
    "object_agreement": 0.12,
    "positive_coverage": 0.14,
    "negative_compatibility": 0.18,
    "hard_compatibility": 0.20,
    "metadata_completeness": 0.02,
    "price_compatibility": 0.04,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class AdmissionModel:
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    catalog_sha256: str
    training_sha256: str
    validation_sha256: str
    model_sha256: str

    @classmethod
    def load(cls, path: str | Path, catalog_sha256: str) -> "AdmissionModel":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != MODEL_SCHEMA \
                or payload.get("feature_version") != FEATURE_VERSION:
            raise ValueError("Unsupported admission model")
        if payload.get("catalog_sha256") != catalog_sha256:
            raise ValueError("Admission model catalog hash mismatch")
        feature_names = tuple(payload.get("feature_names", ()))
        if feature_names != FEATURE_NAMES:
            raise ValueError("Admission model feature order mismatch")
        arrays = [tuple(payload.get(key, ())) for key in ("mean", "scale", "coefficients")]
        if any(len(values) != len(FEATURE_NAMES) for values in arrays):
            raise ValueError("Admission model vector size mismatch")
        if any(type(value) not in (int, float) or not math.isfinite(value)
               for values in arrays for value in values):
            raise ValueError("Admission model vectors must be finite")
        if any(value <= 0 for value in arrays[1]):
            raise ValueError("Admission model scales must be positive")
        intercept = payload.get("intercept")
        if type(intercept) not in (int, float) or not math.isfinite(intercept):
            raise ValueError("Admission model intercept must be finite")
        return cls(
            feature_names,
            tuple(float(value) for value in arrays[0]),
            tuple(float(value) for value in arrays[1]),
            tuple(float(value) for value in arrays[2]),
            float(intercept),
            catalog_sha256,
            str(payload.get("training_sha256", "")),
            str(payload.get("validation_sha256", "")),
            _sha256(source),
        )

    def score(self, features: dict[str, float]) -> float:
        return self.intercept + sum(
            coefficient * (features[name] - mean) / scale
            for name, mean, scale, coefficient in zip(
                self.feature_names, self.mean, self.scale, self.coefficients, strict=True,
            )
        )


def _coverage(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0
    haystack = set(terms(text))
    return len(query_terms & haystack) / len(query_terms)


def _mean_evidence(product, preferences: list[Preference]) -> float:
    if not preferences:
        return 0.0
    return sum(preference_evidence(product, preference) for preference in preferences) / len(preferences)


def admission_features(
    candidates: list[Candidate], preferences: list[Preference], plan: RetrievalPlan | None,
) -> list[dict[str, float]]:
    """Extract bounded inspectable features for every retained candidate."""
    if not candidates:
        return []
    query = plan.lexical_query if plan is not None else " ".join(
        preference.value for preference in preferences if preference.active and preference.polarity == 1
    )
    query_terms = set(terms(query))
    scores = [candidate.score for candidate in candidates]
    low, high = min(scores), max(scores)
    span = high - low
    positive = [item for item in preferences if item.active and item.polarity == 1 and item.attribute != "budget"]
    negative = [item for item in preferences if item.active and item.polarity == -1]
    hard = [item for item in preferences if item.active and item.hard and item.polarity != 0]
    budgets = [item for item in preferences if item.active and item.polarity == 1 and item.attribute == "budget"]
    requested = plan.object_types if plan is not None else tuple(
        item.value for item in positive if item.attribute == "category"
    )
    requested_families = {family for value in requested if (family := requested_family(value)) is not None}
    rows = []
    for index, candidate in enumerate(candidates):
        product = candidate.product
        product_type = classify_product(product)
        if accessory_mismatch(product, requested):
            object_agreement = -1.0
        elif requested_families and product_type.object_type in requested_families:
            object_agreement = 1.0
        else:
            object_agreement = 0.0
        populated = sum(bool(product.fields.get(field, "").strip()) for field in (
            "title", "categories", "features", "details", "description", "store",
        ))
        fields = product.fields
        rows.append({
            "bm25_normalized": 1.0 if span == 0 else (candidate.score - low) / span,
            "rank_reciprocal": 1.0 / (index + 1),
            "route_agreement": min(1.0, len([
                value for key, value in candidate.route_scores.items()
                if key not in {"constraint_penalty", "object_penalty", "price_preference", "tiny_catalog_tail"}
                and type(value) in (int, float) and value > 0
            ]) / 3.0),
            "title_coverage": _coverage(query_terms, fields.get("title", "")),
            "category_coverage": _coverage(query_terms, fields.get("categories", "")),
            "feature_coverage": _coverage(query_terms, fields.get("features", "")),
            "detail_coverage": _coverage(query_terms, fields.get("details", "")),
            "description_coverage": _coverage(query_terms, fields.get("description", "")),
            "all_field_coverage": _coverage(query_terms, product.text),
            "object_agreement": object_agreement,
            "positive_coverage": _mean_evidence(product, positive),
            "negative_compatibility": _mean_evidence(product, negative),
            "hard_compatibility": _mean_evidence(product, hard),
            "metadata_completeness": (populated + (product.price is not None)) / 7.0,
            "price_compatibility": _mean_evidence(product, budgets),
        })
    return rows


def score_all_candidates(
    candidates: list[Candidate], preferences: list[Preference], plan: RetrievalPlan | None,
    mode: str, model: AdmissionModel | None = None,
) -> tuple[list[Candidate], dict]:
    """Score the complete retained pool and return a stable target-independent order."""
    if mode not in {"fusion", "linear"}:
        raise ValueError("All-pool scoring requires fusion or linear mode")
    if mode == "linear" and model is None:
        raise ValueError("Linear admission requires a loaded model")
    rows = admission_features(candidates, preferences, plan)
    scored = []
    score_by_id = {}
    for original_rank, (candidate, features) in enumerate(zip(candidates, rows, strict=True)):
        score = model.score(features) if model is not None else sum(
            features[name] * weight for name, weight in FUSION_WEIGHTS.items()
        )
        parts = dict(candidate.route_scores)
        parts["admission_score"] = score
        scored.append((score, original_rank, Candidate(candidate.product, candidate.score, parts), features))
        score_by_id[candidate.product.parent_asin] = score
    scored.sort(key=lambda row: (-row[0], row[1], row[2].product.parent_asin))
    return [row[2] for row in scored], {
        "mode": mode,
        "feature_version": FEATURE_VERSION,
        "pool_size": len(candidates),
        "model_sha256": model.model_sha256 if model is not None else None,
        "fallback": None,
        "score_min": min(score_by_id.values()) if score_by_id else 0.0,
        "score_max": max(score_by_id.values()) if score_by_id else 0.0,
    }


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
    candidates: list[Candidate], preferences: list[Preference], limit: int, mode: str,
    plan: RetrievalPlan | None = None, model: AdmissionModel | None = None,
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
    if mode == "cover":
        return _cover(candidates, preferences, limit)
    scored, _ = score_all_candidates(candidates, preferences, plan, mode, model)
    return scored[:limit]
