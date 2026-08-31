"""Small catalog-calibrated scorer with unchanged safe tiers and candidate membership."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path

from .product_features import (
    FACET_PATTERNS,
    FIELD_ORDER,
    ProductFeatureStore,
    ProductFeatures,
    evidence_product,
    resolve_query,
    terms,
)
from .retrieval import SearchResult


FEATURE_VERSION = "catalog-fields-v1"
TOKEN_FIELDS = ("title", "features", "details", "description")
FEATURE_NAMES = tuple(f"{field}_token_coverage" for field in TOKEN_FIELDS) + tuple(
    f"{field}_phrase_support" for field in TOKEN_FIELDS
) + ("whole_field_support", "best_field_coverage", "cross_field_coverage",
     "missing_information", "facet_contradiction", "exclusion_match", "category_coverage")
FEATURE_SCALES = (1.0,) * len(FEATURE_NAMES)
MODEL_KEYS = frozenset({"feature_names", "scales", "weights", "catalog_sha256", "config_sha256"})
MAX_WEIGHT = 8.0
MAX_MODEL_BYTES = 16_384


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_field(value: object) -> str:
    if isinstance(value, dict):
        return " ; ".join(f"{key}: {text_field(child)}" for key, child in value.items())
    if isinstance(value, list):
        return " ; ".join(text_field(child) for child in value)
    return "" if value is None else str(value)


def product_features(product: dict) -> ProductFeatures:
    existing = product.get("_features")
    if isinstance(existing, ProductFeatures):
        return existing
    store = ProductFeatureStore(max_size=1)
    return store.add(str(product.get("parent_asin", "candidate")),
                     {name: text_field(product.get(name)) for name in FIELD_ORDER})


def feature_vector(product: dict | ProductFeatures, query, category: str = "") -> tuple[float, ...]:
    features = product if isinstance(product, ProductFeatures) else product_features(product)
    query = resolve_query(features, query)
    positive = [item for item in query.evidence if item.source not in {"category", "exclusion"}
                and item.tokens and not item.is_budget]
    field_coverage = [0.0] * len(TOKEN_FIELDS)
    phrase_support = [0.0] * len(TOKEN_FIELDS)
    whole = best = cross = missing = contradiction = 0.0
    for item in positive:
        view = evidence_product(features, item)
        sequences = view.affirmed_sequences
        if len(sequences) != len(FIELD_ORDER):
            raise ValueError("Incomplete field features")
        wanted = set(item.tokens)
        union = set().union(*(set(sequence) for sequence in sequences))
        covered = len(wanted & union) / len(wanted)
        cross += covered
        missing += 1 - covered
        best += max((len(wanted & set(sequence)) / len(wanted) for sequence in sequences), default=0.0)
        whole += any(tuple(item.tokens) == sequence for sequence in sequences)
        for index, name in enumerate(TOKEN_FIELDS):
            sequence = sequences[FIELD_ORDER.index(name)]
            field_coverage[index] += len(wanted & set(sequence)) / len(wanted)
            phrase_support[index] += " " + item.normalized_query + " " in " " + " ".join(sequence) + " "
        contradicts = False
        for attribute, values in item.facets:
            pattern = FACET_PATTERNS.get(attribute)
            actual = set(pattern.findall(" ; ".join(" ".join(sequence) for sequence in sequences))) if pattern is not None else set()
            contradicts |= bool(values and actual and set(values).isdisjoint(actual))
        contradiction += contradicts
    count = max(1, len(positive))
    excluded = [item for item in query.evidence if item.source == "exclusion" and item.tokens]
    exclusion_match = 0.0
    for item in excluded:
        view = evidence_product(features, item)
        union = set().union(*(set(sequence) for sequence in view.affirmed_sequences))
        exclusion_match += set(item.tokens) <= union
    category_tokens = set(terms(category))
    category_coverage = len(category_tokens & set(features.category_tokens)) / max(1, len(category_tokens))
    vector = tuple(value / count for value in field_coverage + phrase_support + [whole, best, cross, missing, contradiction])
    return vector + (exclusion_match / max(1, len(excluded)), category_coverage)


def safe_tier(product: dict) -> tuple:
    flags = ("_semantic_violation", "_exact_constraint_index_match", "_category_leaf_match")
    if not set(flags + ("_hard_constraint_exact_count", "_hard_constraint_count")) <= set(product):
        raise ValueError("Constraint tier metadata is missing")
    if any(name in product and type(product[name]) is not bool for name in flags):
        raise ValueError("Invalid constraint flag")
    exact, count = product.get("_hard_constraint_exact_count", 0), product.get("_hard_constraint_count", 0)
    if type(exact) is not int or type(count) is not int or not 0 <= exact <= count:
        raise ValueError("Invalid hard-constraint counts")
    return (not product.get("_semantic_violation", False),
            product.get("_exact_constraint_index_match", False),
            count > 0 and exact == count, exact, product.get("_category_leaf_match", False))


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate model key")
        result[key] = value
    return result


@dataclass(frozen=True)
class LinearModel:
    weights: tuple[float, ...]
    scales: tuple[float, ...]
    catalog_sha256: str
    config_sha256: str
    model_sha256: str

    def score(self, vector: tuple[float, ...]) -> float:
        if len(vector) != len(self.weights) or any(not math.isfinite(value) or not 0 <= value <= 1 for value in vector):
            raise ValueError("Invalid feature vector")
        return sum(weight * value / scale for weight, value, scale in zip(self.weights, vector, self.scales, strict=True))


class CatalogLinearRanker:
    def __init__(self, catalog_path: str | Path, model_path: str | Path | None = None, *,
                 expected_model_sha256: str | None = None, expected_config_sha256: str | None = None,
                 prefix_limit: int = 30) -> None:
        if type(prefix_limit) is not int or not 1 <= prefix_limit <= 100:
            raise ValueError("prefix_limit must be between 1 and 100")
        self.prefix_limit = prefix_limit
        self.model: LinearModel | None = None
        self.load_error: str | None = None
        self.last_diagnostics: dict = {}
        if model_path is None:
            self.load_error = "no_model"
            return
        try:
            if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                   for value in (expected_model_sha256, expected_config_sha256)):
                raise ValueError("External model and config digest pins are required")
            with Path(model_path).open("rb") as handle:
                raw = handle.read(MAX_MODEL_BYTES + 1)
            if len(raw) > MAX_MODEL_BYTES or hashlib.sha256(raw).hexdigest() != expected_model_sha256:
                raise ValueError("Model size or digest mismatch")
            payload = json.loads(raw, object_pairs_hook=_unique_object)
            if not isinstance(payload, dict) or set(payload) != MODEL_KEYS:
                raise ValueError("Unknown model schema")
            if (payload["feature_names"] != list(FEATURE_NAMES) or payload["scales"] != list(FEATURE_SCALES)
                    or any(type(value) not in (int, float) for value in payload["scales"])):
                raise ValueError("Unknown feature schema or scaling")
            weights = payload["weights"]
            if (not isinstance(weights, list) or len(weights) != len(FEATURE_NAMES)
                    or any(type(value) not in (int, float) or not math.isfinite(value) or abs(value) > MAX_WEIGHT for value in weights)):
                raise ValueError("Invalid model weights")
            catalog_digest = sha256_file(catalog_path)
            if payload["catalog_sha256"] != catalog_digest or payload["config_sha256"] != expected_config_sha256:
                raise ValueError("Catalog or training configuration mismatch")
            self.model = LinearModel(tuple(float(value) for value in weights), FEATURE_SCALES,
                                     catalog_digest, expected_config_sha256, expected_model_sha256)
        except (OSError, ValueError, TypeError, OverflowError, RecursionError) as error:
            self.load_error = type(error).__name__

    @property
    def enabled(self) -> bool:
        return self.model is not None

    def rerank(self, result: SearchResult, state, limit: int = 10) -> SearchResult:
        candidates = result.candidates
        identifiers = [product.get("parent_asin") for product in candidates]
        self.last_diagnostics = {"candidate_context_ids": identifiers, "ranked_context_ids": identifiers,
                                 "model_applied": False, "fallback": self.load_error,
                                 "prefix_limit": self.prefix_limit}
        if self.model is None or not candidates:
            return result
        try:
            if (any(not isinstance(identifier, str) or not identifier for identifier in identifiers)
                    or len(identifiers) != len(set(identifiers))):
                raise ValueError("Candidate IDs must be unique nonempty strings")
            if any(identifier not in identifiers for identifier, _ in result.recommendations):
                raise ValueError("Recommendations do not belong to the candidate context")
            prefix = candidates[:self.prefix_limit]
            if any(type(product.get("_rank_score")) not in (int, float)
                   or not math.isfinite(product["_rank_score"]) for product in candidates):
                raise ValueError("Candidate scores must be finite")
            query = ProductFeatureStore(max_size=1).compile_query(state.evidence, state.user_profile)
            vectors = [feature_vector(product, query, state.category_text) for product in prefix]
            scores = [self.model.score(vector) for vector in vectors]
            groups: dict[tuple, list[int]] = {}
            for index, product in enumerate(prefix):
                groups.setdefault(safe_tier(product), []).append(index)
            ordered = list(candidates)
            for positions in groups.values():
                ranked = sorted(positions, key=lambda index: (-scores[index], index))
                for position, replacement in zip(positions, ranked, strict=True):
                    ordered[position] = candidates[replacement]
            ordered_ids = [product["parent_asin"] for product in ordered]
            if set(ordered_ids) != set(identifiers) or len(ordered_ids) != len(identifiers):
                raise ValueError("Calibration changed candidate membership")
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as error:
            self.last_diagnostics["fallback"] = type(error).__name__
            return result
        self.last_diagnostics.update({"ranked_context_ids": ordered_ids, "model_applied": True,
                                      "ordering_changed": ordered_ids != identifiers,
                                      "model_sha256": self.model.model_sha256})
        return replace(result,
                       recommendations=[(product["parent_asin"], float(product["_rank_score"])) for product in ordered[:limit]],
                       candidates=ordered)

    def close(self) -> None:
        self.model = None
        self.load_error = "closed"
        self.last_diagnostics = {}


class CatalogRankingSearch:
    """Experiment hook; all retrieval, question, and breadth inputs remain local."""

    def __init__(self, inner, ranker: CatalogLinearRanker) -> None:
        self.inner, self.ranker = inner, ranker

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    @property
    def last_diagnostics(self) -> dict:
        return self.ranker.last_diagnostics

    def search_with_context(self, state, limit: int = 10) -> SearchResult:
        return self.ranker.rerank(self.inner.search_with_context(state, limit), state, limit)

    def close(self) -> None:
        try:
            self.inner.close()
        finally:
            self.ranker.close()
