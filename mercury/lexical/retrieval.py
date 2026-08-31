from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .constraint_index import (
    ConstraintIndex,
    SQLiteConstraintIndex,
    default_catalog_index_path,
    open_catalog_index,
)
from .dialogue import Evidence, SessionState
from .product_features import (
    BUDGET_RE,
    FACET_PATTERNS,
    FIELD_ORDER,
    FIELD_WEIGHTS,
    CompiledQuery,
    ProductFeatures,
    ProductFeatureStore,
    ProductQuestionFeatures,
    alternative_values,
    component_value,
    evidence_contradiction,
    evidence_product,
    hard_evidence_match,
    resolve_query,
    terms,
)
from .ranking import (
    DEFAULT_RANKING_POLICIES,
    IntentRouter,
    RankingMode,
    RankingPolicies,
    RankingPolicy,
)
from .vector_index import CatalogVectorIndex, VectorIndex


QUALITY_REVIEW_WEIGHT = 1.05
FEATURE_CACHE_SIZE = 5_000
VECTOR_ROUTE_LIMIT = 250
# Conservative optional-vector confidence thresholds.
VECTOR_MIN_SIMILARITY = 0.616618
VECTOR_MIN_MARGIN = 0.011216
# Preserve the old RRF vector route's theoretical maximum: 85 * 0.2 / (60 + 1).
VECTOR_MAX_CONTRIBUTION = 85.0 * 0.2 / 61.0

# Kept as separate constants so route weights can be calibrated without
# changing retrieval structure.
BROAD_OR_ROUTE_WEIGHT = 1.0
PHRASE_ROUTE_WEIGHT = 1.0
CATEGORY_ROUTE_WEIGHT = 1.0
HARD_CONSTRAINT_AND_ROUTE_WEIGHT = 3.0

# FTS5 column order: parent_asin, title, categories, features, details, store,
# description, price, average_rating, rating_number.
BM25_COLUMN_WEIGHTS = (0.0, 4.0, 8.0, 4.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class SearchResult:
    recommendations: list[tuple[str, float]]
    candidates: list[dict]
    prompt_tokens: int = 0
    ranking_mode: RankingMode | None = None
    candidate_ids: tuple[str, ...] | None = None
    vector_stage: dict = field(default_factory=dict)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _or_expression(values: list[str], limit: int = 48) -> str:
    unique = list(dict.fromkeys(token for value in values for token in terms(value)))[:limit]
    return " OR ".join(f'"{token}"' for token in unique)


def _phrase_expression(evidence: list[Evidence], limit: int = 4) -> str:
    tokenized = [
        (item, terms(branch))
        for item in evidence
        if item.source not in {"category", "exclusion"}
        for branch in alternative_values(item.text)
    ]
    chunks = sorted(
        ((item, item_terms) for item, item_terms in tokenized if item_terms),
        key=lambda pair: (len(set(pair[1])), pair[0].weight, pair[0].turn),
        reverse=True,
    )
    phrases: list[str] = []
    for _, item_terms in chunks[:limit]:
        chunk_terms = item_terms[:14]
        if chunk_terms:
            phrases.append('"' + " ".join(chunk_terms) + '"')
    return " OR ".join(phrases)


def _hard_constraint_and_expression(
    category_text: str,
    evidence: list[Evidence],
    limit: int = 6,
) -> str:
    """Require the category and every active non-budget hard phrase."""
    category_terms = terms(category_text)[:14]
    if not category_terms:
        return ""

    hard_phrases: list[str] = []
    seen: set[str] = set()
    for item in evidence:
        if item.source not in {"hard_constraint", "override"}:
            continue
        if BUDGET_RE.search(item.text):
            continue
        branches = [" ".join(terms(branch)[:14]) for branch in alternative_values(item.text)]
        normalized = " OR ".join(f'"{branch}"' for branch in branches if branch)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        hard_phrases.append(f"({normalized})" if len(branches) > 1 else normalized)
        if len(hard_phrases) >= limit:
            break

    if not hard_phrases:
        return ""
    category_phrase = '"' + " ".join(category_terms) + '"'
    return " AND ".join([category_phrase, *hard_phrases])


class CatalogSearch:
    """Multi-route FTS retrieval plus deterministic constraint reranking."""

    def __init__(
        self,
        catalog_path: str | Path,
        feature_cache_size: int = FEATURE_CACHE_SIZE,
        *,
        enable_vector_reranker: bool = False,
        ranking_policies: RankingPolicies = DEFAULT_RANKING_POLICIES,
        intent_router: IntentRouter | None = None,
        vector_index: VectorIndex | None = None,
        catalog_index_path: str | Path | None = None,
        use_prebuilt_index: bool = True,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.feature_store = ProductFeatureStore(max_size=feature_cache_size)
        self.ranking_policies = ranking_policies
        self.intent_router = intent_router or IntentRouter()
        self.catalog_index_path = (
            Path(catalog_index_path)
            if catalog_index_path is not None
            else default_catalog_index_path(self.catalog_path)
        )
        prebuilt_connection = (
            open_catalog_index(self.catalog_path, self.catalog_index_path)
            if use_prebuilt_index
            else None
        )
        self.using_prebuilt_index = prebuilt_connection is not None
        if prebuilt_connection is not None:
            self.connection = prebuilt_connection
            self.constraint_index = SQLiteConstraintIndex(self.connection)
            self._row_id_by_asin = {
                str(parent_asin): int(row_id)
                for parent_asin, row_id in self.connection.execute(
                    "SELECT parent_asin, row_id FROM product_rows"
                )
            }
        else:
            self.connection = sqlite3.connect(":memory:")
            self.constraint_index = ConstraintIndex()
            self._row_id_by_asin: dict[str, int] = {}
            self._build_index()
        self.vector_index = vector_index
        if self.vector_index is None and enable_vector_reranker:
            self.vector_index = CatalogVectorIndex(self.catalog_path)

    def close(self) -> None:
        if self.vector_index is not None:
            self.vector_index.close()
        self.connection.close()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "price UNINDEXED, average_rating UNINDEXED, rating_number UNINDEXED, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, ...]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for row_id, line in enumerate(handle, start=1):
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                self.constraint_index.add_product(product)
                self._row_id_by_asin[parent_asin] = row_id
                fields = {
                    "title": _text(product.get("title")),
                    "categories": _text(product.get("categories")),
                    "features": _text(product.get("features")),
                    "details": _text(product.get("details")),
                    "store": _text(product.get("store")),
                    "description": _text(product.get("description")),
                }
                batch.append((
                    parent_asin,
                    fields["title"],
                    fields["categories"],
                    fields["features"],
                    fields["details"],
                    fields["store"],
                    fields["description"],
                    _text(product.get("price")),
                    _text(product.get("average_rating")),
                    _text(product.get("rating_number")),
                ))
                if len(batch) >= 1000:
                    cursor.executemany(
                        "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch
                    )
                    batch.clear()
        if batch:
            cursor.executemany(
                "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch
            )
        self.connection.commit()

    def _hydrate(self, product: dict) -> dict:
        fields = {
            field: str(product.get(field) or "")
            for field in FIELD_WEIGHTS
        }
        product["_features"] = self.feature_store.get_or_add(
            str(product["parent_asin"]),
            fields,
            price=product.get("price"),
            average_rating=product.get("average_rating"),
            rating_number=product.get("rating_number"),
        )
        return product

    def _route(self, expression: str, limit: int) -> list[dict]:
        if not expression:
            return []
        bm25_weights = ", ".join(str(weight) for weight in BM25_COLUMN_WEIGHTS)
        rows = self.connection.execute(
            "SELECT parent_asin, title, categories, features, details, store, description, "
            "price, average_rating, rating_number "
            "FROM products WHERE products MATCH ? "
            f"ORDER BY bm25(products, {bm25_weights}) "
            "LIMIT ?",
            (expression, limit),
        ).fetchall()
        keys = (
            "parent_asin", "title", "categories", "features", "details", "store",
            "description", "price", "average_rating", "rating_number",
        )
        products: list[dict] = []
        for row in rows:
            product = dict(zip(keys, row))
            products.append(self._hydrate(product))
        return products

    def _exact_constraint_route(self, state: SessionState) -> list[dict]:
        asins = self.constraint_index.exact_intersection(
            state.category_text,
            (
                item.text
                for item in state.evidence
                if item.source not in {"category", "exclusion"}
            ),
        )
        row_ids = sorted(
            self._row_id_by_asin[parent_asin]
            for parent_asin in asins
            if parent_asin in self._row_id_by_asin
        )
        if not row_ids:
            return []
        placeholders = ",".join("?" for _ in row_ids)
        rows = self.connection.execute(
            "SELECT parent_asin, title, categories, features, details, store, "
            "description, price, average_rating, rating_number "
            f"FROM products WHERE rowid IN ({placeholders})",
            row_ids,
        ).fetchall()
        keys = (
            "parent_asin", "title", "categories", "features", "details", "store",
            "description", "price", "average_rating", "rating_number",
        )
        return [self._hydrate(dict(zip(keys, row))) for row in rows]

    def _vector_route(self, rows: list[tuple[int, float]]) -> list[dict]:
        if not rows:
            return []
        row_ids = [row_id for row_id, _ in rows]
        placeholders = ",".join("?" for _ in row_ids)
        fetched = self.connection.execute(
            "SELECT rowid, parent_asin, title, categories, features, details, store, "
            "description, price, average_rating, rating_number "
            f"FROM products WHERE rowid IN ({placeholders})",
            row_ids,
        ).fetchall()
        keys = (
            "parent_asin", "title", "categories", "features", "details", "store",
            "description", "price", "average_rating", "rating_number",
        )
        by_row_id = {
            int(row[0]): dict(zip(keys, row[1:]))
            for row in fetched
        }
        result: list[dict] = []
        for row_id, vector_score in rows:
            product = by_row_id.get(row_id)
            if product is not None:
                product["_vector_score"] = vector_score
                result.append(product)
        return result

    def search(self, state: SessionState, limit: int = 10) -> list[tuple[str, float]]:
        return self.search_with_context(state, limit).recommendations

    def _additional_routes(self, state: SessionState) -> list[tuple[float, list[dict]]]:
        """Optional admission routes; the standard backend adds none."""
        return []

    def search_with_context(self, state: SessionState, limit: int = 10) -> SearchResult:
        if not state.evidence:
            return SearchResult(recommendations=[], candidates=[], candidate_ids=())

        positive_evidence = [
            item for item in state.evidence if item.source != "exclusion"
        ]
        exact_constraint_route = self._exact_constraint_route(state)
        exact_constraint_matches = {
            str(product["parent_asin"]) for product in exact_constraint_route
        }
        routes: list[tuple[float, list[dict]]] = []
        if exact_constraint_route:
            routes.append((0.0, exact_constraint_route))
        routes.append((BROAD_OR_ROUTE_WEIGHT, self._route(
            _or_expression([item.text for item in positive_evidence]), 350
        )))

        latest = state.latest_evidence
        if latest is not None:
            phrase_route = self._route(_phrase_expression(positive_evidence), 180)
            if phrase_route:
                routes.append((PHRASE_ROUTE_WEIGHT, phrase_route))

        if state.category_text:
            category_route = self._route(_or_expression([state.category_text], limit=16), 180)
            if category_route:
                routes.append((CATEGORY_ROUTE_WEIGHT, category_route))

            hard_constraint_route = self._route(
                _hard_constraint_and_expression(state.category_text, state.evidence),
                180,
            )
            if hard_constraint_route:
                routes.append((HARD_CONSTRAINT_AND_ROUTE_WEIGHT, hard_constraint_route))

        routes.extend(self._additional_routes(state))
        rrf: defaultdict[str, float] = defaultdict(float)
        candidates: dict[str, dict] = {}
        for route_weight, route in routes:
            for rank, product in enumerate(route, start=1):
                parent_asin = str(product["parent_asin"])
                rrf[parent_asin] += route_weight / (60.0 + rank)
                candidates.setdefault(parent_asin, product)

        query = self.feature_store.compile_query(state.evidence, state.user_profile)
        routing = self.intent_router.route(state)
        policy = self.ranking_policies.for_mode(routing.mode)
        base_scores: dict[str, float] = {}
        hard_constraint_exactness: dict[str, tuple[int, int]] = {}
        category_leaf_matches: dict[str, bool] = {}
        constraint_sequence_matches: dict[str, bool] = {}
        catalog_tiebreaks: dict[str, tuple[float, float, int]] = {}
        exact_hard_matches: set[str] = set()
        semantic_violations: dict[str, bool] = {}
        token_document_frequency = self._candidate_token_frequency(candidates.values())
        for parent_asin, product in candidates.items():
            features = product["_features"]
            semantic_violations[parent_asin] = self._semantic_violation(features, query)
            needs_facets = policy.contradiction_penalty > 0.0 and any(
                item.source in {"hard_constraint", "override"} and item.facets
                for item in query.evidence
            )
            question_features = (
                self.feature_store.question_features(product)
                if needs_facets
                else None
            )
            score = 85.0 * policy.rrf_scale * rrf[parent_asin]
            score += policy.constraint_scale * self._constraint_score(features, query)
            score += policy.price_scale * self._price_score(features, query)
            score += policy.quality_scale * self._quality_tiebreak(features)
            score += self._constraint_fit_adjustment(
                features, question_features, query, policy
            )
            score += self._budget_violation_adjustment(features, query, policy)
            profile_bonus = self._profile_bonus(features, state, policy)
            score += profile_bonus
            product["_profile_bonus"] = profile_bonus
            rating_alignment = self._profile_rating_alignment(
                features, state.user_profile
            )
            score += policy.profile_rating_scale * rating_alignment
            product["_profile_rating_alignment"] = rating_alignment
            base_scores[parent_asin] = score
            exact_count, hard_constraint_count = self._hard_constraint_exactness(
                features, state.evidence
            )
            hard_constraint_exactness[parent_asin] = (
                exact_count,
                hard_constraint_count,
            )
            category_leaf_matches[parent_asin] = self._category_leaf_match(
                features, state.category_text
            )
            constraint_sequence_matches[parent_asin] = (
                self._constraint_sequence_match(features, query)
            )
            catalog_tiebreaks[parent_asin] = self._catalog_tiebreak(
                features,
                query,
                state.category_text,
                token_document_frequency,
                len(candidates),
            )
            if hard_constraint_count > 0 and exact_count == hard_constraint_count:
                exact_hard_matches.add(parent_asin)

        # Dense retrieval never admits candidates. It can only adjust lexical
        # candidates after query, category, absolute-similarity, and margin gates.
        vector_prompt_tokens = 0
        vector_scores: dict[str, float] = {}
        vector_confident = False
        structured_query = state.semantic_query()
        vector_stage = {"attempted": False, "returned_count": 0, "contribution_count": 0,
                        "confidence_gate": False, "status": "not_eligible"}
        if (
            policy.vector_scale > 0.0
            and self.vector_index is not None
            and candidates
            and state.category_text
            and structured_query
        ):
            vector_result = self.vector_index.search(structured_query, VECTOR_ROUTE_LIMIT)
            try:
                receipt = getattr(self.vector_index, "last_call_status", {})
            except Exception:
                receipt = {}
            receipt = receipt if isinstance(receipt, dict) else {}
            vector_stage.update(attempted=True, returned_count=len(vector_result.rows))
            status = receipt.get("status")
            known_statuses = {"not_called", "not_eligible", "cache_hit", "client_unavailable",
                              "inference_succeeded", "inference_failed", "backend_unavailable",
                              "empty_query", "invalid_similarity", "empty_result"}
            vector_stage["status"] = (status if isinstance(status, str) and status in known_statuses else
                                      "returned_results" if vector_result.rows else "empty_result_unknown")
            inference_attempted = receipt.get("inference_attempted")
            vector_stage["inference_attempted"] = inference_attempted if type(inference_attempted) is bool else None
            error_type = receipt.get("error_type")
            if isinstance(error_type, str) and len(error_type) <= 80 and error_type.isascii() and error_type.isidentifier():
                vector_stage["error_type"] = error_type
            vector_prompt_tokens = vector_result.prompt_tokens
            category_vector_route = [
                product
                for product in self._vector_route(vector_result.rows)
                if self._category_match(product, state.category_text)
            ]
            if category_vector_route:
                top_score = float(category_vector_route[0]["_vector_score"])
                runner_score = (
                    float(category_vector_route[1]["_vector_score"])
                    if len(category_vector_route) > 1
                    else VECTOR_MIN_SIMILARITY
                )
                top_id = str(category_vector_route[0]["parent_asin"])
                vector_confident = (
                    top_id in candidates
                    and top_score >= VECTOR_MIN_SIMILARITY
                    and top_score - runner_score >= VECTOR_MIN_MARGIN
                )
                vector_scores = {
                    str(product["parent_asin"]): float(product["_vector_score"])
                    for product in category_vector_route
                    if str(product["parent_asin"]) in candidates
                }

        ranked: list[tuple[str, float]] = []
        best_lexical_score = max(base_scores.values(), default=0.0)
        has_exact_hard_match = bool(exact_hard_matches)
        for parent_asin, base_score in base_scores.items():
            similarity = vector_scores.get(parent_asin, 0.0)
            contribution = self._bounded_vector_contribution(
                similarity=similarity,
                base_score=base_score,
                best_lexical_score=best_lexical_score,
                vector_confident=vector_confident,
                has_exact_hard_match=has_exact_hard_match,
                is_exact_hard_match=parent_asin in exact_hard_matches,
            ) * policy.vector_scale
            candidates[parent_asin]["_vector_score"] = similarity
            candidates[parent_asin]["_vector_contribution"] = contribution
            candidates[parent_asin]["_ranking_mode"] = routing.mode.value
            ranked.append((parent_asin, base_score + contribution))
            vector_stage["contribution_count"] += int(contribution != 0.0)
        vector_stage["confidence_gate"] = vector_confident
        # Exact hard-constraint coverage defines explicit ranking tiers. The
        # requested leaf category and a cohesive sequence of disclosed details
        # then break otherwise ambiguous ties before the calibrated score.
        ranked.sort(key=lambda item: (
            int(semantic_violations[item[0]]),
            -int(item[0] in exact_constraint_matches),
            -int(
                hard_constraint_exactness[item[0]][1] > 0
                and hard_constraint_exactness[item[0]][0]
                == hard_constraint_exactness[item[0]][1]
            ),
            -hard_constraint_exactness[item[0]][0],
            -int(category_leaf_matches[item[0]]),
            -int(constraint_sequence_matches[item[0]]),
            tuple(-value for value in catalog_tiebreaks[item[0]]),
            -item[1],
            item[0],
        ))
        context: list[dict] = []
        for parent_asin, score in ranked[:100]:
            product = dict(candidates[parent_asin])
            product["_rank_score"] = score
            product["_semantic_violation"] = semantic_violations[parent_asin]
            exact_count, hard_count = hard_constraint_exactness[parent_asin]
            product["_hard_constraint_exact_count"] = exact_count
            product["_hard_constraint_count"] = hard_count
            product["_category_leaf_match"] = category_leaf_matches[parent_asin]
            product["_constraint_sequence_match"] = (
                constraint_sequence_matches[parent_asin]
            )
            product["_catalog_tiebreak"] = catalog_tiebreaks[parent_asin]
            product["_exact_constraint_index_match"] = (
                parent_asin in exact_constraint_matches
            )
            context.append(product)
        return SearchResult(
            recommendations=ranked[:limit],
            candidates=context,
            prompt_tokens=vector_prompt_tokens,
            ranking_mode=routing.mode,
            candidate_ids=tuple(candidates),
            vector_stage=vector_stage,
        )

    @staticmethod
    def _constraint_fit_adjustment(
        product: ProductFeatures,
        product_facets: ProductQuestionFeatures | None,
        query: CompiledQuery,
        policy: RankingPolicy,
    ) -> float:
        score = 0.0
        hard_sources = {"hard_constraint", "override"}
        query = resolve_query(product, query)
        for item in query.evidence:
            if not item.tokens or item.source == "category" or item.is_budget:
                continue
            view = evidence_product(product, item)
            if item.scope and not view.token_weights:
                continue
            if item.source == "exclusion" and not CatalogSearch._exclusion_match(view, item):
                continue
            matched = sum(token in view.token_weights for token in item.tokens)
            coverage = matched / len(item.tokens)
            exact = (
                len(item.tokens) >= 2
                and item.normalized_query in view.normalized_text
            )
            if item.source == "exclusion":
                score -= item.weight * (
                    policy.soft_coverage_bonus * coverage
                    + policy.soft_exact_bonus * float(exact)
                )
            elif item.source in hard_sources:
                score += item.weight * (
                    policy.hard_coverage_bonus * coverage
                    - policy.hard_missing_penalty * (1.0 - coverage)
                    + policy.hard_exact_bonus * float(exact)
                )
                for attribute, expected_values in item.facets:
                    actual_values = set(FACET_PATTERNS[attribute].findall(view.normalized_text)
                                        if item.scope else product_facets.facet_values(attribute)
                                        if product_facets is not None else ())
                    if (
                        expected_values
                        and actual_values
                        and actual_values.isdisjoint(expected_values)
                    ):
                        score -= item.weight * policy.contradiction_penalty
            else:
                score += item.weight * (
                    policy.soft_coverage_bonus * coverage
                    + policy.soft_exact_bonus * float(exact)
                )
        return score

    @staticmethod
    def _budget_violation_adjustment(
        product: ProductFeatures,
        query: CompiledQuery,
        policy: RankingPolicy,
    ) -> float:
        if product.price is None or policy.budget_violation_penalty <= 0.0:
            return 0.0
        score = 0.0
        for budget in query.budgets:
            relative_error = abs(product.price - budget.amount) / max(
                budget.amount, 10.0
            )
            if budget.mode in {"under", "below", "maximum", "max"}:
                violation = (2.0 if budget.amount <= 0.0 else
                             max(0.0, (product.price - budget.amount) / budget.amount))
            else:
                violation = max(0.0, relative_error - 0.35)
            score -= (
                budget.weight
                * policy.budget_violation_penalty
                * min(violation, 2.0)
            )
        return score

    @staticmethod
    def _category_match(product: dict, requested_category: str) -> bool:
        requested = set(terms(requested_category))
        product_categories = set(terms(str(product.get("categories") or "")))
        return bool(requested) and requested.issubset(product_categories)

    @staticmethod
    def _bounded_vector_contribution(
        *,
        similarity: float,
        base_score: float,
        best_lexical_score: float,
        vector_confident: bool,
        has_exact_hard_match: bool,
        is_exact_hard_match: bool,
    ) -> float:
        if (
            not vector_confident
            or similarity < VECTOR_MIN_SIMILARITY
            or best_lexical_score - base_score > VECTOR_MAX_CONTRIBUTION
            or (has_exact_hard_match and not is_exact_hard_match)
        ):
            return 0.0
        return min(
            VECTOR_MAX_CONTRIBUTION,
            max(0.0, similarity) * VECTOR_MAX_CONTRIBUTION,
        )

    @staticmethod
    def _exact_hard_constraint_match(
        product: ProductFeatures | dict, evidence: list[Evidence]
    ) -> bool:
        exact_count, hard_constraint_count = CatalogSearch._hard_constraint_exactness(
            product, evidence
        )
        return hard_constraint_count > 0 and exact_count == hard_constraint_count

    @staticmethod
    def _hard_constraint_exactness(
        product: ProductFeatures | dict, evidence: list[Evidence]
    ) -> tuple[int, int]:
        hard_constraints = [
            item
            for item in evidence
            if item.source in {"hard_constraint", "override"}
            and not BUDGET_RE.search(item.text)
            and terms(item.text)
        ]
        if not hard_constraints:
            return 0, 0
        if not isinstance(product, ProductFeatures):
            product = ProductFeatureStore(max_size=1).add("candidate", {
                field: _text(product.get(field)) for field in FIELD_WEIGHTS
            })
        exact_count = sum(hard_evidence_match(product, item.text) for item in hard_constraints)
        return exact_count, len(hard_constraints)

    @staticmethod
    def _category_leaf_match(
        product: ProductFeatures, requested_category: str
    ) -> bool:
        """Prefer an exact catalog leaf over a deeper category containing it."""
        generic_taxonomy_tokens = {"clothing", "shoes", "jewelry"}
        requested = tuple(
            token for token in terms(requested_category)
            if token not in generic_taxonomy_tokens
        )
        category = tuple(
            token for token in product.category_tokens
            if token not in generic_taxonomy_tokens
        )
        return bool(requested) and category[-len(requested):] == requested

    @staticmethod
    def _constraint_sequence_match(
        product: ProductFeatures, query: CompiledQuery
    ) -> bool:
        """Detect a cohesive catalog block spanning multiple disclosed details."""
        query = resolve_query(product, query)
        for item in query.evidence:
            if item.scope and item.source != "exclusion":
                view = evidence_product(product, item)
                if not all(token in view.token_weights for token in item.tokens):
                    return False
        chunks: list[tuple[str, ...]] = []
        for item in query.evidence:
            if (
                item.source in {"category", "exclusion"}
                or item.is_budget
                or not item.tokens
            ):
                continue
            chunk = tuple(item.tokens)
            if chunk in chunks:
                continue
            # A shorter detail such as ``polyester`` adds no ordering signal
            # when a more specific active detail already contains it.
            if any(
                len(chunk) <= len(other)
                and any(
                    other[index:index + len(chunk)] == chunk
                    for index in range(len(other) - len(chunk) + 1)
                )
                for other in (
                    tuple(candidate.tokens)
                    for candidate in query.evidence
                    if candidate is not item and candidate.tokens
                )
            ):
                continue
            chunks.append(chunk)

        if len(chunks) < 2:
            return False
        sequence = " ".join(token for chunk in chunks for token in chunk)
        return len(sequence.split()) >= 3 and sequence in product.normalized_text

    @staticmethod
    def _candidate_token_frequency(products: object) -> dict[str, int]:
        """Count candidate documents, not occurrences, for rarity weighting."""
        frequency: defaultdict[str, int] = defaultdict(int)
        for product in products:
            features = product.get("_features") if isinstance(product, dict) else None
            if not isinstance(features, ProductFeatures):
                continue
            for token in features.token_weights:
                frequency[token] += 1
        return dict(frequency)

    @staticmethod
    def _catalog_tiebreak(
        product: ProductFeatures,
        query: CompiledQuery,
        requested_category: str,
        token_document_frequency: dict[str, int],
        candidate_count: int,
    ) -> tuple[float, float, int]:
        """Return label-free field proximity, coherence, and taxonomy signals.

        These intentionally sit below hard-constraint tiers. They resolve
        catalog ambiguity before popularity can reward a merely well-reviewed
        sibling whose matching words are scattered across unrelated fields.
        """
        query = resolve_query(product, query)
        active_chunks: list[tuple[str, ...]] = []
        chunk_products: dict[tuple[str, ...], ProductFeatures] = {}
        for item in query.evidence:
            if item.source in {"category", "exclusion"} or item.is_budget:
                continue
            chunk = tuple(item.tokens)
            if not chunk or chunk in active_chunks:
                continue
            if any(
                len(chunk) < len(other)
                and any(
                    other[index:index + len(chunk)] == chunk
                    for index in range(len(other) - len(chunk) + 1)
                )
                for other in (
                    tuple(candidate.tokens)
                    for candidate in query.evidence
                    if candidate is not item and candidate.tokens
                )
            ):
                continue
            active_chunks.append(chunk)
            chunk_products[chunk] = evidence_product(product, item)

        field_score = 0.0
        feature_matches = 0
        feature_sequence = product.field_sequences[FIELD_ORDER.index("features")]
        for chunk in active_chunks:
            view = chunk_products[chunk]
            rarity = sum(
                math.log1p(candidate_count / max(token_document_frequency.get(token, 1), 1))
                for token in set(chunk)
            ) / len(set(chunk))
            best = 0.0
            for sequence in view.field_sequences:
                positions = [
                    index for index in range(len(sequence) - len(chunk) + 1)
                    if sequence[index:index + len(chunk)] == chunk
                ]
                if positions:
                    best = max(best, rarity * (1.0 + min(len(chunk), 8) / 8.0))
            field_score += best
            feature_sequence = view.field_sequences[FIELD_ORDER.index("features")]
            if any(
                feature_sequence[index:index + len(chunk)] == chunk
                for index in range(len(feature_sequence) - len(chunk) + 1)
            ):
                feature_matches += 1

        # Multiple independently disclosed chunks in the feature field are a
        # stronger catalog-record match than the same words scattered across
        # title, taxonomy, description, or details.
        coherence = (
            feature_matches / len(active_chunks)
            if active_chunks
            else 0.0
        )
        requested = tuple(terms(requested_category))
        category = product.category_tokens
        leaf_specificity = 0
        if requested:
            for width in range(1, len(requested) + 1):
                if category[-width:] == requested[-width:]:
                    leaf_specificity = width
                else:
                    break
        return round(field_score, 9), round(coherence, 9), leaf_specificity

    @staticmethod
    def _constraint_score(product: ProductFeatures, query: CompiledQuery) -> float:
        score = 0.0
        query = resolve_query(product, query)
        for item in query.evidence:
            if not item.tokens:
                continue
            view = evidence_product(product, item)
            if item.scope and item.source != "exclusion":
                value_tokens = terms(component_value(item.normalized_query))
                if not value_tokens or not any(token in view.token_weights for token in value_tokens):
                    continue
            if item.source == "exclusion" and not CatalogSearch._exclusion_match(view, item):
                continue
            matched_weight = 0.0
            matched_terms = 0
            for token in item.tokens:
                best_field_weight = view.token_weights.get(token, 0.0)
                matched_weight += best_field_weight
                matched_terms += int(best_field_weight > 0.0)
            coverage = matched_terms / len(item.tokens)
            field_affinity = matched_weight / (
                len(item.tokens) * max(FIELD_WEIGHTS.values())
            )
            item_score = item.weight * (1.9 * coverage + 0.4 * field_affinity)

            if len(item.tokens) >= 2 and item.normalized_query in view.normalized_text:
                specificity = min(2.0, 0.55 + 0.22 * len(item.tokens))
                item_score += item.weight * specificity
            if coverage >= 0.999:
                item_score += item.weight * 0.45
            score += -item_score if item.source == "exclusion" else item_score
        if query.preference_tokens:
            matches = sum(
                token in product.token_weights
                for token in query.preference_tokens
            )
            score += 0.45 * matches / len(query.preference_tokens)
        return score

    @staticmethod
    def _exclusion_match(product: ProductFeatures, item) -> bool:
        return bool(item.tokens) and any(
            sequence[start:start + len(item.tokens)] == item.tokens
            for sequence in product.affirmed_sequences
            for start in range(len(sequence) - len(item.tokens) + 1)
        )

    @staticmethod
    def _semantic_violation(product: ProductFeatures, query: CompiledQuery) -> bool:
        """Explicit contradictions outrank lexical tiers; unknown fields stay neutral."""
        query = resolve_query(product, query)
        for item in query.evidence:
            view = evidence_product(product, item)
            if item.source == "exclusion" and CatalogSearch._exclusion_match(view, item):
                return True
            if item.source in {"hard_constraint", "override"} and evidence_contradiction(product, item):
                return True
        return False

    @staticmethod
    def _profile_bonus(
        product: ProductFeatures, state: SessionState, policy: RankingPolicy
    ) -> float:
        """A bounded tie-break from locally learned, non-conflicting preferences."""
        profile = state.long_term_profile
        if profile is None or not profile.learned:
            return 0.0
        current_attributes = {
            item.attribute for item in state.evidence if item.attribute is not None
        }
        exclusions = [item.text.casefold() for item in state.evidence if item.source == "exclusion"]
        cap = 0.4 if policy is not DEFAULT_RANKING_POLICIES.browsing else 1.0
        total = 0.0
        affected_facets = set(state.no_preference_attributes)
        affected_facets.update(
            name for item in state.evidence
            if item.operation.value in {"replace", "exclude"} and item.source != "category"
            for name, pattern in FACET_PATTERNS.items() if pattern.search(item.text)
        )
        for preference in profile.learned.values():
            if preference.attribute in affected_facets or any(
                FACET_PATTERNS[name].search(preference.value)
                for name in affected_facets if name in FACET_PATTERNS
            ):
                continue
            if preference.attribute in current_attributes and preference.attribute != "other":
                continue
            value_tokens = tuple(terms(preference.value))
            if not value_tokens or any(preference.value.casefold() in value for value in exclusions):
                continue
            coverage = sum(token in product.token_weights for token in value_tokens) / len(value_tokens)
            total += 0.35 * preference.confidence * coverage
        return min(cap, total)

    @staticmethod
    def _profile_rating_alignment(
        product: ProductFeatures, user_profile: dict,
    ) -> float:
        """Return bounded affinity to the customer's historical rating pattern."""
        try:
            prior_rating = float(user_profile.get("average_prior_rating"))
        except (AttributeError, TypeError, ValueError):
            return 0.0
        if not 1.0 <= prior_rating <= 5.0 or not 0.0 < product.average_rating <= 5.0:
            return 0.0
        return 1.0 - abs(product.average_rating - prior_rating) / 4.0

    @staticmethod
    def _price_score(product: ProductFeatures, query: CompiledQuery) -> float:
        if product.price is None:
            return 0.0

        score = 0.0
        for budget in query.budgets:
            if budget.mode in {"under", "below", "maximum", "max"}:
                closeness = (
                    1.0
                    if product.price <= budget.amount
                    else max(
                        0.0,
                        0.0 if budget.amount <= 0.0 else
                        1.0 - (product.price - budget.amount) / budget.amount,
                    )
                )
            else:
                closeness = max(
                    0.0,
                    1.0
                    - abs(product.price - budget.amount) / max(budget.amount, 10.0),
                )
            score += budget.weight * 1.4 * closeness
        return score

    @staticmethod
    def _quality_tiebreak(product: ProductFeatures) -> float:
        return (
            min(max(product.average_rating, 0.0), 5.0) * 0.02
            + math.log1p(product.rating_number) * QUALITY_REVIEW_WEIGHT
        )
