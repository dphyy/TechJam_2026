"""Additional candidate admission without copying or replacing the ranker."""

from __future__ import annotations

from dataclasses import replace
from itertools import zip_longest

from mercury.lexical.dialogue import SessionState
from mercury.lexical.product_features import FIELD_ORDER
from mercury.lexical.retrieval import CatalogSearch, SearchResult, _or_expression
from mercury.lexical.vector_index import catalog_sha256


VIEW_WEIGHTS = {
    "identity": (0.0, 8.0, 6.0, 2.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0),
    "structured": (0.0, 3.0, 8.0, 3.0, 8.0, 3.0, 1.0, 0.0, 0.0, 0.0),
    "descriptive": (0.0, 2.0, 2.0, 8.0, 4.0, 1.0, 6.0, 0.0, 0.0, 0.0),
}
ROW_FIELDS = ("parent_asin", "title", "categories", "features", "details", "store",
              "description", "price", "average_rating", "rating_number")
VIEW_LIMIT = 32
FIELD_LIMIT = 24
BUCKET_LIMIT = 4
BUCKET_ROW_LIMIT = 24
MAX_ADDITIONAL_CANDIDATES = 192
UNION_BUDGET = 4096


class FusionCatalogSearch(CatalogSearch):
    """Reuse one FTS connection, exact index, and bounded feature cache."""

    def __init__(self, *args, additional_admission: bool = True, **kwargs) -> None:
        if type(additional_admission) is not bool:
            raise ValueError("additional_admission must be boolean")
        if kwargs.get("enable_vector_reranker") or kwargs.get("vector_index") is not None:
            raise ValueError("this admission experiment is model-free")
        self.additional_admission = additional_admission
        self.diagnostics: dict = {}
        self._base_routes: list[list[str]] = []
        self._added_ids: list[str] = []
        self._extra_routes: dict[str, list[str]] = {}
        self._extra_queries: dict[str, str] = {}
        self._bucket_values: list[str] = []
        super().__init__(*args, **kwargs)
        self.catalog_digest = catalog_sha256(self.catalog_path)

    def close(self) -> None:
        self.diagnostics.clear()
        self._base_routes.clear()
        self._added_ids.clear()
        self._extra_routes.clear()
        self._extra_queries.clear()
        self._bucket_values.clear()
        super().close()

    def _route(self, expression: str, limit: int) -> list[dict]:
        products = super()._route(expression, limit)
        self._base_routes.append([str(product["parent_asin"]) for product in products])
        return products

    def _exact_constraint_route(self, state: SessionState) -> list[dict]:
        products = super()._exact_constraint_route(state)
        self._base_routes.append([str(product["parent_asin"]) for product in products])
        return products

    def _weighted_route(self, expression: str, weights: tuple[float, ...], limit: int,
                        *, category_bucket: str | None = None) -> list[dict]:
        if not expression:
            return []
        where = "products MATCH ?"
        arguments: list = [expression]
        if category_bucket is not None:
            where += " AND categories = ?"
            arguments.append(category_bucket)
        # Column names and weights are fixed code; all message/catalog text is bound.
        sql_weights = ", ".join(str(weight) for weight in weights)
        rows = self.connection.execute(
            f"SELECT {', '.join(ROW_FIELDS)} FROM products WHERE {where} "
            f"ORDER BY bm25(products, {sql_weights}), rowid LIMIT ?", (*arguments, limit),
        ).fetchall()
        return [self._hydrate(dict(zip(ROW_FIELDS, row))) for row in rows]

    def _additional_routes(self, state: SessionState) -> list[tuple[float, list[dict]]]:
        base_ids = set(identifier for route in self._base_routes for identifier in route)
        budget = min(MAX_ADDITIONAL_CANDIDATES, max(0, UNION_BUDGET - len(base_ids)))
        if not self.additional_admission or not budget:
            return []
        positive = [item.text for item in state.evidence if item.source != "exclusion"]
        expression = _or_expression(positive)
        if not expression:
            return []
        routes: list[list[dict]] = []

        def add(name: str, query: str, weights: tuple[float, ...], limit: int,
                bucket: str | None = None) -> None:
            products = self._weighted_route(query, weights, limit, category_bucket=bucket)
            self._extra_queries[name] = query
            self._extra_routes[name] = [str(product["parent_asin"]) for product in products]
            routes.append(products)

        for name, weights in VIEW_WEIGHTS.items():
            add("view:" + name, expression, weights, VIEW_LIMIT)
        for field in FIELD_ORDER:
            add("field:" + field, f"{field} : ({expression})", VIEW_WEIGHTS["structured"], FIELD_LIMIT)

        category = _or_expression([state.category_text], limit=16)
        if category:
            # Catalog taxonomy paths form buckets; no session or outcome labels.
            self._bucket_values = [str(row[0]) for row in self.connection.execute(
                "SELECT categories, count(*) AS members FROM products WHERE products MATCH ? "
                "GROUP BY categories ORDER BY members DESC, categories LIMIT ?",
                (f"categories : ({category})", BUCKET_LIMIT),
            )]
            for index, bucket in enumerate(self._bucket_values):
                add(f"bucket:{index}", expression, VIEW_WEIGHTS["descriptive"], BUCKET_ROW_LIMIT, bucket)

        # Round-robin prevents one large route from consuming the entire budget.
        # Existing candidates receive no extra votes; new members enter at zero.
        seen = set(base_ids)
        admitted = []
        for row in zip_longest(*routes):
            for product in row:
                if product is None:
                    continue
                identifier = str(product["parent_asin"])
                if identifier in seen:
                    continue
                seen.add(identifier)
                admitted.append(product)
                self._added_ids.append(identifier)
                if len(admitted) == budget:
                    return [(0.0, admitted)]
        return [(0.0, admitted)] if admitted else []

    def search_with_context(self, state: SessionState, limit: int = 10) -> SearchResult:
        self._base_routes = []
        self._added_ids = []
        self._extra_routes = {}
        self._extra_queries = {}
        self._bucket_values = []
        # Only the result slice changes: every admitted candidate uses the
        # inherited ranking body exactly once, retaining complete stage evidence.
        result = super().search_with_context(state, limit=len(self._row_id_by_asin))
        base_ids = list(dict.fromkeys(identifier for route in self._base_routes for identifier in route))
        raw_union = list(dict.fromkeys((*base_ids, *self._added_ids)))
        ranked_ids = [identifier for identifier, _ in result.recommendations]
        if set(raw_union) != set(ranked_ids) or len(ranked_ids) != len(set(ranked_ids)):
            raise RuntimeError("ranking changed candidate membership")
        if len(self._added_ids) > MAX_ADDITIONAL_CANDIDATES \
                or len(raw_union) > max(len(base_ids), UNION_BUDGET):
            raise RuntimeError("additional admission exceeded its candidate budget")
        self.diagnostics = {
            "additional_admission": self.additional_admission, "turn": state.last_turn,
            "catalog_sha256": self.catalog_digest,
            "runtime": {"search_indexes": 1, "neural_requested": False, "neural_loaded": False,
                        "verified_prebuilt_index_loaded": self.using_prebuilt_index},
            "stage_ids": {
                "base_union": base_ids, "additional_admitted": list(self._added_ids),
                "raw_union": raw_union, "raw_ranked": ranked_ids,
                "ranked_top100": [str(product["parent_asin"]) for product in result.candidates],
                "neural_prefix": [], "requested_prefix": ranked_ids[:limit],
            },
            "additional_routes": dict(self._extra_routes), "additional_queries": dict(self._extra_queries),
            "category_buckets": list(self._bucket_values),
            "budgets": {"additional_candidates": MAX_ADDITIONAL_CANDIDATES, "union": UNION_BUDGET,
                        "effective_union_bound": max(len(base_ids), UNION_BUDGET),
                        "inherited_union_overflow": len(base_ids) > UNION_BUDGET},
            "membership_preserved": True,
            "candidate_frequency_recomputed_for_union": bool(self._added_ids),
        }
        return replace(result, recommendations=result.recommendations[:limit])
