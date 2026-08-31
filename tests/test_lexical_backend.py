from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mercury.lexical.preprocessing import build_catalog_index
from mercury.lexical.agent import Agent
from mercury.lexical.config import (
    AgentConfig,
    FULL_BREADTH_POLICY,
    FULL_WIDTH_CONFIG,
    RecommendationPolicy,
)
from mercury.lexical.constraint_index import ConstraintIndex, normalize_constraint
from mercury.lexical.dialogue import Evidence, PreferenceOperation, SessionState
from mercury.lexical.product_features import FIELD_WEIGHTS, ProductFeatureStore, terms
from mercury.lexical.question_planner import AdaptiveQuestionPlanner
from mercury.lexical.ranking import (
    DEFAULT_RANKING_POLICIES,
    LEGACY_POLICY,
    IntentRouter,
    RankingMode,
)
from mercury.lexical.retrieval import (
    CatalogSearch,
    QUALITY_REVIEW_WEIGHT,
    _hard_constraint_and_expression,
    _or_expression,
    _phrase_expression,
)
from mercury.lexical.vector_index import CatalogVectorIndex, catalog_sha256


def _legacy_constraint_score(
    product: dict, evidence: list[Evidence], user_profile: dict | None = None
) -> float:
    field_tokens = {
        field: set(terms(str(product.get(field) or "")))
        for field in FIELD_WEIGHTS
    }
    normalized_fields = {
        field: " ".join(terms(str(product.get(field) or "")))
        for field in FIELD_WEIGHTS
    }
    score = 0.0
    for item in evidence:
        query_terms = list(dict.fromkeys(terms(item.text)))
        if not query_terms:
            continue
        matched_weight = 0.0
        matched_terms = 0
        for token in query_terms:
            best_field_weight = max(
                (
                    weight
                    for field, weight in FIELD_WEIGHTS.items()
                    if token in field_tokens[field]
                ),
                default=0.0,
            )
            matched_weight += best_field_weight
            matched_terms += int(best_field_weight > 0.0)
        coverage = matched_terms / len(query_terms)
        field_affinity = matched_weight / (
            len(query_terms) * max(FIELD_WEIGHTS.values())
        )
        score += item.weight * (1.9 * coverage + 0.4 * field_affinity)
        normalized_query = " ".join(query_terms)
        if len(query_terms) >= 2 and any(
            normalized_query in value for value in normalized_fields.values()
        ):
            score += item.weight * min(2.0, 0.55 + 0.22 * len(query_terms))
        if coverage >= 0.999:
            score += item.weight * 0.45
    tags = user_profile.get("preference_tags") if isinstance(user_profile, dict) else None
    if isinstance(tags, list) and tags:
        preference_terms = {token for tag in tags for token in terms(str(tag))}
        product_terms = set().union(*field_tokens.values())
        if preference_terms:
            score += 0.45 * len(preference_terms & product_terms) / len(preference_terms)
    return score


class DialogueStateTest(unittest.TestCase):
    def test_free_form_answer_preserves_specific_evidence(self) -> None:
        state = SessionState(user_profile={})
        state.observe("I'm looking for Shirts, but I'm still exploring.", 1)
        state.observe("Breathable cotton would be ideal for warm weather", 2)
        self.assertIn(
            "breathable cotton would be ideal for warm weather",
            [item.text.lower() for item in state.evidence],
        )

    def test_accumulates_constraints_and_removes_opening_preference_on_override(self) -> None:
        state = SessionState(user_profile={})
        state.observe("I'm looking for Shoes. I prefer red.", 1)
        state.observe("For that, what matters is: leather; wide width.", 2)
        state.observe("Actually, ignore my earlier preference. What I need is: black.", 3)
        evidence = [item.text.lower() for item in state.evidence]
        self.assertIn("shoes", evidence)
        self.assertIn("leather", evidence)
        self.assertIn("wide width", evidence)
        self.assertIn("black", evidence)
        self.assertNotIn("i prefer red", evidence)

    def test_override_replaces_only_the_active_values_for_its_attribute(self) -> None:
        state = SessionState(user_profile={})
        state.observe("I'm looking for Shoes. I prefer red.", 1)
        state.record_question("material")
        state.observe("For that, what matters is: leather.", 2)
        state.record_question("color")
        state.observe("For that, what matters is: blue.", 3)

        state.observe("Actually, what I need is: black.", 4)

        active = {(item.attribute, item.text.lower()) for item in state.evidence}
        self.assertIn(("material", "leather"), active)
        self.assertIn(("color", "black"), active)
        self.assertNotIn(("color", "i prefer red"), active)
        self.assertNotIn(("color", "blue"), active)
        self.assertEqual(state.evidence[-1].operation, PreferenceOperation.REPLACE)

    def test_append_keeps_multiple_values_in_one_attribute_set(self) -> None:
        state = SessionState(user_profile={})
        state.observe("I'm looking for Shoes, but I'm still exploring.", 1)
        state.record_question("material")
        state.observe("For that, what matters is: leather; suede.", 2)

        material = [
            item.text.lower() for item in state.evidence
            if item.attribute == "material"
        ]
        self.assertEqual(material, ["leather", "suede"])
        self.assertTrue(all(
            item.operation == PreferenceOperation.APPEND
            for item in state.evidence if item.attribute == "material"
        ))

    def test_no_preference_is_not_positive_search_evidence(self) -> None:
        state = SessionState(user_profile={})
        state.observe("I'm looking for Jackets, but I'm still exploring.", 1)
        state.record_question("other")
        state.observe("I don't have a preference for other; please use your judgment.", 2)
        self.assertEqual([item.text.lower() for item in state.evidence], ["jackets"])

    def test_exclusion_supersedes_conflicting_positive_evidence(self) -> None:
        state = SessionState(user_profile={})
        state.observe("I'm looking for Shoes. I prefer leather.", 1)
        state.record_question("material")
        state.observe("I don't want leather.", 2)

        self.assertNotIn("i prefer leather", [item.text.lower() for item in state.evidence])
        self.assertEqual(state.evidence[-1].source, "exclusion")

    def test_exclusion_is_absent_from_positive_query_routes(self) -> None:
        evidence = [
            Evidence("Shoes", 1.4, "category", 1),
            Evidence("leather", 3.8, "exclusion", 2, "material"),
        ]
        state = SessionState(user_profile={}, evidence=evidence, category_text="Shoes")

        self.assertNotIn("leather", _or_expression([
            item.text for item in evidence if item.source != "exclusion"
        ]))
        self.assertNotIn("leather", _phrase_expression(evidence))
        self.assertNotIn("leather", state.semantic_query() or "")


class IntentRouterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.router = IntentRouter()

    def test_open_ended_session_stays_in_browsing_mode_as_preferences_accumulate(
        self,
    ) -> None:
        state = SessionState(user_profile={})
        state.observe("I'm looking for Shirts, but I'm still exploring.", 1)
        state.record_question("material")
        state.observe("For that, what matters is: breathable cotton.", 2)

        decision = self.router.route(state)

        self.assertEqual(decision.mode, RankingMode.BROWSING)
        self.assertEqual(decision.reasons, ("open_ended_start",))

    def test_explicit_requirement_routes_to_buying_without_scenario_labels(self) -> None:
        state = SessionState(user_profile={})
        state.observe(
            "I'm looking for Shoes. A key requirement is: waterproof wide width.",
            1,
        )

        decision = self.router.route(state)

        self.assertEqual(decision.mode, RankingMode.BUYING)
        self.assertIn("explicit_requirement", decision.reasons)

    def test_open_session_can_transition_to_buying_after_explicit_override(self) -> None:
        state = SessionState(user_profile={})
        state.observe("I'm looking for Shoes, but I'm still exploring.", 1)
        self.assertEqual(self.router.route(state).mode, RankingMode.BROWSING)

        state.observe("Actually, what I need is: black leather.", 2)

        self.assertEqual(self.router.route(state).mode, RankingMode.BUYING)


class AdaptiveQuestionPlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.feature_store = ProductFeatureStore()
        self.planner = AdaptiveQuestionPlanner(self.feature_store)

    def _candidate(self, title: str, score: float) -> dict:
        candidate = {
            "parent_asin": title,
            "title": title,
            "categories": "Clothing Shirts",
            "features": "",
            "details": "",
            "store": "Example",
            "description": "",
            "price": "50",
            "_rank_score": score,
        }
        candidate["_features"] = self.feature_store.add(
            title,
            {field: str(candidate.get(field) or "") for field in FIELD_WEIGHTS},
            price=candidate["price"],
        )
        return candidate

    def test_question_attribute_changes_with_candidate_differences(self) -> None:
        material_state = SessionState(user_profile={})
        material_state.asked_attributes.extend(["other", "other"])
        material_candidates = [
            self._candidate("cotton shirt", 20.0),
            self._candidate("leather shirt", 14.0),
            self._candidate("polyester shirt", 10.0),
        ]
        material_plan = self.planner.choose(
            material_state, material_candidates, 1
        )

        color_state = SessionState(user_profile={})
        color_state.asked_attributes.extend(["other", "other"])
        color_candidates = [
            self._candidate("red shirt", 20.0),
            self._candidate("blue shirt", 14.0),
            self._candidate("green shirt", 10.0),
        ]
        color_plan = self.planner.choose(
            color_state, color_candidates, 1
        )

        self.assertEqual(material_plan.attribute, "material")
        self.assertEqual(color_plan.attribute, "color")
        self.assertIn("cotton", material_plan.message)
        self.assertIn("red", color_plan.message)
        self.assertNotEqual(material_plan.message, color_plan.message)
        self.assertGreater(material_plan.information_gain, 0.0)
        self.assertGreater(material_plan.answerability, 0.0)
        self.assertAlmostEqual(
            material_plan.expected_value,
            material_plan.information_gain * material_plan.answerability,
        )

    def test_early_question_prioritizes_must_have_without_repeating_boundary(self) -> None:
        state = SessionState(user_profile={})
        candidates = [
            self._candidate("cotton shirt", 20.0),
            self._candidate("leather shirt", 14.0),
        ]

        plan = self.planner.choose(state, candidates, 1)
        self.assertEqual(plan.attribute, "other")

        state.no_preference_attributes.add("other")
        next_plan = self.planner.choose(state, candidates, 2)
        self.assertNotEqual(next_plan.attribute, "other")

    def test_information_gain_prioritizes_high_rank_candidate_differences(self) -> None:
        planner = AdaptiveQuestionPlanner(self.feature_store, score_temperature=1.0)
        scores = [10.0 - index for index in range(5)] + [0.0] * 95
        probabilities = planner._score_probabilities(scores)
        material = [
            (f"material-{index}",) if index < 5 else ("other",)
            for index in range(100)
        ]
        color = [
            ("same",) if index < 49 else (f"color-{index}",)
            for index in range(100)
        ]

        uniform_material = planner._information_gain("material", material)
        uniform_color = planner._information_gain("color", color)
        weighted_material = planner._information_gain(
            "material", material, probabilities
        )
        weighted_color = planner._information_gain("color", color, probabilities)

        self.assertGreater(
            uniform_color.information_gain, uniform_material.information_gain
        )
        self.assertGreater(
            weighted_material.information_gain, weighted_color.information_gain
        )

    def test_score_softmax_is_stable_and_temperature_is_validated(self) -> None:
        probabilities = self.planner._score_probabilities([10_000.0, 9_999.0])

        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertGreater(probabilities[0], probabilities[1])
        with self.assertRaises(ValueError):
            AdaptiveQuestionPlanner(self.feature_store, score_temperature=0.0)


class AgentRetrievalTest(unittest.TestCase):
    @staticmethod
    def _catalog_rows() -> list[dict]:
        return [
            {
                "parent_asin": "A", "title": "Everyday Boot", "categories": ["Shoes"],
                "features": ["synthetic", "standard width"], "details": {},
                "store": "Example", "description": [], "price": 40,
                "average_rating": 4.8, "rating_number": 500,
            },
            {
                "parent_asin": "B", "title": "Trail Boot", "categories": ["Shoes"],
                "features": ["full grain leather", "wide width"], "details": {},
                "store": "Example", "description": [], "price": 60,
                "average_rating": 4.0, "rating_number": 10,
            },
        ]

    @classmethod
    def _write_catalog(cls, directory: str) -> Path:
        catalog = Path(directory) / "catalog.jsonl"
        catalog.write_text(
            "".join(json.dumps(row) + "\n" for row in cls._catalog_rows()),
            encoding="utf-8",
        )
        return catalog

    @staticmethod
    def _features(
        store: ProductFeatureStore,
        parent_asin: str,
        *,
        title: str = "",
        categories: str = "",
        features: str = "",
        details: str = "",
        price: float = 50.0,
        average_rating: float = 4.0,
        rating_number: int = 1,
    ):
        fields = {field: "" for field in FIELD_WEIGHTS}
        fields.update({
            "title": title,
            "categories": categories,
            "features": features,
            "details": details,
        })
        return store.add(
            parent_asin,
            fields,
            price=price,
            average_rating=average_rating,
            rating_number=rating_number,
        )

    def test_precomputed_features_are_reused_and_read_only(self) -> None:
        store = ProductFeatureStore()
        product = self._features(
            store,
            "A",
            title="Cotton running shirt",
            features="breathable lightweight fabric",
        )
        self.assertIs(store.get("A"), product)
        self.assertEqual(len(store), 1)
        with self.assertRaises(TypeError):
            product.token_weights[0] = 99.0  # type: ignore[index]
        with self.assertRaises(ValueError):
            self._features(store, "A", title="duplicate")

    def test_hard_constraint_route_requires_category_and_exact_phrases(self) -> None:
        evidence = [
            Evidence("Shirts", 1.4, "category", 1),
            Evidence("sleeveless design", 3.8, "hard_constraint", 2),
            Evidence("100% polyester", 6.0, "override", 3),
            Evidence("under $50", 3.8, "hard_constraint", 3),
            Evidence("lightweight", 2.5, "clarification", 2),
        ]

        self.assertEqual(
            _hard_constraint_and_expression("Shirts", evidence),
            '"shirts" AND "sleeveless design" AND "100 polyester"',
        )

    def test_catalog_constraint_index_intersects_category_and_exact_values(self) -> None:
        index = ConstraintIndex()
        index.add_product({
            "parent_asin": "MATCH",
            "title": "Dress",
            "categories": ["Clothing", "Women", "Dresses"],
            "features": ["100% Cotton"],
            "details": {"Department": "Womens", "Color": "Blue"},
        })
        index.add_product({
            "parent_asin": "WRONG_CATEGORY",
            "title": "Shirt",
            "categories": ["Clothing", "Men", "Shirts"],
            "features": ["100% Cotton"],
            "details": {"Department": "Womens", "Color": "Blue"},
        })

        self.assertEqual(normalize_constraint("100% Cotton"), "100 cotton")
        self.assertEqual(
            index.exact_intersection(
                "Women Dresses", ["100% cotton", "Department: Womens"]
            ),
            {"MATCH"},
        )

    def test_exact_constraint_index_admits_candidates_without_fts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._write_catalog(directory)
            search = CatalogSearch(catalog)
            state = SessionState(user_profile={}, category_text="Shoes")
            state.evidence.extend([
                Evidence("Shoes", 1.4, "category", 1),
                Evidence("full grain leather", 3.8, "hard_constraint", 1),
            ])
            try:
                with patch.object(search, "_route", return_value=[]):
                    result = search.search_with_context(state, limit=2)
            finally:
                search.close()

        self.assertEqual(result.recommendations[0][0], "B")
        self.assertTrue(result.candidates[0]["_exact_constraint_index_match"])

    def test_prebuilt_catalog_index_preserves_rankings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._write_catalog(directory)
            state = SessionState(user_profile={}, category_text="Shoes")
            state.evidence.extend([
                Evidence("Shoes", 1.4, "category", 1),
                Evidence("full grain leather", 3.8, "hard_constraint", 1),
            ])
            in_memory = CatalogSearch(catalog, use_prebuilt_index=False)
            try:
                expected = in_memory.search(state, limit=2)
            finally:
                in_memory.close()

            artifact = build_catalog_index(catalog)
            prebuilt = CatalogSearch(catalog)
            try:
                actual = prebuilt.search(state, limit=2)
                self.assertTrue(prebuilt.using_prebuilt_index)
            finally:
                prebuilt.close()

        self.assertTrue(artifact.name.endswith("_index.sqlite3"))
        self.assertEqual(actual, expected)

    def test_stale_prebuilt_catalog_index_falls_back_safely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._write_catalog(directory)
            build_catalog_index(catalog)
            catalog.write_text(
                catalog.read_text(encoding="utf-8").replace(
                    "Everyday Boot", "Everyday Boots", 1
                ),
                encoding="utf-8",
            )
            search = CatalogSearch(catalog)
            try:
                self.assertFalse(search.using_prebuilt_index)
                self.assertTrue(search.search)
            finally:
                search.close()

    def test_hard_constraint_route_is_disabled_without_hard_constraints(self) -> None:
        evidence = [Evidence("lightweight", 2.5, "clarification", 2)]

        self.assertEqual(_hard_constraint_and_expression("Shirts", evidence), "")

    def test_leaf_category_match_ignores_generic_taxonomy_nodes(self) -> None:
        store = ProductFeatureStore()
        fields = {field: "" for field in FIELD_WEIGHTS}
        fields["categories"] = "Clothing Shoes Jewelry Women Clothing Leggings"
        leaf = store.add("leaf", fields)
        deeper_fields = dict(fields)
        deeper_fields["categories"] += " Athletic"
        deeper = store.add("deeper", deeper_fields)

        self.assertTrue(CatalogSearch._category_leaf_match(leaf, "Women Leggings"))
        self.assertFalse(
            CatalogSearch._category_leaf_match(deeper, "Women Leggings")
        )

    def test_constraint_sequence_match_uses_specific_active_details(self) -> None:
        store = ProductFeatureStore()
        matching = self._features(
            store,
            "matching",
            features="100% Polyester Imported Zipper closure Machine Wash",
        )
        scattered = self._features(
            store,
            "scattered",
            features="100% Polyester lining Imported buttons Zipper closure",
        )
        query = store.compile_query([
            Evidence("polyester", 3.3, "clarification", 2),
            Evidence("100% Polyester", 3.3, "clarification", 2),
            Evidence("Imported", 3.3, "clarification", 3),
            Evidence("Zipper closure", 3.3, "clarification", 3),
            Evidence("polyester", 6.0, "override", 4),
        ])

        self.assertTrue(CatalogSearch._constraint_sequence_match(matching, query))
        self.assertFalse(CatalogSearch._constraint_sequence_match(scattered, query))

    def test_catalog_tiebreak_rewards_same_field_phrase_coherence(self) -> None:
        store = ProductFeatureStore()
        cohesive = self._features(
            store,
            "cohesive",
            categories="Women Clothing Tunics",
            features="60% polyester button closure hand wash only",
        )
        scattered = self._features(
            store,
            "scattered",
            categories="Women Clothing Tunics",
            title="60% polyester tunic",
            features="button detail",
            details="closure hand wash only",
        )
        query = store.compile_query([
            Evidence("60% polyester", 3.3, "clarification", 2),
            Evidence("button closure", 3.3, "clarification", 3),
            Evidence("hand wash only", 3.3, "clarification", 3),
        ])
        frequency = {
            token: sum(token in product.token_weights for product in (cohesive, scattered))
            for token in set(cohesive.token_weights) | set(scattered.token_weights)
        }

        cohesive_score = CatalogSearch._catalog_tiebreak(
            cohesive, query, "Clothing Tunics", frequency, 2
        )
        scattered_score = CatalogSearch._catalog_tiebreak(
            scattered, query, "Clothing Tunics", frequency, 2
        )

        self.assertGreater(cohesive_score, scattered_score)
        self.assertEqual(cohesive_score[1], 1.0)
        self.assertLess(scattered_score[1], 1.0)

    def test_feature_cache_reuses_entries_and_evicts_least_recently_used(self) -> None:
        store = ProductFeatureStore(max_size=2)
        fields = {field: "cotton shirt" for field in FIELD_WEIGHTS}
        first = store.get_or_add("A", fields)
        self.assertIs(store.get_or_add("A", fields), first)
        store.get_or_add("B", fields)
        store.get_or_add("C", fields)
        info = store.cache_info()
        self.assertEqual(info.hits, 1)
        self.assertEqual(info.misses, 3)
        self.assertEqual(info.evictions, 1)
        self.assertEqual(info.current_size, 2)
        with self.assertRaises(KeyError):
            store.get("A")

    def test_cached_constraint_score_matches_previous_formula(self) -> None:
        store = ProductFeatureStore()
        raw_product = {
            "title": "Trail Boot",
            "categories": "Clothing Shoes Hiking Boots",
            "features": "full grain leather waterproof wide width",
            "details": "material leather color brown",
            "store": "Example",
            "description": "comfortable outdoor walking boot",
        }
        product = store.add(
            "B",
            raw_product,
            price=60,
            average_rating=4.4,
            rating_number=120,
        )
        evidence = [
            Evidence("Hiking Boots", 1.4, "category", 1),
            Evidence("full grain leather; wide width", 3.3, "clarification", 2),
            Evidence("unseen query token", 2.0, "clarification", 3),
        ]
        profile = {"preference_tags": ["comfort", "durability"]}
        query = store.compile_query(evidence, profile)
        cached = CatalogSearch._constraint_score(product, query)
        previous = _legacy_constraint_score(raw_product, evidence, profile)
        self.assertAlmostEqual(cached, previous, places=12)

    def test_exclusion_penalizes_matching_product_and_filters_profile_bonus(self) -> None:
        store = ProductFeatureStore()
        leather = self._features(store, "leather", title="Leather walking shoe")
        canvas = self._features(store, "canvas", title="Canvas walking shoe")
        query = store.compile_query(
            [Evidence("leather", 3.8, "exclusion", 2, "material")],
            {"preference_tags": ["leather", "comfort"]},
        )

        self.assertLess(
            CatalogSearch._constraint_score(leather, query),
            CatalogSearch._constraint_score(canvas, query),
        )
        self.assertNotIn("leather", query.preference_tokens)

    def test_exclusion_reranks_non_matching_product_end_to_end(self) -> None:
        rows = [
            {
                "parent_asin": "LEATHER", "title": "Premium leather walking shoe",
                "categories": ["Shoes"], "features": ["leather comfort"],
                "details": {}, "store": "Example", "description": [], "price": 40,
                "average_rating": 5.0, "rating_number": 10000,
            },
            {
                "parent_asin": "CANVAS", "title": "Canvas walking shoe",
                "categories": ["Shoes"], "features": ["canvas comfort"],
                "details": {}, "store": "Example", "description": [], "price": 40,
                "average_rating": 3.0, "rating_number": 1,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            catalog.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            search = CatalogSearch(catalog)
            state = SessionState(user_profile={}, category_text="Shoes")
            state.evidence.extend([
                Evidence("Shoes", 1.4, "category", 1),
                Evidence("leather", 3.8, "exclusion", 2, "material"),
            ])
            try:
                with patch.object(search, "_route", wraps=search._route) as route:
                    ranked = search.search(state, limit=2)
                self.assertTrue(route.call_args_list)
                self.assertTrue(all(
                    "leather" not in call.args[0].casefold()
                    for call in route.call_args_list
                ))
            finally:
                search.close()

        self.assertEqual([parent_asin for parent_asin, _ in ranked], [
            "CANVAS", "LEATHER",
        ])

    def test_popularity_weight_is_reduced_and_bounded(self) -> None:
        store = ProductFeatureStore()
        obscure = CatalogSearch._quality_tiebreak(
            self._features(store, "obscure", rating_number=1)
        )
        popular = CatalogSearch._quality_tiebreak(
            self._features(store, "popular", rating_number=10000)
        )
        self.assertLess(QUALITY_REVIEW_WEIGHT, 1.20)
        self.assertLess(popular - obscure, 9.5)

    def test_profile_rating_alignment_is_bounded_and_missing_safe(self) -> None:
        store = ProductFeatureStore()
        exact = self._features(store, "exact", average_rating=2.0)
        distant = self._features(store, "distant", average_rating=5.0)

        exact_alignment = CatalogSearch._profile_rating_alignment(
            exact, {"average_prior_rating": 2.0}
        )
        distant_alignment = CatalogSearch._profile_rating_alignment(
            distant, {"average_prior_rating": 2.0}
        )

        self.assertEqual(exact_alignment, 1.0)
        self.assertGreater(exact_alignment, distant_alignment)
        self.assertGreaterEqual(distant_alignment, 0.0)
        self.assertEqual(
            CatalogSearch._profile_rating_alignment(exact, {}), 0.0
        )


    def test_budget_proximity_is_scored_without_a_network_model(self) -> None:
        store = ProductFeatureStore()
        evidence = [Evidence("budget around $50", 3.0, "clarification", 2)]
        query = store.compile_query(evidence)
        close = CatalogSearch._price_score(
            self._features(store, "close", price=52), query
        )
        far = CatalogSearch._price_score(
            self._features(store, "far", price=120), query
        )
        self.assertGreater(close, far)

    def test_buying_policy_penalizes_hard_constraint_contradictions(self) -> None:
        store = ProductFeatureStore()
        matching_raw = {
            "parent_asin": "MATCH",
            "title": "Black leather trail boot",
            "features": "wide width waterproof",
            "categories": "Shoes",
            "details": "",
            "store": "Example",
            "description": "",
        }
        conflicting_raw = {
            **matching_raw,
            "parent_asin": "CONFLICT",
            "title": "Red synthetic trail boot",
        }
        matching = store.add("MATCH", matching_raw)
        conflicting = store.add("CONFLICT", conflicting_raw)
        query = store.compile_query(
            [Evidence("black leather", 3.8, "hard_constraint", 1)]
        )
        policy = DEFAULT_RANKING_POLICIES.buying

        matching_score = CatalogSearch._constraint_fit_adjustment(
            matching,
            store.question_features(matching_raw),
            query,
            policy,
        )
        conflicting_score = CatalogSearch._constraint_fit_adjustment(
            conflicting,
            store.question_features(conflicting_raw),
            query,
            policy,
        )

        self.assertGreater(matching_score, 0.0)
        self.assertLess(conflicting_score, 0.0)
        self.assertGreater(matching_score, conflicting_score)

    def test_legacy_policy_adds_no_mode_specific_adjustment(self) -> None:
        store = ProductFeatureStore()
        raw = {
            "parent_asin": "A",
            "title": "Red synthetic boot",
            "features": "standard width",
            "categories": "Shoes",
            "details": "",
            "store": "Example",
            "description": "",
        }
        product = store.add("A", raw, price=120)
        query = store.compile_query(
            [
                Evidence("black leather", 3.8, "hard_constraint", 1),
                Evidence("under $50", 3.0, "clarification", 2),
            ]
        )

        self.assertEqual(
            CatalogSearch._constraint_fit_adjustment(
                product,
                store.question_features(raw),
                query,
                LEGACY_POLICY,
            ),
            0.0,
        )
        self.assertEqual(
            CatalogSearch._budget_violation_adjustment(
                product, query, LEGACY_POLICY
            ),
            0.0,
        )

    def test_buying_policy_penalizes_explicit_budget_violation(self) -> None:
        store = ProductFeatureStore()
        query = store.compile_query(
            [Evidence("under $50", 3.0, "hard_constraint", 1)]
        )
        policy = DEFAULT_RANKING_POLICIES.buying
        affordable = self._features(store, "affordable", price=45)
        expensive = self._features(store, "expensive", price=100)

        affordable_score = CatalogSearch._budget_violation_adjustment(
            affordable, query, policy
        )
        expensive_score = CatalogSearch._budget_violation_adjustment(
            expensive, query, policy
        )

        self.assertEqual(affordable_score, 0.0)
        self.assertLess(expensive_score, affordable_score)

    def test_search_result_reports_inferred_ranking_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._write_catalog(directory)
            agent = Agent(catalog)
            agent.reset("browse", {})
            agent.respond(
                "browse", "I'm looking for Shoes, but I'm still exploring.", 1, 10
            )
            browsing = agent.search.search_with_context(agent._sessions["browse"])

            agent.reset("buy", {})
            agent.respond(
                "buy",
                "I'm looking for Shoes. A key requirement is: wide width.",
                1,
                10,
            )
            buying = agent.search.search_with_context(agent._sessions["buy"])

            self.assertEqual(browsing.ranking_mode, RankingMode.BROWSING)
            self.assertEqual(buying.ranking_mode, RankingMode.BUYING)

    def test_default_recommendation_policy_uses_live_confidence(self) -> None:
        policy = RecommendationPolicy()

        decisive = (10.0, 8.0, 7.9)
        ambiguous = (10.0, 9.9, 9.8)
        self.assertEqual(
            policy.limit_for(
                2, 10, scores=decisive, hard_constraint_coverage=1.0,
                has_hard_constraints=True,
            ),
            1,
        )
        self.assertEqual(
            policy.limit_for(
                8, 10, scores=ambiguous, has_answerable_clarification=False,
                turns_remaining=2,
            ),
            10,
        )
        self.assertEqual(
            policy.limit_for(
                6, 10, scores=decisive, hard_constraint_coverage=1.0,
                has_hard_constraints=True, has_answerable_clarification=False,
                turns_remaining=4,
            ),
            10,
        )
        self.assertEqual(
            policy.limit_for(
                3, 10, scores=ambiguous, has_answerable_clarification=True,
                turns_remaining=7,
            ),
            1,
        )
        self.assertEqual(
            policy.limit_for(
                3, 10, scores=ambiguous, has_answerable_clarification=True,
                turns_remaining=7, has_intent_override=True,
            ),
            1,
        )
        self.assertEqual(
            policy.limit_for(
                3, 10, scores=ambiguous, has_answerable_clarification=True,
                turns_remaining=7, has_no_preference_reply=True,
            ),
            1,
        )
        self.assertEqual(
            policy.limit_for(
                3, 10, scores=decisive, has_answerable_clarification=False,
                turns_remaining=7,
            ),
            5,
        )
        self.assertEqual(
            policy.limit_for(
                3, 10, scores=ambiguous, has_answerable_clarification=True,
                clarification_expected_value=0.05, turns_remaining=7,
            ),
            10,
        )
        self.assertEqual(
            policy.limit_for(
                3, 10, scores=decisive, has_answerable_clarification=True,
                clarification_expected_value=0.05, turns_remaining=7,
            ),
            5,
        )

    def test_full_breadth_policy_is_available_for_controlled_comparisons(self) -> None:
        config = AgentConfig(recommendation_policy=FULL_BREADTH_POLICY)

        self.assertEqual(config.recommendation_policy.limit_for(1, 10), 10)


    def test_default_agent_does_not_construct_vector_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._write_catalog(directory)
            with patch("mercury.lexical.retrieval.CatalogVectorIndex") as vector_type:
                agent = Agent(catalog)
                try:
                    self.assertIsNone(agent.search.vector_index)
                    vector_type.assert_not_called()
                finally:
                    agent.close()

    def test_vector_reranker_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._write_catalog(directory)
            config = AgentConfig(enable_vector_reranker=True)
            with patch("mercury.lexical.retrieval.CatalogVectorIndex") as vector_type:
                agent = Agent(catalog, config=config)
                try:
                    vector_type.assert_called_once_with(catalog)
                    self.assertIs(agent.search.vector_index, vector_type.return_value)
                finally:
                    agent.close()

    def test_conversation_reranks_exact_constraint_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._write_catalog(directory)
            agent = Agent(catalog)
            agent.reset("s", {})
            first_response = agent.respond(
                "s", "I'm looking for Shoes, but I'm still exploring.", 1, 10
            )
            self.assertEqual(len(first_response["recommendations"]), 1)

            response = agent.respond(
                "s", "For that, what matters is: full grain leather; wide width.", 2, 10
            )
            # The remaining question has no discriminative value, so the
            # policy exposes alternatives instead of collapsing to top-1.
            self.assertEqual(len(response["recommendations"]), 2)
            self.assertEqual(response["recommendations"][0]["parent_asin"], "B")

    def test_exact_hard_constraint_tiers_dominate_numeric_score(self) -> None:
        rows = [
            {
                "parent_asin": "NONE", "title": "Black trail leather shoe",
                "categories": ["Shoes"], "features": ["standard width"],
                "details": {}, "store": "Example", "description": [], "price": 40,
                "average_rating": 5.0, "rating_number": 1000,
            },
            {
                "parent_asin": "SOME", "title": "Black leather shoe",
                "categories": ["Shoes"], "features": ["standard width"],
                "details": {}, "store": "Example", "description": [], "price": 40,
                "average_rating": 4.0, "rating_number": 100,
            },
            {
                "parent_asin": "ALL", "title": "Black leather shoe",
                "categories": ["Shoes"], "features": ["wide width"],
                "details": {}, "store": "Example", "description": [], "price": 40,
                "average_rating": 3.0, "rating_number": 1,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            catalog.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            search = CatalogSearch(catalog)
            state = SessionState(user_profile={})
            state.evidence.extend([
                Evidence("Shoes", 1.4, "category", 1),
                Evidence("black leather", 3.8, "hard_constraint", 1),
                Evidence("wide width", 3.8, "hard_constraint", 1),
            ])
            inflated_quality = {"NONE": 1_000.0, "SOME": 500.0, "ALL": 0.0}
            try:
                with patch.object(
                    CatalogSearch,
                    "_quality_tiebreak",
                    side_effect=lambda product: inflated_quality[product.parent_asin],
                ):
                    ranked = search.search(state, limit=3)
            finally:
                search.close()

        self.assertEqual([parent_asin for parent_asin, _ in ranked], [
            "ALL", "SOME", "NONE",
        ])

    def test_cache_capacity_does_not_change_recommendations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._write_catalog(directory)
            small_cache = Agent(catalog, feature_cache_size=1)
            normal_cache = Agent(catalog, feature_cache_size=5000)
            for agent in (small_cache, normal_cache):
                agent.reset("s", {})
                agent.respond(
                    "s", "I'm looking for Shoes, but I'm still exploring.", 1, 10
                )
            message = "For that, what matters is: full grain leather; wide width."
            small_response = small_cache.respond("s", message, 2, 10)
            normal_response = normal_cache.respond("s", message, 2, 10)
            self.assertEqual(
                small_response["recommendations"], normal_response["recommendations"]
            )


class LexicalBackendBoundaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.catalog = Path(self.temp.name) / "catalog.jsonl"
        self.catalog.write_text("\n".join(json.dumps({
            "parent_asin": str(index), "title": "Blue cotton shirt",
            "categories": ["Shirts"], "features": ["cotton"],
        }) for index in range(12)))

    def agent(self, **kwargs: object) -> Agent:
        agent = Agent(self.catalog, **kwargs)
        self.addCleanup(agent.close)
        return agent

    def test_full_width_control_bypasses_independent_sibling_deferral(self) -> None:
        default = self.agent()
        control = self.agent(config=FULL_WIDTH_CONFIG)
        for agent in (default, control):
            agent.reset("session", {})
        message = "I'm looking for Shirts, but I'm still exploring."
        self.assertEqual(default.respond("session", message, 1, 10)["recommendations"], [])
        for turn in range(1, 11):
            response = control.respond("session", message, turn, 10)
            identifiers = [item["parent_asin"] for item in response["recommendations"]]
            self.assertEqual(len(identifiers), 10)
            self.assertEqual(len(set(identifiers)), 10)
            self.assertTrue(set(identifiers) <= {str(index) for index in range(12)})
            self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})
        self.assertNotIn("session", control._ambiguity_deferred)

    def test_request_profile_seed_is_kept_without_cross_session_learning(self) -> None:
        agent = self.agent()
        agent.reset("first", {"profile_id": "shared", "preference_tags": ["cotton"]})
        first = agent._sessions["first"].long_term_profile
        first.observe("material", "leather", 1, durable=True, replacement=False)
        agent.reset("second", {"profile_id": "shared", "preference_tags": ["linen"]})
        second = agent._sessions["second"].long_term_profile
        self.assertIsNot(first, second)
        self.assertEqual({item.value for item in second.learned.values()}, {"linen"})
        self.assertIn("leather", {item.value for item in first.learned.values()})

    def test_cross_session_profile_sharing_requires_opt_in(self) -> None:
        agent = self.agent(share_profile_memory=True)
        agent.reset("first", {"profile_id": "shared"})
        first = agent._sessions["first"].long_term_profile
        first.observe("material", "leather", 1, durable=True, replacement=False)
        agent.reset("second", {"profile_id": "shared"})
        self.assertIs(agent._sessions["second"].long_term_profile, first)

    def test_lru_eviction_and_reset_clear_every_session_map(self) -> None:
        agent = self.agent(max_sessions=2, config=FULL_WIDTH_CONFIG)
        agent.reset("first", {})
        agent.reset("second", {})
        agent._ambiguity_deferred.update({"first", "second"})
        agent.respond("first", "A cotton shirt", 1, 10)
        agent.reset("third", {})
        self.assertEqual(list(agent._sessions), ["first", "third"])
        self.assertEqual(set(agent._profile_ids), {"first", "third"})
        self.assertEqual(set(agent.profile_store.profiles), {"first", "third"})
        self.assertNotIn("second", agent._ambiguity_deferred)
        with self.assertRaises(RuntimeError):
            agent.respond("second", "A shirt", 2, 10)
        previous = agent._sessions["first"]
        agent.reset("first", {})
        self.assertIsNot(agent._sessions["first"], previous)
        self.assertNotIn("first", agent._ambiguity_deferred)
        agent.close()
        for retained in (agent._sessions, agent._profile_ids,
                         agent._ambiguity_deferred, agent.profile_store.profiles):
            self.assertFalse(retained)

    def test_invalid_session_capacity_is_rejected_before_catalog_load(self) -> None:
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                Agent("missing-catalog.jsonl", max_sessions=size)

    def test_default_is_importable_without_optional_clients_or_array_library(self) -> None:
        code = """
import importlib.abc
import socket
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'openai', 'dotenv'}:
            raise ImportError('optional dependency unavailable')
def denied(*args, **kwargs):
    raise AssertionError('network access attempted')
sys.meta_path.insert(0, BlockOptional())
socket.create_connection = denied
socket.socket.connect = denied
from mercury.lexical import Agent, FULL_WIDTH_CONFIG
agent = Agent(sys.argv[1], config=FULL_WIDTH_CONFIG)
try:
    agent.reset('offline', {})
    result = agent.respond('offline', 'A cotton shirt', 1, 10)
    assert len(result['recommendations']) == 10
    assert result['usage'] == {'prompt_tokens': 0, 'completion_tokens': 0}
finally:
    agent.close()
"""
        result = subprocess.run(
            [sys.executable, "-c", code, str(self.catalog)],
            cwd=Path(__file__).resolve().parents[1], capture_output=True,
            text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_optional_vector_artifact_and_query_cache_use_injected_client(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("optional array library unavailable")
        vectors = Path(self.temp.name) / "vectors.npy"
        metadata = Path(self.temp.name) / "vectors.json"
        np.save(vectors, np.asarray([[1.0, 0.0], [0.0, 1.0]] * 6, dtype=np.float32))
        metadata.write_text(json.dumps({
            "model": "test-encoder", "dimensions": 2, "row_count": 12,
            "catalog_sha256": catalog_sha256(self.catalog), "normalized": True,
        }))
        client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[1.0, 0.0])],
            usage=SimpleNamespace(prompt_tokens=7),
        )))
        index = CatalogVectorIndex(
            self.catalog, vectors_path=vectors, metadata_path=metadata, client=client,
        )
        self.addCleanup(index.close)
        with patch.object(client.embeddings, "create", wraps=client.embeddings.create) as embed:
            first = index.search("cotton shirt", limit=6)
            second = index.search("cotton shirt", limit=6)
        self.assertEqual(first.rows, second.rows)
        self.assertEqual({row for row, _ in first.rows}, {1, 3, 5, 7, 9, 11})
        self.assertEqual((first.prompt_tokens, second.prompt_tokens), (7, 0))
        self.assertEqual(embed.call_count, 1)
        metadata.write_text(metadata.read_text().replace(catalog_sha256(self.catalog), "0" * 64))
        invalid = CatalogVectorIndex(
            self.catalog, vectors_path=vectors, metadata_path=metadata, client=client,
        )
        self.addCleanup(invalid.close)
        self.assertFalse(invalid.enabled)
        self.assertEqual(invalid.search("cotton shirt").rows, [])


class LexicalKnownSemanticsTest(unittest.TestCase):
    @unittest.expectedFailure
    def test_override_retires_color_received_through_open_question(self) -> None:
        state = SessionState(user_profile={})
        state.record_question("other")
        state.observe("For that, what matters is: blue.", 1)
        state.observe("Actually, what I need is: black.", 2)
        self.assertNotIn("blue", [item.text for item in state.evidence])

    @unittest.expectedFailure
    def test_short_negative_is_not_positive_search_evidence(self) -> None:
        state = SessionState(user_profile={})
        state.observe("I'm looking for Shoes, but I'm still exploring.", 1)
        state.observe("No leather.", 2)
        leather = [item for item in state.evidence if "leather" in item.text.casefold()]
        self.assertTrue(leather)
        self.assertTrue(all(item.operation is PreferenceOperation.EXCLUDE for item in leather))

    @unittest.expectedFailure
    def test_disjunctive_material_is_satisfied_by_one_branch(self) -> None:
        product = {"features": "100% cotton", "categories": "Shirts"}
        exact_count, hard_count = CatalogSearch._hard_constraint_exactness(
            product, [Evidence("cotton or linen", 3.8, "hard_constraint", 1)],
        )
        self.assertEqual((exact_count, hard_count), (1, 1))


if __name__ == "__main__":
    unittest.main()
