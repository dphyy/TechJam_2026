from __future__ import annotations

import json
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path

from mercury.lexical.constraint_index import ConstraintIndex, open_catalog_index
from mercury.lexical.dialogue import Evidence, PreferenceOperation, SessionState
from mercury.lexical.preprocessing import build_catalog_index
from mercury.lexical.product_features import ProductFeatureStore
from mercury.lexical.ranking import DEFAULT_RANKING_POLICIES
from mercury.lexical.retrieval import CatalogSearch, _hard_constraint_and_expression


class ActiveEvidenceTest(unittest.TestCase):
    def test_correction_retires_only_named_facet_after_open_question(self):
        state = SessionState({})
        state.record_question("other")
        state.observe("For that, what matters is: blue; full grain leather.", 1)
        state.observe("Actually, black instead.", 2)
        texts = [item.text.casefold() for item in state.evidence]
        self.assertNotIn("blue", texts)
        self.assertIn("full grain leather", texts)
        self.assertTrue(any("black" in text for text in texts))

    def test_positive_raw_phrase_is_not_rewritten(self):
        state = SessionState({})
        state.observe("Key requirement is: 80% cotton; machine washable.", 1)
        self.assertEqual([(x.text, x.weight, x.source) for x in state.evidence],
                         [("80% cotton", 3.8, "hard_constraint"),
                          ("machine washable", 3.8, "hard_constraint")])

    def test_ordinary_style_phrases_do_not_become_component_requirements(self):
        store = ProductFeatureStore()
        for wording in ("long sleeve", "short sleeve", "long sleeve cotton", "large pockets"):
            with self.subTest(wording=wording):
                query = store.compile_query([Evidence(wording, 3.8, "hard_constraint", 1)])
                self.assertIsNone(query.evidence[0].scope)

    def test_short_and_mixed_negatives_have_no_positive_copy(self):
        for text in ("No leather.", "Without leather.", "I want cotton, but no leather."):
            with self.subTest(text=text):
                state = SessionState({})
                state.observe(text, 1)
                mentions = [x for x in state.evidence if "leather" in x.text.casefold()]
                self.assertTrue(mentions)
                self.assertTrue(all(x.operation is PreferenceOperation.EXCLUDE for x in mentions))
                if "cotton" in text:
                    self.assertTrue(any("cotton" in x.text and x.source != "exclusion" for x in state.evidence))

    def test_retraction_does_not_retain_old_color_or_remove_material(self):
        state = SessionState({})
        state.observe("Key requirement is: blue; cotton.", 1)
        state.observe("No preference for color.", 2)
        self.assertEqual([x.text for x in state.evidence], ["cotton"])

    def test_component_correction_and_neutrality_keep_other_owner(self):
        state = SessionState({})
        state.observe("Key requirement is: lining: cotton; upper: leather.", 1)
        state.observe("Actually, what I need is: lining: wool.", 2)
        self.assertEqual({x.text for x in state.evidence}, {"lining: wool", "upper: leather"})
        state.observe("No preference for lining material.", 3)
        self.assertEqual([x.text for x in state.evidence], ["upper: leather"])

    def test_negated_branch_correction_removes_old_positive(self):
        state = SessionState({})
        state.observe("Key requirement is: blue.", 1)
        state.observe("Not blue, but black.", 2)
        self.assertTrue(any("black" in x.text for x in state.evidence if x.source != "exclusion"))
        self.assertFalse(any("blue" in x.text for x in state.evidence if x.source != "exclusion"))

    def test_quoted_negative_and_uncertain_mentions_do_not_become_requirements(self):
        state = SessionState({})
        state.observe('The label says "no leather"; I want cotton.', 1)
        self.assertFalse(any("leather" in x.text for x in state.evidence))
        state.observe("Maybe wool, I am not sure.", 2)
        self.assertFalse(any("wool" in x.text for x in state.evidence))

    def test_excluding_one_alternative_retains_the_other(self):
        state = SessionState({})
        state.observe("Key requirement is: cotton or linen.", 1)
        state.observe("No linen.", 2)
        self.assertEqual([x.text for x in state.evidence if x.source != "exclusion"], ["cotton"])

    def test_contrast_keeps_new_value_and_excludes_old_value(self):
        for wording in ("black rather than blue", "black instead of blue"):
            with self.subTest(wording=wording):
                state = SessionState({})
                state.observe("Key requirement is: blue.", 1)
                state.observe(wording, 2)
                self.assertEqual([x.text for x in state.evidence if x.source != "exclusion"], ["black"])

    def test_correction_keeps_other_facet_in_the_same_original_phrase(self):
        state = SessionState({})
        state.observe("Key requirement is: blue cotton.", 1)
        state.observe("Actually, what I need is: black.", 2)
        self.assertEqual({x.text for x in state.evidence}, {"cotton", "black"})

    def test_category_replacement_does_not_keep_stale_category(self):
        state = SessionState({})
        state.observe("I'm looking for shirts.", 1)
        state.observe("Actually, I'm looking for shoes instead.", 2)
        self.assertEqual([x.text for x in state.evidence if x.source == "category"], ["shoes"])


class EvidenceRankingTest(unittest.TestCase):
    def setUp(self):
        self.store = ProductFeatureStore()

    def product(self, key, **fields):
        return self.store.add(key, fields)

    def test_or_satisfies_one_hard_group_and_scores_strongest_witness_once(self):
        item = Evidence("cotton or linen", 3.8, "hard_constraint", 1)
        one = self.product("one", features="cotton")
        both = self.product("both", features="cotton linen")
        query = self.store.compile_query([item])
        self.assertEqual(CatalogSearch._hard_constraint_exactness(one, [item]), (1, 1))
        self.assertEqual(CatalogSearch._constraint_score(one, query),
                         CatalogSearch._constraint_score(both, query))
        self.assertIn('("cotton" OR "linen")', _hard_constraint_and_expression("shirts", [item]))

    def test_or_index_unions_branches_then_intersects_other_requirements(self):
        index = ConstraintIndex()
        for key, material, color in (("a", "cotton", "blue"), ("b", "linen", "blue"), ("c", "wool", "blue"), ("d", "linen", "red")):
            index.add_product({"parent_asin": key, "categories": ["shirts"], "features": [material, color]})
        self.assertEqual(index.exact_intersection("shirts", ["cotton or linen", "blue"]), {"a", "b"})

    def test_component_support_cannot_come_from_another_component(self):
        query = self.store.compile_query([Evidence("lining: cotton", 3.8, "hard_constraint", 1)])
        correct = self.product("correct", details="lining cotton upper leather")
        wrong = self.product("wrong", details="lining leather upper cotton")
        unknown = self.product("unknown", features="cotton")
        self.assertAlmostEqual(CatalogSearch._constraint_score(correct, query), 13.756, places=12)
        self.assertGreater(CatalogSearch._constraint_score(correct, query), CatalogSearch._constraint_score(wrong, query))
        self.assertEqual(CatalogSearch._constraint_score(unknown, query), 0)

    def test_scoped_exclusion_does_not_penalize_unrelated_owner_or_unknown(self):
        query = self.store.compile_query([Evidence("leather lining", 3.8, "exclusion", 1)])
        safe = self.product("safe", details="lining cotton upper leather")
        unsafe = self.product("unsafe", details="lining leather upper cotton")
        unknown = self.product("unknown", features="leather")
        self.assertEqual(CatalogSearch._constraint_score(safe, query), 0)
        self.assertEqual(CatalogSearch._constraint_score(unknown, query), 0)
        self.assertLess(CatalogSearch._constraint_score(unsafe, query), 0)

    def test_zero_budget_and_nonfinite_catalog_numbers_are_safe(self):
        product = self.store.add("priced", {"title": "shirt"}, price=10, rating_number="1e309", average_rating="inf")
        query = self.store.compile_query([Evidence("under $0", 3.8, "hard_constraint", 1)])
        self.assertTrue(math.isfinite(CatalogSearch._price_score(product, query)))
        self.assertTrue(math.isfinite(CatalogSearch._budget_violation_adjustment(product, query, DEFAULT_RANKING_POLICIES.buying)))
        self.assertEqual(product.rating_number, 0)
        self.assertEqual(product.average_rating, 0)

    def test_catalog_negation_is_not_proof_of_excluded_material(self):
        query = self.store.compile_query([Evidence("leather", 3.8, "exclusion", 1)])
        for key, wording in enumerate(("no leather", "leather-free", "without leather", "not leather")):
            product = self.product(str(key), features=wording)
            self.assertEqual(CatalogSearch._constraint_score(product, query), 0)
            self.assertFalse(CatalogSearch._semantic_violation(product, query))

    def test_component_positive_does_not_accept_negated_catalog_value(self):
        product = self.product("negative", details="lining: no cotton")
        evidence = [Evidence("lining: cotton", 3.8, "hard_constraint", 1)]
        self.assertEqual(CatalogSearch._constraint_score(product, self.store.compile_query(evidence)), 0)
        self.assertEqual(CatalogSearch._hard_constraint_exactness(product, evidence), (0, 1))

    def test_nonfinite_and_boolean_catalog_numbers_are_neutral(self):
        for key, number in enumerate((True, False, "inf", "nan", "-inf", "1e309")):
            product = self.store.add(f"number{key}", {}, price=number, average_rating=number, rating_number=number)
            self.assertIsNone(product.price)
            self.assertEqual(product.average_rating, 0)
            self.assertEqual(product.rating_number, 0)

    def test_contradiction_precedes_popularity_and_exact_positive_tiers(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            records = [{"parent_asin": "unsafe", "categories": ["shirts"], "features": ["cotton", "leather"], "rating_number": 1_000_000}]
            records.extend({"parent_asin": f"safe{index}", "categories": ["shirts"], "features": ["cotton"]} for index in range(11))
            catalog.write_text("\n".join(json.dumps(record) for record in records) + "\n")
            state = SessionState({})
            state.observe("I'm looking for shirts.", 1)
            state.observe("Key requirement is: cotton; no leather.", 2)
            with CatalogSearchContext(catalog) as search:
                result = search.search_with_context(state, limit=10)
            self.assertEqual(len(result.recommendations), 10)
            self.assertNotIn("unsafe", [key for key, _ in result.recommendations])
            self.assertEqual({row["parent_asin"] for row in result.candidates}, {row["parent_asin"] for row in records})


class PersistentIdentityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.catalog = Path(self.temp.name) / "catalog.jsonl"
        self.catalog.write_text("\n".join(json.dumps({"parent_asin": key, "categories": ["shirts"], "features": [material]}) for key, material in (("a", "cotton"), ("b", "linen"))) + "\n")
        self.index = build_catalog_index(self.catalog)

    def test_row_identity_swap_is_rejected(self):
        with sqlite3.connect(self.index) as connection:
            connection.execute("UPDATE product_rows SET row_id = row_id + 10")
            connection.execute("UPDATE product_rows SET row_id = 13 - row_id")
        self.assertIsNone(open_catalog_index(self.catalog, self.index))

    def test_missing_row_table_falls_back(self):
        with sqlite3.connect(self.index) as connection:
            connection.execute("DROP TABLE product_rows")
        self.assertIsNone(open_catalog_index(self.catalog, self.index))

    def test_unknown_constraint_membership_falls_back(self):
        with sqlite3.connect(self.index) as connection:
            connection.execute("INSERT INTO constraint_entries VALUES ('constraint_to_asins', 'silk', 'missing')")
        self.assertIsNone(open_catalog_index(self.catalog, self.index))

    def test_false_constraint_for_existing_product_falls_back(self):
        with sqlite3.connect(self.index) as connection:
            connection.execute("INSERT INTO constraint_entries VALUES ('constraint_to_asins', 'silk', 'a')")
        self.assertIsNone(open_catalog_index(self.catalog, self.index))

    def test_changed_product_field_with_unchanged_metadata_falls_back(self):
        with sqlite3.connect(self.index) as connection:
            connection.execute("UPDATE products SET features = 'silk' WHERE parent_asin = 'a'")
        self.assertIsNone(open_catalog_index(self.catalog, self.index))

    def test_missing_required_product_column_falls_back(self):
        with sqlite3.connect(self.index) as connection:
            connection.execute("DROP TABLE products")
            connection.execute("CREATE TABLE products(parent_asin TEXT)")
        self.assertIsNone(open_catalog_index(self.catalog, self.index))

    def test_or_exact_route_has_persistent_parity(self):
        with CatalogSearchContext(self.catalog) as search:
            self.assertTrue(search.using_prebuilt_index)
            self.assertEqual(search.constraint_index.exact_intersection("shirts", ["cotton or linen"]), {"a", "b"})


class CatalogSearchContext(CatalogSearch):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    unittest.main()
