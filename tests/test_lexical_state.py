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
    def test_natural_correction_replaces_multiple_facets_and_preserves_kept_evidence(self):
        for opening in ("Correction: make that", "Correction:", "Make that", "Let me correct that:"):
            with self.subTest(opening=opening):
                state = SessionState({})
                state.observe("A key requirement is: navy; wool; machine washable.", 1)
                kept = state.evidence[-1]
                state.observe(f"{opening} green and linen, but keep machine washable.", 2)
                self.assertEqual({item.text for item in state.evidence}, {"green and linen", "machine washable"})
                self.assertIn(kept, state.evidence)
                self.assertEqual(len([item for item in state.evidence if item.turn == 2]), 1)
                replacement = next(item for item in state.evidence if item.turn == 2)
                self.assertEqual(replacement.operation, PreferenceOperation.REPLACE)
                self.assertEqual(replacement.source, "override")

    def test_existing_override_phrase_replaces_every_named_facet(self):
        state = SessionState({})
        state.observe("A key requirement is: black; wool; side vents.", 1)
        state.observe("Actually, what I need is: green linen.", 2)
        self.assertEqual({item.text for item in state.evidence}, {"green linen", "side vents"})

    def test_multifacet_replacement_preserves_compatible_composition(self):
        state = SessionState({})
        state.observe("A key requirement is: black; 80% cotton and 20% polyester; rounded hem.", 1)
        state.observe("Correction: blue cotton.", 2)
        self.assertEqual({item.text for item in state.evidence},
                         {"blue cotton", "80% cotton and 20% polyester", "rounded hem"})

    def test_multifacet_replacement_is_atomic_across_value_groups(self):
        state = SessionState({})
        state.observe("A key requirement is: black; wool.", 1)
        state.observe("Actually, what I need is: blue cotton; red.", 2)
        self.assertEqual({item.text for item in state.evidence}, {"blue cotton", "red"})

    def test_multifacet_component_correction_preserves_another_owner(self):
        state = SessionState({})
        state.observe("A key requirement is: lining: black cotton; upper: brown leather; side vents.", 1)
        state.observe("Correction: lining: blue silk.", 2)
        self.assertEqual({item.text for item in state.evidence},
                         {"lining: blue silk", "upper: brown leather", "side vents"})

    def test_unasserted_correction_words_do_not_change_active_preferences(self):
        for message in ('"Correction: make that blue canvas."',
                        'The label says "Correction: make that blue canvas."',
                        "Do not make that blue canvas.",
                        "I'm not making a correction: blue canvas.",
                        "If I make a correction, make that blue canvas.",
                        "Maybe a correction to blue canvas."):
            with self.subTest(message=message):
                state = SessionState({})
                state.observe("A key requirement is: black; leather.", 1)
                before = list(state.evidence)
                state.observe(message, 2)
                self.assertEqual(state.evidence, before)

    def test_quoted_correction_does_not_hide_an_independent_request(self):
        state = SessionState({})
        state.observe("A key requirement is: black; leather.", 1)
        state.observe('The label says "Correction: blue canvas", and I need an adjustable strap.', 2)
        self.assertEqual({item.text for item in state.evidence},
                         {"black", "leather", "I need an adjustable strap"})

    def test_explicit_correction_keeps_a_requested_literal_contrast(self):
        state = SessionState({})
        phrase = 'a graphic reading "blue rather than black"'
        state.observe(f"Correction: {phrase}.", 1)
        self.assertEqual([item.text for item in state.evidence], [phrase])
        self.assertEqual(state.evidence[0].source, "override")

    def test_unasserted_correction_cannot_activate_earlier_override_controls(self):
        messages = (
            '"Correction: ignore my earlier preference. What I need is: blue canvas."',
            'The label says "Correction: ignore my earlier preference. What I need is: blue canvas."',
            "If I make a correction, ignore my earlier preference. What I need is: blue canvas.",
            '"Correction: blue canvas rather than black leather."',
            "If I make a correction, blue canvas instead of black leather.",
        )
        for message in messages:
            with self.subTest(message=message):
                state = SessionState({})
                state.observe("I'm looking for bags. I prefer black leather.", 1)
                before = list(state.evidence)
                state.observe(message, 2)
                self.assertEqual(state.evidence, before)
                self.assertEqual(state.category_text, "bags")

    def test_exclusive_replacement_retires_an_incompatible_combination(self):
        for value in ("blue only", "only blue", "color: blue only", "blue exclusively"):
            with self.subTest(value=value):
                state = SessionState({})
                state.observe("A key requirement is: red and blue; cotton.", 1)
                state.observe(f"Actually, what I need is: {value}.", 2)
                self.assertEqual({item.text for item in state.evidence}, {"cotton", value})

    def test_incidental_only_word_does_not_invent_exclusive_composition(self):
        state = SessionState({})
        state.observe("A key requirement is: cotton and polyester.", 1)
        state.observe("Actually, what I need is: cotton only for work.", 2)
        self.assertTrue(any(item.text == "cotton and polyester" for item in state.evidence))

    def test_reported_quote_keeps_an_independent_affirmative_clause(self):
        for separator in (", and ", "; ", ". ", " and "):
            with self.subTest(separator=separator):
                state = SessionState({})
                state.observe('The label says "wool"' + separator + 'I need cotton.', 1)
                self.assertEqual([item.text for item in state.evidence], ["I need cotton"])

    def test_first_person_request_inside_reported_quote_is_not_an_instruction(self):
        state = SessionState({})
        state.observe('The label says "wool, and I need silk", and I need cotton.', 1)
        self.assertEqual([item.text for item in state.evidence], ["I need cotton"])

    def test_uncertain_clause_keeps_an_independent_clear_requirement(self):
        state = SessionState({})
        state.observe("Maybe wool, and I need cotton.", 1)
        self.assertEqual([item.text for item in state.evidence], ["I need cotton"])

    def test_no_additional_preference_preserves_existing_requirements(self):
        for wording in ("I have no additional preference for feature.",
                        "I don't have an additional material preference.",
                        "No additional preference for lining material."):
            with self.subTest(wording=wording):
                state = SessionState({})
                state.record_question("feature")
                state.observe("Key requirement is: rounded hem; cotton; lining: silk.", 1)
                before = list(state.evidence)
                state.observe(wording, 2)
                self.assertEqual(state.evidence, before)

    def test_omitting_additional_still_retracts_the_named_preference(self):
        state = SessionState({})
        state.observe("Key requirement is: rounded hem; cotton.", 1)
        state.observe("I have no preference for material.", 2)
        self.assertEqual([item.text for item in state.evidence], ["rounded hem"])

    def test_neutral_answer_preserves_the_requested_question_attribute(self):
        for attribute in ("other", "feature", "style", "material"):
            with self.subTest(attribute=attribute):
                state = SessionState({})
                state.record_question(attribute)
                state.observe("I have no preference.", 1)
                self.assertEqual(state.no_preference_attributes, {attribute})

    def test_explicit_neutral_attribute_overrides_the_last_question(self):
        for attribute in ("other", "feature"):
            with self.subTest(attribute=attribute):
                state = SessionState({})
                state.record_question("material")
                state.observe(f"I have no preference for {attribute}.", 1)
                self.assertEqual(state.no_preference_attributes, {attribute})

    def test_absence_word_in_category_does_not_destroy_category(self):
        state = SessionState({})
        state.observe("I'm looking for no show socks. I prefer cotton.", 1)
        self.assertEqual(state.category_text, "no show socks")
        self.assertTrue(any(item.text == "I prefer cotton" for item in state.evidence))

    def test_explicit_literal_quote_is_a_requirement_not_uncertainty(self):
        state = SessionState({})
        message = 'I am looking for shirts. A key requirement is: a graphic reading "maybe later".'
        state.observe(message, 1)
        self.assertEqual(state.category_text, "shirts")
        self.assertTrue(any(item.text == 'a graphic reading "maybe later"' and item.source == "hard_constraint"
                            for item in state.evidence))

    def test_incidental_hedge_after_conjunction_preserves_explicit_raw_requirement(self):
        state = SessionState({})
        phrase = 'Printed text says "ready now", but maybe tomorrow is the next line'
        state.observe(f"A key requirement is: {phrase}.", 1)
        self.assertEqual([item.text for item in state.evidence], [phrase])

    def test_mixed_uncertainty_does_not_erase_clear_clause(self):
        state = SessionState({})
        state.observe("I want cotton; maybe wool.", 1)
        self.assertTrue(any("cotton" in item.text for item in state.evidence))
        self.assertFalse(any("wool" in item.text for item in state.evidence))

    def test_explicit_opening_retraction_occurs_before_exclusion_handling(self):
        state = SessionState({})
        state.observe("I'm looking for shirts. I prefer red.", 1)
        state.observe("Actually, ignore my earlier preference. What I need is: cotton; no leather.", 2)
        self.assertFalse(any("red" in item.text for item in state.evidence))
        self.assertTrue(any(item.text == "cotton" for item in state.evidence))
        self.assertEqual(state.category_text, "shirts")

    def test_compatible_material_retains_precise_composition(self):
        state = SessionState({})
        state.observe("For that, what matters is: 80% cotton and 20% polyester.", 1)
        state.observe("Actually, what I need is: cotton.", 2)
        self.assertEqual({item.text for item in state.evidence}, {"80% cotton and 20% polyester", "cotton"})

    def test_changed_material_percentage_retires_incompatible_composition(self):
        state = SessionState({})
        state.observe("For that, what matters is: 80% Cotton and 20% Polyester.", 1)
        state.observe("Actually, what I need is: 100% cotton.", 2)
        self.assertEqual([item.text for item in state.evidence], ["100% cotton"])

    def test_new_generic_feature_does_not_retract_unrelated_features(self):
        state = SessionState({})
        state.observe("For that, what matters is: rounded hem; machine washable.", 1)
        state.observe("Actually, what I need is: side vents.", 2)
        self.assertEqual({item.text for item in state.evidence}, {"rounded hem", "machine washable", "side vents"})

    def test_replacing_an_explicit_detail_field_retires_only_that_field(self):
        state = SessionState({})
        state.observe("For that, what matters is: closure: zipper; care: machine wash.", 1)
        state.observe("Actually, what I need is: closure: buttons.", 2)
        self.assertEqual({item.text for item in state.evidence}, {"closure: buttons", "care: machine wash"})

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

    def test_named_facet_inside_no_preference_retracts_only_that_facet(self):
        state = SessionState({})
        state.observe("I'm looking for shirts.", 1)
        state.observe("For that, what matters is: red; cotton; rounded hem.", 2)
        state.observe("Actually, what I need is: blue.", 3)
        state.observe("I don't have a color preference.", 4)
        self.assertEqual({item.text for item in state.evidence}, {"shirts", "cotton", "rounded hem"})

    def test_scoped_exclusion_retires_a_reordered_alternative(self):
        state = SessionState({})
        state.observe("For that, what matters is: lining: cotton or silk.", 1)
        state.observe("No cotton lining.", 2)
        self.assertEqual([item.text for item in state.evidence if item.source != "exclusion"], ["lining: silk"])

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

    def test_natural_correction_ranks_the_new_combination_without_narrowing_output(self):
        from mercury.lexical.agent import Agent
        from mercury.lexical.config import FULL_WIDTH_CONFIG

        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            products = [{"parent_asin": "wanted", "categories": ["bags"],
                         "features": ["green linen", "adjustable strap"]}]
            products.extend({"parent_asin": f"old{index}", "categories": ["bags"],
                             "features": ["navy wool", "adjustable strap"], "rating_number": 100000}
                            for index in range(11))
            catalog.write_text("\n".join(json.dumps(product) for product in products) + "\n")
            agent = Agent(catalog, config=FULL_WIDTH_CONFIG)
            self.addCleanup(agent.close)
            agent.reset("authored", {})
            agent.respond("authored", "I'm looking for bags. A key requirement is: navy; wool; adjustable strap.", 1, 10)
            response = agent.respond("authored", "Correction: make that green and linen, but keep the adjustable strap.", 2, 10)
            self.assertEqual(response["recommendations"][0]["parent_asin"], "wanted")
            self.assertEqual(len(response["recommendations"]), 10)
            self.assertEqual(agent.last_diagnostics["stage_counts"]["retrieval_union"], len(products))

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
