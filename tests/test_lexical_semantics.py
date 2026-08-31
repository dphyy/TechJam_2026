from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from mercury.fusion import make_agent
from mercury.lexical.constraint_index import ConstraintIndex, _catalog_text, open_catalog_index
from mercury.lexical.dialogue import Evidence
from mercury.lexical.product_features import (
    ProductFeatureStore, exclusive_facet_values, hard_evidence_match, resolve_query, terms,
)
from mercury.lexical.preprocessing import build_catalog_index
from mercury.lexical.retrieval import CatalogSearch, _text


class LexicalSemanticsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = ProductFeatureStore()

    def product(self, identifier: str, **fields):
        return self.store.add(identifier, fields)

    def test_negated_material_is_contradiction_and_unknown_is_not(self) -> None:
        evidence = [Evidence("cotton", 3.8, "hard_constraint", 1)]
        query = self.store.compile_query(evidence)
        unknown = self.product("unknown", features="plain finish")
        self.assertEqual(CatalogSearch._constraint_score(unknown, query), 0)
        self.assertFalse(CatalogSearch._semantic_violation(unknown, query))
        for index, text in enumerate(("No cotton", "without cotton", "cotton-free", "does not contain cotton")):
            with self.subTest(text=text):
                excluded = self.product(str(index), features=text)
                self.assertEqual(CatalogSearch._constraint_score(excluded, query), 0)
                self.assertEqual(CatalogSearch._hard_constraint_exactness(excluded, evidence), (0, 1))
                self.assertTrue(CatalogSearch._semantic_violation(excluded, query))

    def test_natural_component_prose_assigns_values_to_the_named_owner(self) -> None:
        evidence = [Evidence("upper: cotton", 3.8, "hard_constraint", 1)]
        query = self.store.compile_query(evidence)
        for index, (wrong_text, correct_text) in enumerate((
            ("The lining is cotton and the upper is leather", "The upper is cotton"),
            ("Its lining is cotton; its upper is leather", "Its upper is cotton"),
            ("A cotton lining and a leather upper", "A cotton upper"),
            ("lining cotton and leather upper", "upper cotton"),
            ("A cotton lining and upper leather", "A leather lining and upper cotton"),
        )):
            with self.subTest(wrong_text=wrong_text):
                wrong = self.product(f"wrong{index}", description=wrong_text)
                correct = self.product(f"correct{index}", description=correct_text)
                self.assertEqual(CatalogSearch._hard_constraint_exactness(wrong, evidence), (0, 1))
                self.assertEqual(CatalogSearch._hard_constraint_exactness(correct, evidence), (1, 1))
                self.assertLess(CatalogSearch._constraint_score(wrong, query),
                                CatalogSearch._constraint_score(correct, query))

    def test_negative_or_checks_any_affirmed_branch_not_a_negated_title(self) -> None:
        product = self.product("bad", title="not red", features="blue")
        query = self.store.compile_query([Evidence("red or blue", 3.8, "exclusion", 1)])
        self.assertEqual(resolve_query(product, query).evidence[0].tokens, ("blue",))
        self.assertTrue(CatalogSearch._semantic_violation(product, query))
        self.assertLess(CatalogSearch._constraint_score(product, query), 0)
        unknown = self.product("unknown", title="not red", features="green")
        self.assertFalse(CatalogSearch._semantic_violation(unknown, query))
        self.assertEqual(CatalogSearch._constraint_score(unknown, query), 0)

    def test_single_digit_and_single_letter_values_remain_discriminative(self) -> None:
        for index, (wanted, matching, wrong) in enumerate((
            ("size 8", "Size 8", "Size 9"),
            ("size M", "Size M", "Size S"),
            ("2 mm", "2 mm", "3 mm"),
        )):
            with self.subTest(wanted=wanted):
                evidence = [Evidence(wanted, 3.8, "hard_constraint", 1)]
                query = self.store.compile_query(evidence)
                correct = self.product(f"correct{index}", details=matching)
                incorrect = self.product(f"incorrect{index}", details=wrong)
                self.assertEqual(CatalogSearch._hard_constraint_exactness(correct, evidence), (1, 1))
                self.assertEqual(CatalogSearch._hard_constraint_exactness(incorrect, evidence), (0, 1))
                self.assertGreater(CatalogSearch._constraint_score(correct, query),
                                   CatalogSearch._constraint_score(incorrect, query))

    def test_valid_or_and_explicit_absence_feature_remain_supported(self) -> None:
        evidence = [Evidence("cotton or linen", 3.8, "hard_constraint", 1)]
        query = self.store.compile_query(evidence)
        one = self.product("one", features="cotton")
        both = self.product("both", features="cotton linen")
        self.assertEqual(CatalogSearch._constraint_score(one, query), CatalogSearch._constraint_score(both, query))
        for index, value in enumerate(("fragrance free", "no show socks")):
            with self.subTest(value=value):
                product = self.product(f"absence{index}", features=value)
                items = [Evidence(value, 3.8, "hard_constraint", 1)]
                self.assertEqual(CatalogSearch._hard_constraint_exactness(product, items), (1, 1))
                self.assertGreater(CatalogSearch._constraint_score(product, self.store.compile_query(items)), 0)

    def test_uncertainty_is_neither_positive_proof_nor_explicit_contradiction(self) -> None:
        evidence = [Evidence("cotton", 3.8, "hard_constraint", 1)]
        query = self.store.compile_query(evidence)
        for index, wording in enumerate(("not necessarily cotton", "may not contain cotton",
                                         "It is not clear whether cotton is used")):
            with self.subTest(wording=wording):
                product = self.product(str(index), description=wording)
                self.assertEqual(CatalogSearch._constraint_score(product, query), 0)
                self.assertEqual(CatalogSearch._hard_constraint_exactness(product, evidence), (0, 1))
                self.assertFalse(CatalogSearch._semantic_violation(product, query))
        for index, wording in enumerate(("not only cotton but linen", "not exclusively cotton")):
            with self.subTest(wording=wording):
                product = self.product(f"affirmed{index}", features=wording)
                self.assertEqual(CatalogSearch._hard_constraint_exactness(product, evidence), (1, 1))
                self.assertFalse(CatalogSearch._semantic_violation(product, query))

    def test_an_unknown_or_branch_does_not_become_a_contradiction(self) -> None:
        evidence = [Evidence("cotton or linen", 3.8, "hard_constraint", 1)]
        query = self.store.compile_query(evidence)
        unknown = self.product("unknown", features="No cotton")
        supported = self.product("supported", features="No cotton, linen")
        denied = self.product("denied", features="No cotton or linen")
        self.assertFalse(CatalogSearch._semantic_violation(unknown, query))
        self.assertFalse(CatalogSearch._semantic_violation(supported, query))
        self.assertEqual(CatalogSearch._hard_constraint_exactness(supported, evidence), (1, 1))
        self.assertTrue(CatalogSearch._semantic_violation(denied, query))

    def test_scoped_denial_does_not_leak_to_another_owner(self) -> None:
        product = self.product("scoped", description="The lining is not cotton and the upper is cotton")
        upper = self.store.compile_query([Evidence("upper: cotton", 3.8, "hard_constraint", 1)])
        lining = self.store.compile_query([Evidence("lining: cotton", 3.8, "hard_constraint", 1)])
        self.assertTrue(hard_evidence_match(product, "upper: cotton"))
        self.assertFalse(CatalogSearch._semantic_violation(product, upper))
        self.assertFalse(hard_evidence_match(product, "lining: cotton"))
        self.assertTrue(CatalogSearch._semantic_violation(product, lining))

    def test_component_extraction_retains_generic_values_and_three_owners(self) -> None:
        product = self.product("generic", details="pocket zippered lining cotton upper leather")
        self.assertTrue(hard_evidence_match(product, "pocket: zippered"))
        self.assertTrue(hard_evidence_match(product, "lining: cotton"))
        self.assertTrue(hard_evidence_match(product, "upper: leather"))
        natural = self.product("natural", description="The lining is cotton and the upper is leather and the sole is rubber")
        self.assertTrue(hard_evidence_match(natural, "lining: cotton"))
        self.assertTrue(hard_evidence_match(natural, "upper: leather"))
        self.assertTrue(hard_evidence_match(natural, "sole: rubber"))
        self.assertFalse(hard_evidence_match(natural, "upper: cotton"))

    def test_exclusive_facet_is_a_semantic_operator_with_bounded_scope(self) -> None:
        blue = self.product("blue", details="Color blue")
        mixed = self.product("mixed", details="Color red and blue")
        unknown = self.product("unknown", features="plain finish")
        for wording in ("blue only", "only blue", "color: blue only", "blue exclusively"):
            with self.subTest(wording=wording):
                self.assertEqual(exclusive_facet_values(wording, "color"), {"blue"})
                evidence = [Evidence(wording, 3.8, "hard_constraint", 1)]
                query = self.store.compile_query(evidence)
                self.assertEqual(CatalogSearch._hard_constraint_exactness(blue, evidence), (1, 1))
                self.assertEqual(CatalogSearch._hard_constraint_exactness(mixed, evidence), (0, 1))
                self.assertTrue(CatalogSearch._semantic_violation(mixed, query))
                self.assertFalse(CatalogSearch._semantic_violation(unknown, query))
                self.assertNotIn("only", query.evidence[0].tokens)
                self.assertEqual(evidence[0].text, wording)
        for wording in ('"blue only"', "blue only for work", "not only blue", "only the label says blue"):
            self.assertFalse(exclusive_facet_values(wording, "color"))
        self.assertFalse(exclusive_facet_values("colour: cotton only", "material"))

    def test_exclusive_or_keeps_one_group_and_the_whole_allowed_set(self) -> None:
        allowed = self.product("allowed", details="Color red and blue")
        forbidden = self.product("forbidden", details="Color green and blue")
        single = self.product("single", details="Color red")
        for wording in ("only red or blue", "red or blue only"):
            with self.subTest(wording=wording):
                query = self.store.compile_query([Evidence(wording, 3.8, "hard_constraint", 1)])
                for product in (allowed, single):
                    self.assertTrue(hard_evidence_match(product, wording))
                    self.assertFalse(CatalogSearch._semantic_violation(product, query))
                self.assertFalse(hard_evidence_match(forbidden, wording))
                self.assertTrue(CatalogSearch._semantic_violation(forbidden, query))

    def test_index_does_not_admit_a_denied_facet_as_an_exact_positive(self) -> None:
        index = ConstraintIndex()
        for identifier, wording in (("bad", "not cotton"), ("good", "cotton"), ("absence", "cotton-free")):
            index.add_product({"parent_asin": identifier, "features": [wording], "categories": ["shirts"]})
        self.assertEqual(index.exact_intersection("shirts", ["cotton"]), {"good"})
        self.assertEqual(index.exact_intersection("shirts", ["cotton free"]), {"absence"})

    def test_recursive_catalog_serialization_keeps_entry_boundaries_and_word_tokens(self) -> None:
        value = {"lining": ["No cotton", "Linen"],
                 "upper": {"material": "Leather", "color": "Blue"}}
        old_text = " ".join(f"{key} {item}" for key, item in value.items())
        serialized = _catalog_text(value)
        self.assertEqual(serialized, "lining No cotton; Linen; upper material Leather; color Blue")
        self.assertEqual(_text(value), serialized)
        self.assertEqual(terms(serialized), terms(old_text))

    def test_structured_negation_boundaries_match_live_and_persisted_indexes(self) -> None:
        variants = (
            {"features": ["No cotton", "Linen"]},
            {"features": ["No cotton", ["Linen", "Plain finish"]]},
            {"details": {"Material": "No cotton", "Fabric": "Linen"}},
            {"description": ["No cotton", {"Composition": "Linen"}]},
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, fields in enumerate(variants):
                with self.subTest(fields=fields):
                    path = Path(directory) / f"structured{index}.jsonl"
                    rows = [{"parent_asin": "A", "categories": ["shirts"], **fields},
                            {"parent_asin": "B", "categories": ["shirts"], "features": ["Plain fabric"]}]
                    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

                    def replay(expected_prebuilt: bool):
                        agent = make_agent(path, fullwidth=True)
                        try:
                            self.assertEqual(agent.search.using_prebuilt_index, expected_prebuilt)
                            agent.reset("s", {})
                            agent.respond("s", "I'm looking for shirts.", 1, 10)
                            result = agent.respond("s", "A key requirement is: linen.", 2, 10)
                            self.assertEqual(result["recommendations"][0]["parent_asin"], "A")
                            self.assertEqual(set(agent.diagnostics["stage_ids"]["raw_ranked"]), {"A", "B"})
                            return result
                        finally:
                            agent.close()

                    live = replay(False)
                    artifact = build_catalog_index(path)
                    self.assertEqual(replay(True), live)
                    # An unchanged catalog hash does not excuse stale flattened text.
                    with sqlite3.connect(artifact) as connection:
                        field = next(iter(fields))
                        connection.execute(f"UPDATE products SET {field}=? WHERE parent_asin='A'",
                                           ("No cotton Linen",))
                    stale_bytes = artifact.read_bytes()
                    self.assertIsNone(open_catalog_index(path, artifact))
                    self.assertEqual(replay(False), live)
                    self.assertEqual(artifact.read_bytes(), stale_bytes)

    def test_semantic_repairs_choose_valid_product_end_to_end(self) -> None:
        cases = (
            ("Shirts", {"features": ["No cotton"]}, {"features": ["Cotton"]}, "cotton", False),
            ("Jackets", {"description": "The lining is cotton and the upper is leather"},
             {"description": "The upper is cotton"}, "upper: cotton", False),
            ("Shirts", {"title": "Not red shirt", "features": ["Blue"]},
             {"title": "Green shirt", "features": ["Green"]}, "red or blue", True),
            ("Shoes", {"details": {"Size": "9"}}, {"details": {"Size": "8"}}, "size 8", False),
            ("Shirts", {"details": {"Color": "red and blue"}}, {"details": {"Color": "blue"}}, "blue only", False),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, (category, wrong, correct, requested, exclusion) in enumerate(cases):
                with self.subTest(requested=requested):
                    rows = [{"parent_asin": "A", "title": "Item", "categories": [category],
                             "rating_number": 1000000, **wrong},
                            {"parent_asin": "B", "title": "Item", "categories": [category], **correct}]
                    path = Path(directory) / f"{index}.jsonl"
                    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                    agent = make_agent(path, fullwidth=True)
                    try:
                        agent.reset("s", {})
                        agent.respond("s", f"I'm looking for {category}.", 1, 10)
                        message = f"No {requested}." if exclusion else f"A key requirement is: {requested}."
                        result = agent.respond("s", message, 2, 10)
                        self.assertEqual(result["recommendations"][0]["parent_asin"], "B")
                        self.assertEqual(set(agent.diagnostics["stage_ids"]["raw_ranked"]), {"A", "B"})
                    finally:
                        agent.close()


if __name__ == "__main__":
    unittest.main()
