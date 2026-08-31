import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from mercury.catalog import Catalog
from mercury.field_evidence import FieldEvidenceConfig, field_phrase_evidence
from mercury.retrieval import SparseIndex
from mercury.state import SessionState
from mercury.types import Preference


def preference(value="ratchet buckle", **kwargs):
    return Preference("other", value, 1, f"I need a {value}.", **kwargs)


class FieldEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "catalog.jsonl"
        self.index = None

    def tearDown(self):
        if self.index is not None:
            self.index.close()
        self.temp.cleanup()

    def catalog(self, rows):
        rows = [{"parent_asin": f"base-{i}", "title": "Everyday item",
                 "features": ["standard closure"]} for i in range(12)] + rows
        self.path.write_text("\n".join(json.dumps(row) for row in rows))
        catalog = Catalog(self.path)
        self.index = SparseIndex(catalog)
        return catalog

    def run_arm(self, rows, preferences=None, base=None, **kwargs):
        catalog = self.catalog(rows)
        return field_phrase_evidence(catalog, self.index, base or ["base-0"],
                                     preferences or [preference()], **kwargs)

    def test_admission_recovers_rare_phrase_without_displacing_base(self):
        result = self.run_arm([{"parent_asin": "rare", "features": ["Ratchet buckle"]}],
                              base=["base-1", "base-0"])
        self.assertEqual(result.candidate_ids, ["base-1", "base-0", "rare"])
        self.assertEqual(result.admitted_ids, ["rare"])
        self.assertEqual(result.score_deltas, {})
        witness = result.witnesses["rare"]
        self.assertEqual((witness.source, witness.span, witness.document_frequency),
                         ("features", "Ratchet buckle", 1))

    def test_scoring_only_preserves_membership_order_and_has_bounded_deltas(self):
        result = self.run_arm([{"parent_asin": "rare", "details": {"Closure": "ratchet buckle"}}],
                              base=["base-0", "rare"], arm="scoring_only")
        self.assertEqual(result.candidate_ids, ["base-0", "rare"])
        self.assertEqual(result.admitted_ids, [])
        self.assertEqual(set(result.score_deltas), {"rare"})
        self.assertGreater(result.score_deltas["rare"], 0)
        self.assertLessEqual(result.score_deltas["rare"], FieldEvidenceConfig().score_cap)
        self.assertTrue(math.isfinite(result.score_deltas["rare"]))

    def test_scoring_only_never_adds_missing_matches(self):
        result = self.run_arm([{"parent_asin": "rare", "title": "Ratchet buckle"}],
                              arm="scoring_only")
        self.assertEqual(result.candidate_ids, ["base-0"])
        self.assertEqual(result.score_deltas, {})

    def test_off_does_not_query_index(self):
        catalog = self.catalog([])
        with patch.object(self.index, "search_phrase") as search:
            result = field_phrase_evidence(catalog, self.index, ["base-0"], [preference()], arm="off")
        search.assert_not_called()
        self.assertEqual(result.candidate_ids, ["base-0"])

    def test_negative_neutral_inactive_and_low_confidence_do_not_query(self):
        catalog = self.catalog([{"parent_asin": "rare", "features": ["ratchet buckle"]}])
        for item in (preference(polarity=-1), preference(polarity=0), preference(active=False),
                     preference(confidence=0.2)):
            with self.subTest(item=item), patch.object(self.index, "search_phrase") as search:
                result = field_phrase_evidence(catalog, self.index, ["base-0"], [item])
                search.assert_not_called()
                self.assertEqual(result.admitted_ids, [])

    def test_quoted_negation_and_uncertainty_do_not_become_positive_evidence(self):
        catalog = self.catalog([{"parent_asin": "rare", "features": ["ratchet buckle"]}])
        for source in ('I do not want "ratchet buckle".', "Don't include 'ratchet buckle'.",
                       "Maybe a ratchet buckle.", "Ratchet buckle is not necessary.",
                       "I could use a ratchet buckle.", "This may need a ratchet buckle.",
                       "Snap closure instead of ratchet buckle."):
            with self.subTest(source=source):
                item = preference()
                item.source_text = source
                result = field_phrase_evidence(catalog, self.index, ["base-0"], [item])
                self.assertEqual(result.admitted_ids, [])
                self.assertEqual(result.diagnostics["queries"], 0)

    def test_independent_positive_clause_can_follow_negation(self):
        item = preference()
        item.source_text = "Not a snap closure, but I want a ratchet buckle."
        result = self.run_arm([{"parent_asin": "rare", "features": ["ratchet buckle"]}], [item])
        self.assertEqual(result.admitted_ids, ["rare"])

    def test_negation_inside_a_malformed_positive_value_does_not_query(self):
        catalog = self.catalog([])
        for value in ("not ratchet buckle", "maybe ratchet buckle"):
            with self.subTest(value=value), patch.object(self.index, "search_phrase") as search:
                result = field_phrase_evidence(catalog, self.index, ["base-0"], [preference(value)])
            search.assert_not_called()
            self.assertEqual(result.admitted_ids, [])

    def test_correction_uses_only_active_values_not_old_raw_source(self):
        catalog = self.catalog([
            {"parent_asin": "old", "features": ["ratchet buckle"]},
            {"parent_asin": "new", "features": ["snap closure"]},
        ])
        state = SessionState({})
        state.update("I need a ratchet buckle.", 1)
        state.update("Actually, no ratchet buckle; snap closure instead.", 2)
        result = field_phrase_evidence(catalog, self.index, ["base-0"], state.preferences)
        self.assertNotIn("old", result.admitted_ids)
        self.assertIn("new", result.admitted_ids)

    def test_common_multiword_and_single_word_preferences_have_no_boost(self):
        result = self.run_arm([], [preference("standard closure"), preference("cotton")],
                              arm="admission_and_scoring")
        self.assertEqual(result.admitted_ids, [])
        self.assertEqual(result.score_deltas, {})
        self.assertEqual(result.diagnostics["queries"], 1)
        self.assertEqual(result.diagnostics["common_phrases"], 1)
        self.assertLessEqual(result.diagnostics["posting_rows"], 2)

    def test_phrase_order_and_token_boundaries_are_required(self):
        result = self.run_arm([
            {"parent_asin": "reversed", "title": "Buckle ratchet"},
            {"parent_asin": "suffix", "title": "ratchet buckles"},
            {"parent_asin": "substring", "title": "ratchet buckleboard"},
        ], config=FieldEvidenceConfig(max_document_fraction=0.5))
        self.assertEqual(result.admitted_ids, [])

    def test_phrase_cannot_cross_fields_or_sentences(self):
        result = self.run_arm([
            {"parent_asin": "fields", "title": "ratchet", "description": ["buckle"]},
            {"parent_asin": "sentences", "features": ["ratchet. Buckle"]},
        ], config=FieldEvidenceConfig(max_document_fraction=0.5))
        self.assertEqual(result.admitted_ids, [])

    def test_negated_and_conflicting_raw_catalog_claims_are_neutral(self):
        result = self.run_arm([
            {"parent_asin": "negative", "features": ['not "ratchet buckle"']},
            {"parent_asin": "conflict", "title": "Ratchet buckle",
             "description": ["No ratchet buckle"]},
            {"parent_asin": "uncertain", "features": ["May vary. Maybe ratchet buckle"]},
        ], config=FieldEvidenceConfig(max_document_fraction=0.5), arm="admission_and_scoring")
        self.assertEqual(result.admitted_ids, [])
        self.assertEqual(result.score_deltas, {})

    def test_or_members_are_independent_and_matching_both_does_not_double_score(self):
        rows = [{"parent_asin": "a", "features": ["ratchet buckle"]},
                {"parent_asin": "b", "features": ["snap closure"]},
                {"parent_asin": "both", "features": ["ratchet buckle; snap closure"]}]
        choices = [preference("ratchet buckle", alternative_group="choice"),
                   preference("snap closure", alternative_group="choice")]
        result = self.run_arm(rows, choices, arm="admission_and_scoring",
                              config=FieldEvidenceConfig(max_document_fraction=0.5))
        self.assertEqual(set(result.admitted_ids), {"a", "b", "both"})
        self.assertAlmostEqual(result.score_deltas["a"], result.score_deltas["both"])
        self.assertAlmostEqual(result.score_deltas["b"], result.score_deltas["both"])

    def test_hard_or_does_not_treat_the_other_member_as_a_conjunction(self):
        choices = [Preference("material", value, 1, "Either cotton or silk.", hard=True,
                              alternative_group="choice") for value in ("cotton", "silk")]
        result = self.run_arm([{"parent_asin": "rare", "title": "Silk item",
                               "features": ["ratchet buckle", "not cotton"]}],
                              [*choices, preference()])
        self.assertEqual(result.admitted_ids, ["rare"])

    def test_supported_exclusion_blocks_phrase_support_without_removing_base(self):
        avoided = Preference("material", "wool", 1, "No wool.", polarity=-1)
        result = self.run_arm([{"parent_asin": "rare", "title": "Wool item",
                               "features": ["ratchet buckle"]}], [avoided, preference()],
                              base=["base-0", "rare"], arm="admission_and_scoring")
        self.assertEqual(result.candidate_ids, ["base-0", "rare"])
        self.assertEqual(result.score_deltas, {})
        self.assertGreater(result.diagnostics["guarded_products"], 0)

    def test_low_confidence_exclusions_and_hard_constraints_still_guard(self):
        catalog = self.catalog([{"parent_asin": "rare", "title": "Wool item",
                                 "features": ["ratchet buckle", "not cotton"]}])
        constraints = [Preference("material", "wool", 1, "No wool.",
                                  polarity=-1, confidence=0.5),
                       Preference("material", "cotton", 1, "Must be cotton.",
                                  hard=True, confidence=0.5)]
        for constraint in constraints:
            with self.subTest(constraint=constraint):
                result = field_phrase_evidence(catalog, self.index, ["base-0", "rare"],
                                               [constraint, preference()], arm="admission_and_scoring")
                self.assertEqual(result.candidate_ids, ["base-0", "rare"])
                self.assertEqual(result.score_deltas, {})
                self.assertEqual(result.diagnostics["guarded_products"], 1)

    def test_unknown_hard_constraint_metadata_remains_neutral(self):
        hard = Preference("material", "leather", 1, "Must be leather.", hard=True)
        result = self.run_arm([{"parent_asin": "rare", "features": ["ratchet buckle"]}],
                              [hard, preference()])
        self.assertEqual(result.admitted_ids, ["rare"])

    def test_scope_requires_direct_ownership_in_same_raw_field(self):
        item = preference("cotton blend", scope="lining")
        item.source_text = "I need cotton blend for the lining."
        result = self.run_arm([
            {"parent_asin": "match", "details": {"Lining": "cotton blend"}},
            {"parent_asin": "wrong", "details": {"Body": "cotton blend", "Lining": "silk"}},
            {"parent_asin": "unknown", "title": "Cotton blend item"},
        ], [item], config=FieldEvidenceConfig(max_document_fraction=0.5))
        self.assertEqual(result.admitted_ids, ["match"])
        self.assertEqual(result.witnesses["match"].scope, "lining")

    def test_scoped_guard_requires_observed_ownership_not_another_component(self):
        catalog = self.catalog([
            {"parent_asin": "unknown", "features": ["ratchet buckle"],
             "details": {"Body": "cotton", "Lining": "silk"}},
            {"parent_asin": "contradiction", "features": ["ratchet buckle"],
             "details": {"Lining": "not cotton"}},
        ])
        constraint = Preference("material", "cotton", 1, "Cotton for the lining.",
                                hard=True, scope="lining")
        result = field_phrase_evidence(catalog, self.index, ["base-0"], [constraint, preference()],
                                       config=FieldEvidenceConfig(max_document_fraction=0.5))
        self.assertEqual(result.admitted_ids, ["unknown"])

    def test_scoped_negative_does_not_exclude_body_evidence_for_lining(self):
        avoided = Preference("material", "cotton", 1, "No cotton for the lining.",
                             polarity=-1, scope="lining")
        result = self.run_arm([{"parent_asin": "rare", "features": ["ratchet buckle"],
                               "details": {"Body": "cotton", "Lining": "silk"}}],
                              [avoided, preference()])
        self.assertEqual(result.admitted_ids, ["rare"])

    def test_qualified_phrase_requires_active_positive_owner(self):
        catalog = self.catalog([{"parent_asin": "rare", "features": ["80% cotton"]}])
        qualified = preference("80% cotton", depends_on=("material", "cotton"))
        result = field_phrase_evidence(catalog, self.index, ["base-0"], [qualified])
        self.assertEqual(result.admitted_ids, [])
        owner = Preference("material", "cotton", 1, "Cotton.", confidence=0.5)
        result = field_phrase_evidence(catalog, self.index, ["base-0"], [qualified, owner])
        self.assertEqual(result.admitted_ids, ["rare"])

    def test_numeric_punctuation_cannot_be_erased_to_manufacture_support(self):
        owner = Preference("material", "cotton", 1, "Cotton.")
        result = self.run_arm([
            {"parent_asin": "plain", "features": ["80 cotton"]},
            {"parent_asin": "different", "features": ["80/ cotton"]},
            {"parent_asin": "percent", "features": ["80% cotton"]},
        ], [owner, preference("80% cotton", depends_on=("material", "cotton"))],
            config=FieldEvidenceConfig(max_document_fraction=0.5))
        self.assertEqual(result.admitted_ids, ["percent"])

    def test_work_and_admission_limits_are_respected(self):
        rows = [{"parent_asin": f"rare-{i}", "features": [f"special clasp {i}"]} for i in range(6)]
        prefs = [preference(f"special clasp {i}") for i in range(6)]
        limits = FieldEvidenceConfig(max_phrases=3, max_postings=2, max_admissions=1)
        result = self.run_arm(rows, prefs, config=limits)
        self.assertEqual(result.diagnostics["queries"], 3)
        self.assertLessEqual(result.diagnostics["posting_rows"], 9)
        self.assertEqual(len(result.admitted_ids), 1)
        self.assertEqual(result.candidate_ids[0], "base-0")

    def test_duplicate_preferences_and_fields_do_not_accumulate(self):
        result = self.run_arm([{"parent_asin": "rare", "title": "Ratchet buckle",
                               "features": ["ratchet buckle"], "description": ["ratchet buckle"]}],
                              [preference(), preference()], arm="admission_and_scoring")
        self.assertEqual(result.diagnostics["queries"], 1)
        self.assertLessEqual(result.score_deltas["rare"], FieldEvidenceConfig().score_cap)

    def test_no_per_turn_catalog_iteration(self):
        class IndexOnlyList(list):
            def __iter__(self):
                raise AssertionError("per-turn catalog scan")

        catalog = self.catalog([{"parent_asin": "rare", "title": "Ratchet buckle"}])
        catalog.products = IndexOnlyList(catalog.products)
        result = field_phrase_evidence(catalog, self.index, ["base-0"], [preference()])
        self.assertEqual(result.admitted_ids, ["rare"])

    def test_invalid_configuration_and_arm_fail_explicitly(self):
        for kwargs in ({"max_phrases": 0}, {"score_cap": float("nan")},
                       {"max_document_fraction": 0}, {"minimum_confidence": 1.1},
                       {"max_postings": 257}, {"max_phrases": 33},
                       {"max_admissions": 129}, {"max_phrase_tokens": 33},
                       {"max_field_characters": 64_001}, {"score_cap": 1.1},
                       {"max_postings": True}, {"max_postings": float("inf")},
                       {"max_document_fraction": True}, {"minimum_confidence": True},
                       {"score_cap": True}, {"score_cap": "0.08"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FieldEvidenceConfig(**kwargs)
        catalog = self.catalog([])
        with self.assertRaises(ValueError):
            field_phrase_evidence(catalog, self.index, [], [], arm="invalid")

    def test_field_prefix_bound_does_not_invent_later_support(self):
        result = self.run_arm([{"parent_asin": "rare", "features": ["x " * 100 + "ratchet buckle"]}],
                              config=replace(FieldEvidenceConfig(), max_field_characters=40))
        self.assertEqual(result.admitted_ids, [])

    def test_oversized_field_does_not_hide_a_later_contradiction(self):
        result = self.run_arm([{"parent_asin": "rare", "title": "Ratchet buckle",
                               "description": ["x " * 100 + "No ratchet buckle"]}],
                              config=replace(FieldEvidenceConfig(), max_field_characters=40))
        self.assertEqual(result.admitted_ids, [])

    def test_oversized_preferences_return_base_without_querying(self):
        catalog = self.catalog([])
        with patch.object(self.index, "search_phrase") as search:
            result = field_phrase_evidence(catalog, self.index, ["base-0"], [preference()] * 129)
        search.assert_not_called()
        self.assertEqual(result.candidate_ids, ["base-0"])
        self.assertEqual(result.diagnostics["resource_skips"], 1)


if __name__ == "__main__":
    unittest.main()
