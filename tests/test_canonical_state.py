import json
import unittest
from dataclasses import replace
from pathlib import Path

from experiments.metamorphic_validate import evaluate_pack
from mercury.config import Config
from mercury.state import SessionState


ROOT = Path(__file__).resolve().parents[1]


class CanonicalStateTest(unittest.TestCase):
    def state(self) -> SessionState:
        return SessionState({}, alternatives_mode="grouped", canonical_state_semantics=True)

    def test_query_and_alternative_identity_do_not_depend_on_mention_order(self):
        left, right = self.state(), self.state()
        left.update("A cotton or linen summer shirt", 1)
        right.update("For summer, a linen or cotton shirt", 1)
        self.assertEqual(left.query(), right.query())
        self.assertEqual(left.semantic_signature(), right.semantic_signature())
        left_groups = {item.alternative_group for item in left.active_preferences()
                       if item.alternative_group}
        right_groups = {item.alternative_group for item in right.active_preferences()
                        if item.alternative_group}
        self.assertEqual(left_groups, right_groups)

    def test_correction_crosses_punctuation_but_preserves_unrelated_facts(self):
        state = self.state()
        state.update("I want a black leather travel bag with an adjustable strap", 1)
        state.update("Correction: make that blue, made of canvas", 2)
        facts = {(item.attribute, item.value) for item in state.active_preferences()
                 if item.polarity == 1}
        self.assertIn(("color", "blue"), facts)
        self.assertIn(("material", "canvas"), facts)
        self.assertIn(("category", "bags"), facts)
        self.assertIn(("use_case", "travel"), facts)
        self.assertIn(("feature", "adjustable"), facts)
        self.assertNotIn(("color", "black"), facts)
        self.assertNotIn(("material", "leather"), facts)
        self.assertNotIn(("other", "correction"), facts)

    def test_no_longer_preference_retires_only_the_named_attribute(self):
        state = self.state()
        state.update("I need a blue cotton shirt", 1)
        state.update("I no longer have a color preference", 2)
        facts = {(item.attribute, item.value, item.polarity) for item in state.active_preferences()}
        self.assertIn(("color", "any", 0), facts)
        self.assertIn(("material", "cotton", 1), facts)
        self.assertIn(("category", "shirts", 1), facts)
        self.assertNotIn(("color", "blue", 1), facts)

    def test_v2_semantic_fixture_passes_with_candidate(self):
        pack = json.loads((ROOT / "data/metamorphic_robustness_v2.json").read_text(encoding="utf-8"))
        config = replace(Config.load(ROOT / "configs/canonical_state_semantics.json"), neural_rerank=False)
        result = evaluate_pack(pack, config)
        self.assertEqual(result["failed_cases"], 0)
        self.assertTrue(all(
            case["invariance"][metric] == 1.0
            for case in result["cases"]
            for metric in ("minimum_top120_jaccard", "minimum_top10_overlap", "minimum_rank_correlation")
        ))


if __name__ == "__main__":
    unittest.main()
