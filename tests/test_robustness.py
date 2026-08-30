"""Synthetic natural-language checks, separate from official benchmark scores."""

import json
import tempfile
import unittest
from pathlib import Path

from mercury.agent import Agent
from mercury.config import Config


class ShoppingRobustnessTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.catalog = Path(self.temp.name) / "catalog.jsonl"
        rows = [
            {"parent_asin": "A", "title": "Black leather bag", "price": 90, "categories": ["Bags"]},
            {"parent_asin": "B", "title": "Blue canvas bag with adjustable strap", "price": 35, "categories": ["Bags"]},
            {"parent_asin": "C", "title": "Blue faux leather bag", "price": 30, "details": {"Material": "Faux leather"}, "categories": ["Bags"]},
            {"parent_asin": "D", "title": "Black cotton shirt", "price": 20, "categories": ["Shirts"]},
            {"parent_asin": "E", "title": "Blue canvas bag", "categories": ["Bags"]},
            {"parent_asin": "F", "title": "Blue leather-free bag", "details": {"Material": "Leather-free"}, "categories": ["Bags"]},
        ]
        self.catalog.write_text("\n".join(map(json.dumps, rows)))
        self.agent = Agent(self.catalog, Config(evidence_ranking=False))
        self.agent.reset("shopper", {})

    def tearDown(self):
        self.agent.close()
        self.temp.cleanup()

    def ids(self, message, turn=1):
        return [row["parent_asin"] for row in self.agent.respond("shopper", message, turn, 10)["recommendations"]]

    def test_excluded_material_is_deprioritized_with_optional_evidence_off(self):
        identifiers = self.ids("I need a bag. No leather please.")
        self.assertLess(identifiers.index("B"), identifiers.index("A"))
        self.assertLess(identifiers.index("C"), identifiers.index("A"))
        self.assertLess(identifiers.index("F"), identifiers.index("A"))

    def test_unknown_price_is_not_treated_as_over_budget(self):
        identifiers = self.ids("A bag under $40 please.")
        self.assertIn("E", identifiers)
        self.assertLess(identifiers.index("E"), identifiers.index("A"))

    def test_correction_retracts_only_changed_preferences(self):
        self.ids("A black leather bag with an adjustable strap.")
        self.ids("Actually, blue canvas instead.", 2)
        query = self.agent.last_diagnostics["query"]
        for value in ("blue", "canvas", "bags", "adjustable"):
            self.assertIn(value, query)
        for value in ("black", "leather"):
            self.assertNotIn(value, query)

    def test_same_message_correction_keeps_final_color_only(self):
        self.ids("A red shirt. On second thought, black.")
        query = self.agent.last_diagnostics["query"]
        self.assertIn("black", query)
        self.assertNotIn("red", query)

    def test_current_request_outranks_profile(self):
        self.agent.reset("shopper", {"preferred_color": "blue", "preferred_category": "Bags"})
        self.assertEqual(self.ids("A black cotton shirt.")[0], "D")

    def test_no_new_information_does_not_change_recommendations(self):
        first = self.ids("A blue canvas bag.")
        second = self.ids("I have no additional preferences to share.", 2)
        self.assertEqual(first, second)
        self.assertTrue(self.agent.last_diagnostics["cache_hit"])

    def test_budget_update_invalidates_cached_ranking(self):
        self.ids("A leather bag.")
        identifiers = self.ids("My budget is under $40.", 2)
        self.assertFalse(self.agent.last_diagnostics["cache_hit"])
        self.assertIn("A", identifiers)
        self.assertGreater(self.agent.last_diagnostics["price_adjustments"]["B"], 0.0)
        self.assertLess(self.agent.last_diagnostics["price_adjustments"]["A"], 0.0)

    def test_correction_paraphrases_have_the_same_active_material(self):
        for phrase in ("Actually canvas instead.", "Switch to canvas.", "On second thought, canvas."):
            with self.subTest(phrase=phrase):
                self.agent.reset("shopper", {})
                self.ids("A leather bag.")
                self.ids(phrase, 2)
                query = self.agent.last_diagnostics["query"]
                self.assertIn("canvas", query)
                self.assertNotIn("leather", query)


if __name__ == "__main__":
    unittest.main()
