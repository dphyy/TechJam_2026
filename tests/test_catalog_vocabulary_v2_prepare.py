import unittest

from experiments.catalog_vocabulary_v2_prepare import build, build_cases


class CatalogVocabularyV2PrepareTest(unittest.TestCase):
    def test_builder_records_ambiguity_margin_and_suppresses_ambiguous_claims(self):
        rows = []
        for index in range(10):
            rows.append({
                "parent_asin": str(index), "title": "Trail Gaiters",
                "categories": ["Outdoor", "Trail Gaiters"],
                "details": {"Style": "Trail Gaiters" if index < 5 else "Plain"},
            })
        payload = build(rows, "catalog")
        aliases = {row["alias"]: row for row in payload["aliases"]}
        self.assertNotIn("trail gaiters", aliases)
        self.assertGreaterEqual(aliases["outdoor"]["ambiguity_margin"], 0.2)
        self.assertEqual(aliases["outdoor"]["role"], "object")

    def test_frozen_suite_balances_state_retrieval_and_adversarial_lanes(self):
        aliases = []
        for index in range(6):
            aliases.append({
                "alias": f"novel objectword{index}", "attribute": "category",
                "canonical": f"novel objectword{index}", "support": 8,
                "confidence": 1.0, "ambiguity_margin": 1.0,
                "state_eligible": index < 3, "role": "object", "method": "category_path",
            })
        fixture = build_cases({"aliases": aliases}, state_count=2, retrieval_count=2,
                              adversarial_count=4)
        kinds = [row["kind"] for row in fixture["cases"]]
        self.assertEqual(kinds.count("state_positive"), 2)
        self.assertEqual(kinds.count("retrieval_positive"), 2)
        self.assertEqual(len([kind for kind in kinds if kind in {"negated_alias", "ordinary_context"}]), 4)


if __name__ == "__main__":
    unittest.main()
