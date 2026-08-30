import unittest

from experiments.catalog_vocabulary_prepare import build, build_cases, normalize, variants


class CatalogVocabularyPrepareTest(unittest.TestCase):
    def test_normalization_and_inflection_are_conservative(self):
        self.assertEqual(normalize("Rain-Jackets!"), "rain jackets")
        self.assertIn("rain jacket", variants("rain jackets"))
        self.assertNotIn("cover upss", variants("cover ups"))

    def test_builder_requires_support_and_removes_cross_attribute_aliases(self):
        rows = [
            {
                "parent_asin": str(index),
                "title": "Trail Gaiters",
                "categories": ["Outdoor", "Trail Gaiters"],
                "details": {"Style": "Trail Gaiters"},
            }
            for index in range(5)
        ]
        payload = build(rows, "catalog")
        # The same normalized alias claims category and style, so conservative
        # cross-attribute ambiguity removes it entirely.
        self.assertNotIn("trail gaiters", {row["alias"] for row in payload["aliases"]})
        self.assertFalse(any(row["support"] < 5 for row in payload["aliases"]))

    def test_cases_exclude_static_subspans(self):
        aliases = [
            {"alias": f"novel objectword{index}", "attribute": "category",
             "canonical": f"novel objectword{index}", "support": 5,
             "confidence": 1.0, "method": "category_path"}
            for index in range(3)
        ]
        fixture = build_cases({"aliases": aliases}, count=2)
        self.assertEqual(len(fixture["cases"]), 2)


if __name__ == "__main__":
    unittest.main()
