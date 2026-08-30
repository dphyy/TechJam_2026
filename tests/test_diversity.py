import unittest

from mercury.catalog import product_from_dict
from mercury.diversity import diversify_candidates, facet_signature
from mercury.types import Candidate


class DiversityTest(unittest.TestCase):
    def candidates(self):
        rows = [
            {"parent_asin": "leader", "title": "Blue cotton shirt"},
            {"parent_asin": "similar", "title": "Red cotton shirt"},
            {"parent_asin": "different", "title": "Blue polyester boots"},
            {"parent_asin": "unknown", "title": "Product"},
        ]
        return [Candidate(product_from_dict(row), 10.0 - index)
                for index, row in enumerate(rows)]

    def test_facet_signature_is_typed_and_deterministic(self):
        signature = facet_signature(self.candidates()[0].product)
        self.assertIn(("material", "cotton"), signature)
        self.assertIn(("color", "blue"), signature)
        self.assertIn(("category", "boots"), facet_signature(self.candidates()[2].product))

    def test_diversity_anchors_leader_and_can_promote_a_different_option(self):
        candidates = self.candidates()
        diversified = diversify_candidates(candidates, .9)
        self.assertEqual(diversified[0], candidates[0])
        self.assertEqual(diversified[1].product.parent_asin, "different")
        self.assertEqual({item.product.parent_asin for item in diversified},
                         {item.product.parent_asin for item in candidates})
        self.assertEqual([item.score for item in diversified if item.product.parent_asin == "different"], [8.0])

    def test_zero_strength_and_missing_facets_do_not_invent_novelty(self):
        candidates = self.candidates()
        self.assertEqual(diversify_candidates(candidates, 0), candidates)
        diversified = diversify_candidates(candidates, .5)
        self.assertGreater(diversified.index(candidates[-1]), 0)


if __name__ == "__main__":
    unittest.main()
