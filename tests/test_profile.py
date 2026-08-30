import unittest

from mercury.catalog import product_from_dict
from mercury.profile import distill_profile, rank_profile_prior
from mercury.types import Candidate


class ProfileTest(unittest.TestCase):
    def test_distillation_is_bounded_deduplicated_and_malformed_safe(self):
        profile = {"preference_tags": ["Comfort", "comfort", 4, "x " * 30, "durability"]}
        self.assertEqual(distill_profile(profile), ("comfort", "durability"))
        for malformed in (None, [], {"preference_tags": "comfort"}, {"preference_tags": [None]}):
            with self.subTest(malformed=malformed):
                self.assertEqual(distill_profile(malformed), ())

    def test_profile_is_only_a_small_deterministic_prior(self):
        comfort = Candidate(product_from_dict({"parent_asin": "a", "title": "Comfort walking shoes"}), .99)
        relevant = Candidate(product_from_dict({"parent_asin": "b", "title": "Running shoes"}), 1.0)
        ranked = rank_profile_prior([relevant, comfort], ("comfort",), .005)
        self.assertEqual([item.product.parent_asin for item in ranked], ["b", "a"])
        tied = rank_profile_prior([Candidate(relevant.product, 1), Candidate(comfort.product, 1)],
                                  ("comfort",), .005)
        self.assertEqual(tied[0].product.parent_asin, "a")
        self.assertEqual(tied, rank_profile_prior([Candidate(relevant.product, 1),
                                                   Candidate(comfort.product, 1)], ("comfort",), .005))


if __name__ == "__main__":
    unittest.main()
