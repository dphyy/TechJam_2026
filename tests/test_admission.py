import unittest

from mercury.admission import select_rerank_prefix
from mercury.types import Candidate, Preference, Product


def candidate(identifier: str, title: str, score: float) -> Candidate:
    return Candidate(Product(identifier, title, {"title": title}), score)


class RerankAdmissionTest(unittest.TestCase):
    def test_prefix_preserves_the_existing_ranking(self):
        candidates = [candidate(f"p{index}", "shirt", 100 - index) for index in range(12)]
        selected = select_rerank_prefix(candidates, [], 5, "prefix")
        self.assertEqual([item.product.parent_asin for item in selected], ["p0", "p1", "p2", "p3", "p4"])
        self.assertTrue(all(left is right for left, right in zip(selected, candidates)))

    def test_stratified_keeps_leaders_and_spreads_the_tail_without_duplicates(self):
        candidates = [candidate(f"p{index:02d}", "shirt", 100 - index) for index in range(30)]
        selected = select_rerank_prefix(candidates, [], 10, "stratified")
        identifiers = [item.product.parent_asin for item in selected]
        self.assertEqual(identifiers, ["p00", "p01", "p02", "p03", "p04", "p05", "p11", "p17", "p23", "p29"])
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_cover_admits_low_ranked_source_supported_rare_preference(self):
        candidates = [candidate(f"p{index:02d}", "cotton shirt", 100 - index) for index in range(39)]
        candidates.append(candidate("blue", "blue cotton shirt", 1.0))
        preferences = [
            Preference("material", "cotton", 1, "cotton", hard=True),
            Preference("color", "blue", 1, "blue"),
            Preference("color", "black", 1, "black", active=False),
        ]
        selected = select_rerank_prefix(candidates, preferences, 20, "cover")
        identifiers = [item.product.parent_asin for item in selected]
        self.assertEqual(identifiers[:10], [f"p{index:02d}" for index in range(10)])
        self.assertIn("blue", identifiers)
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual([item.product.parent_asin for item in candidates][-1], "blue")

    def test_rejects_unknown_modes_and_bad_limits(self):
        candidates = [candidate("one", "shirt", 1.0)]
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            select_rerank_prefix(candidates, [], 1, "oracle")
        with self.assertRaisesRegex(ValueError, "positive"):
            select_rerank_prefix(candidates, [], 0, "prefix")
