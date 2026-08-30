from __future__ import annotations

import unittest

from experiments.compare_paging import paging_behavior


def turn(recommendations: list[str], ranked: list[str], page: int = 0,
         reason: str = "base_head", override: bool = False) -> dict:
    return {
        "response": {"recommendations": [{"parent_asin": item} for item in recommendations]},
        "diagnostics": {
            "ranked_ids": ranked,
            "slate_page": page,
            "slate_selection": {"reason": reason, "override_reset": override},
        },
    }


class PagingComparisonTest(unittest.TestCase):
    def test_counts_duplicate_and_unseen_stable_head_choices(self):
        ranked = [f"P{index:02d}" for index in range(20)]
        head, tail = ranked[:10], ranked[10:]
        metrics = paging_behavior([[
            turn(head, ranked),
            turn(head, ranked),
            turn(tail, ranked, 1, "highest_ranked_unseen"),
        ]])
        self.assertEqual(metrics["eligible_stable_head_turns"], 2)
        self.assertEqual(metrics["exact_adjacent_duplicate_slates"], 1)
        self.assertEqual(metrics["highest_ranked_unseen_selections"], 1)
        self.assertEqual(metrics["mean_unique_products_through_turn"]["3"], 20)

    def test_override_is_a_reset_not_a_stable_head_opportunity(self):
        ranked = [f"P{index:02d}" for index in range(10)]
        metrics = paging_behavior([[
            turn(ranked, ranked),
            turn(ranked, ranked, override=True),
        ]])
        self.assertEqual(metrics["eligible_stable_head_turns"], 0)
        self.assertEqual(metrics["exact_adjacent_duplicate_slates"], 0)
        self.assertEqual(metrics["override_resets"], 1)


if __name__ == "__main__":
    unittest.main()
