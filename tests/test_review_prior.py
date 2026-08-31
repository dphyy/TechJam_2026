import unittest

from mercury.catalog import product_from_dict
from mercury.config import Config
from mercury.neural import fuse_neural_logits
from mercury.ranking import rank_constraints
from mercury.review_prior import ADJUSTMENT_KEY, rank_review_prior, review_signal
from mercury.types import Candidate, Preference


def candidate(identifier, count=0, stars=None, score=1.0, title="Shirt"):
    return Candidate(product_from_dict({"parent_asin": identifier, "title": title,
                                       "rating_number": count, "average_rating": stars}), score)


class ReviewPriorTest(unittest.TestCase):
    def test_safe_metadata_and_no_text_pollution(self):
        for value in (None, True, -1, 1.5, float("nan"), float("inf"), "unknown", [], {}):
            with self.subTest(value=value):
                product = candidate("a", value, value).product
                self.assertEqual(product.rating_number, 0)
        product = candidate("a", "1,234", "4.5").product
        self.assertEqual(product.rating_number, 1234)
        self.assertEqual(product.average_rating, 4.5)
        self.assertNotIn("1234", product.text)
        for value in (None, True, 0, -1, 6, "unknown", float("nan"), float("inf"), [], {}):
            self.assertIsNone(candidate("a", stars=value).product.average_rating)

    def test_extreme_counts_saturate_and_all_signals_are_bounded(self):
        for count in (0, 1, 20, 500000, 10**100, "9" * 5000):
            for stars in (None, 1, 3, 5):
                for mode in ("count", "raw_stars", "stars", "mixed"):
                    signal = review_signal(candidate("a", count, stars).product, mode)
                    self.assertLessEqual(abs(signal), 1.0)
        self.assertEqual(review_signal(candidate("a", 10**100).product, "count"), 1)

    def test_unknown_neutral_and_quality_shrinkage(self):
        for mode in ("count", "raw_stars", "stars", "mixed"):
            self.assertEqual(review_signal(candidate("a").product, mode), 0)
        self.assertLess(review_signal(candidate("a", 1, 5).product, "stars"),
                        review_signal(candidate("b", 1000, 4.5).product, "stars"))
        self.assertEqual(review_signal(candidate("a", 0, 5).product, "stars"), 0)
        self.assertEqual(review_signal(candidate("a", 0, 5).product, "raw_stars"), 1)

    def test_bounded_idempotent_and_disable_removes_bonus(self):
        pool = [candidate("a", 500000, 5), candidate("b", 5, 1)]
        first = rank_review_prior(pool, "mixed", .3)
        second = rank_review_prior(first, "mixed", .3)
        for one, two in zip(first, second):
            self.assertAlmostEqual(one.score, two.score)
            self.assertLessEqual(abs(one.route_scores.get(ADJUSTMENT_KEY, 0)), .3)
        removed = rank_review_prior(first, "none", 0)
        for item in removed:
            self.assertAlmostEqual(item.score, 1)
            self.assertNotIn(ADJUSTMENT_KEY, item.route_scores)
        self.assertEqual(pool[0].route_scores, {})

    def test_does_not_overturn_clear_relevance_gap(self):
        pool = [candidate("relevant", 0, score=1), candidate("popular", 500000, score=.5)]
        self.assertEqual(rank_review_prior(pool, "count", .3)[0].product.parent_asin, "relevant")

    def test_preserves_even_tiny_guarded_gap(self):
        clean = candidate("clean", 1000, 1, score=.5)
        bad = candidate("bad", 500000, 5, score=.4999)
        bad.route_scores["constraint_penalty"] = 1.0
        ranked = rank_review_prior([clean, bad], "stars", .3)
        self.assertEqual([item.product.parent_asin for item in ranked], ["clean", "bad"])
        self.assertGreater(ranked[0].score, ranked[1].score)
        repeated = rank_review_prior(ranked, "stars", .3)
        self.assertAlmostEqual(repeated[0].score, ranked[0].score)

    def test_neural_replacement_discards_admission_bonus_on_head_and_tail(self):
        pool = rank_review_prior([candidate("a", 500000), candidate("b", 1000)], "count", .3)
        fused = fuse_neural_logits(pool, {"a": 1}, .75)
        self.assertTrue(all(ADJUSTMENT_KEY not in item.route_scores for item in fused))
        post = rank_review_prior(fused, "count", .02)
        self.assertAlmostEqual(post[0].score - fused[0].score, .02)
        again = rank_review_prior(post, "count", .02)
        self.assertAlmostEqual(again[0].score, post[0].score)

    def test_second_constraint_check_repairs_neural_reintroduction(self):
        preference = Preference("material", "leather", 1, "No leather", hard=True, polarity=-1)
        pool = [candidate("bad", title="Leather shirt"), candidate("clean", title="Cotton shirt")]
        guarded = rank_constraints(pool, [preference])
        self.assertEqual(guarded[0].product.parent_asin, "clean")
        fused = fuse_neural_logits(guarded, {"bad": 10, "clean": 0}, .75)
        self.assertEqual(fused[0].product.parent_asin, "bad")
        self.assertEqual(rank_constraints(fused, [preference])[0].product.parent_asin, "clean")

    def test_config_bounds_and_stage_choices(self):
        for values in ({"review_prior_pre_weight": .31}, {"review_prior_post_weight": .021},
                       {"review_prior_pre_weight": float("nan")}, {"review_prior_post_weight": True},
                       {"review_prior_mode": "target"}, {"constraint_check_stage": "sometimes"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Config.from_dict(values)
        self.assertEqual(Config().constraint_check_stage, "both")
        self.assertEqual(Config().review_prior_mode, "none")


if __name__ == "__main__":
    unittest.main()
