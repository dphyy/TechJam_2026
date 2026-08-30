import unittest

from mercury.catalog import product_from_dict
from mercury.ranking import (budget_preference_score, evidence_score, preference_evidence,
                             rank_candidates, rank_constraints, rank_product_compatibility,
                             rank_soft_negatives, rank_soft_prices, value_matches)
from mercury.types import Candidate, Preference


def preference(attribute, value, polarity=1, hard=False, alternative_group=None):
    return Preference(attribute, value, 1, value, polarity=polarity, hard=hard,
                      alternative_group=alternative_group)


class RankingTest(unittest.TestCase):
    def test_missing_price_is_unknown_not_zero_or_contradiction(self):
        product = product_from_dict({"parent_asin": "a", "title": "Cotton shirt"})
        signal = preference_evidence(product, preference("budget", "<= 50", hard=True))
        self.assertEqual(signal, 0.0)

    def test_lower_bound_price_cannot_prove_affordability(self):
        low = product_from_dict({"parent_asin": "a", "price": "from $20"})
        high = product_from_dict({"parent_asin": "b", "price": "from $80"})
        limit = preference("budget", "<= 50", hard=True)
        self.assertEqual(preference_evidence(low, limit), 0.0)
        self.assertLess(preference_evidence(high, limit), 0.0)

    def test_soft_price_ranking_boosts_fit_without_excluding_uncertain_prices(self):
        candidates = [
            Candidate(product_from_dict({"parent_asin": "expensive", "price": 80}), 1.0),
            Candidate(product_from_dict({"parent_asin": "unknown"}), 1.0),
            Candidate(product_from_dict({"parent_asin": "from-low", "price": "from $20"}), 1.0),
            Candidate(product_from_dict({"parent_asin": "fit", "price": 40}), 1.0),
        ]
        ranked = rank_soft_prices(candidates, [preference("budget", "<= 50")], .02)
        self.assertEqual([item.product.parent_asin for item in ranked],
                         ["fit", "from-low", "unknown", "expensive"])
        self.assertEqual({item.product.parent_asin for item in ranked},
                         {item.product.parent_asin for item in candidates})
        self.assertNotIn("price_preference", ranked[1].route_scores)
        self.assertNotIn("price_preference", ranked[2].route_scores)

    def test_approximate_budget_uses_continuous_proximity_and_is_idempotent(self):
        budget = preference("budget", "<= 100", hard=False)
        candidates = [
            Candidate(product_from_dict({"parent_asin": "far-over", "price": 200}), 1.0),
            Candidate(product_from_dict({"parent_asin": "under", "price": 75}), 1.0),
            Candidate(product_from_dict({"parent_asin": "unknown"}), 1.0),
            Candidate(product_from_dict({"parent_asin": "target", "price": 100}), 1.0),
        ]
        self.assertEqual(budget_preference_score(candidates[3].product, budget), 1.0)
        self.assertEqual(budget_preference_score(candidates[1].product, budget), 0.5)
        self.assertEqual(budget_preference_score(candidates[0].product, budget), -1.0)
        ranked = rank_soft_prices(candidates, [budget], .02)
        self.assertEqual([item.product.parent_asin for item in ranked],
                         ["target", "under", "unknown", "far-over"])
        self.assertEqual(rank_soft_prices(ranked, [budget], .02), ranked)

    def test_soft_negative_demotes_without_becoming_a_constraint(self):
        candidates = [
            Candidate(product_from_dict({"parent_asin": "avoided", "title": "Leather bag"}), 1.0),
            Candidate(product_from_dict({"parent_asin": "clean", "title": "Cotton bag"}), 1.0),
            Candidate(product_from_dict({"parent_asin": "unknown", "title": "Travel bag"}), 1.0),
        ]
        avoided = preference("material", "leather", polarity=-1, hard=False)
        self.assertEqual(rank_constraints(candidates, [avoided]), candidates)
        ranked = rank_soft_negatives(candidates, [avoided], .02)
        self.assertEqual([item.product.parent_asin for item in ranked],
                         ["clean", "unknown", "avoided"])
        self.assertEqual(rank_soft_negatives(ranked, [avoided], .02), ranked)

    def test_negated_material_is_not_positive_support(self):
        product = product_from_dict({"parent_asin": "a", "title": "Faux leather bag"})
        self.assertLess(preference_evidence(product, preference("material", "leather")), 0.0)

    def test_negated_field_matches_contradict_requests_and_support_exclusions(self):
        for title, value in (("Not waterproof jacket", "waterproof"),
                             ("Jacket without pockets", "pockets")):
            for source in ("title", "categories", "features", "details", "description", "store"):
                product = product_from_dict({"parent_asin": "a", source: title})
                for polarity in (1, -1):
                    with self.subTest(source=source, value=value, polarity=polarity):
                        self.assertLess(preference_evidence(
                            product, preference("feature", value, polarity)) * polarity, 0.0)

    def test_structured_qualified_material_does_not_support_plain_material(self):
        for material in ("Faux leather", "Leather-free", "Imitation leather", "Synthetic leather"):
            product = product_from_dict({"parent_asin": "a", "details": {"Material": material}})
            for polarity in (1, -1):
                with self.subTest(material=material, polarity=polarity):
                    self.assertLess(preference_evidence(
                        product, preference("material", "leather", polarity)) * polarity, 0.0)

    def test_full_qualified_material_value_remains_supported(self):
        product = product_from_dict({"parent_asin": "a", "details": {"Material": "Faux leather"}})
        self.assertEqual(preference_evidence(product, preference("material", "faux leather")), 0.95)
        self.assertEqual(preference_evidence(product, preference("material", "faux leather", -1)), -0.95)

    def test_full_qualified_material_remains_supported_from_title_only(self):
        for material in ("Faux leather", "Faux-leather"):
            product = product_from_dict({"parent_asin": "a", "title": material + " bag"})
            with self.subTest(material=material):
                self.assertGreater(preference_evidence(
                    product, preference("material", material.lower())), 0.0)
                self.assertLess(preference_evidence(
                    product, preference("material", material.lower(), -1)), 0.0)
                self.assertLess(preference_evidence(product, preference("material", "leather")), 0.0)
                self.assertGreater(preference_evidence(product, preference("material", "leather", -1)), 0.0)

    def test_explicit_absence_support_and_unknown_are_distinct(self):
        for title, expected in (("Waterproof jacket", 1), ("Not waterproof jacket", -1), ("Jacket", 0)):
            product = product_from_dict({"parent_asin": "a", "title": title})
            with self.subTest(title=title):
                signal = preference_evidence(product, preference("feature", "waterproof", hard=True))
                self.assertEqual((signal > 0) - (signal < 0), expected)

    def test_direct_avoidance_text_does_not_support_the_avoided_value(self):
        for text in ("Please avoid soaking the strap.", "Avoiding soaking protects the strap.",
                     "Avoidance of soaking is recommended."):
            product = product_from_dict({"parent_asin": "a", "features": [text]})
            with self.subTest(text=text):
                self.assertFalse(value_matches("soaking", text))
                self.assertLess(preference_evidence(product, preference("other", "soaking")), 0.0)
                self.assertGreater(preference_evidence(product, preference("other", "soaking", -1)), 0.0)

    def test_open_vocabulary_exclusion_matches_compact_token_phrase(self):
        product = product_from_dict({"parent_asin": "a", "title": "Replacement shoe laces for running shoes"})
        self.assertGreater(preference_evidence(product, preference("other", "replacement laces")), 0.0)
        self.assertLess(preference_evidence(product, preference("other", "replacement laces", -1)), 0.0)

    def test_open_vocabulary_phrase_matching_handles_repeated_terms(self):
        phrase = "replacement laces running shoes"
        product = product_from_dict({"parent_asin": "a", "title": " ".join([phrase] * 20)})
        self.assertGreater(preference_evidence(product, preference("other", phrase)), 0.0)

    def test_mixed_avoidance_and_affirmative_source_is_unknown(self):
        product = product_from_dict({"parent_asin": "a", "features": ["Avoid soaking."],
                                     "description": "Suitable for soaking."})
        self.assertEqual(preference_evidence(product, preference("other", "soaking")), 0.0)
        self.assertEqual(preference_evidence(product, preference("other", "soaking", -1)), 0.0)

    def test_conflicting_product_sources_remain_unknown(self):
        rows = [
            {"title": "Not waterproof jacket", "features": ["Waterproof"]},
            {"title": "Not waterproof. Waterproof cover included"},
        ]
        for row in rows:
            product = product_from_dict({"parent_asin": "a", **row})
            for polarity in (1, -1):
                with self.subTest(row=row, polarity=polarity):
                    self.assertEqual(preference_evidence(
                        product, preference("feature", "waterproof", polarity, hard=True)), 0.0)

    def test_span_matching_accepts_later_unnegated_occurrence(self):
        self.assertFalse(value_matches("pocket", "Jacket without pockets"))
        self.assertFalse(value_matches("waterproof", "Not waterproof; non-waterproof"))
        self.assertTrue(value_matches("waterproof", "Not waterproof. Waterproof cover included"))

    def test_positive_and_negative_evidence_have_correct_sign(self):
        product = product_from_dict({"parent_asin": "a", "title": "Cotton shirt",
                                     "details": {"Material": "100% cotton"}})
        self.assertGreater(preference_evidence(product, preference("material", "cotton")), 0)
        self.assertLess(preference_evidence(product, preference("material", "cotton", -1)), 0)
        self.assertEqual(preference_evidence(product, preference("material", "wool")), 0)

    def test_category_variants_match_without_title_only_assumption(self):
        product = product_from_dict({"parent_asin": "a", "title": "Everyday classic",
                                     "categories": ["Fashion", "Tops", "Shirts"]})
        self.assertGreater(preference_evidence(product, preference("category", "shirt")), 0)

    def test_ranking_preserves_unique_catalog_ids_and_input_scores(self):
        a = product_from_dict({"parent_asin": "a", "title": "Blue cotton shirt"})
        b = product_from_dict({"parent_asin": "b", "title": "Red wool coat"})
        candidates = [Candidate(b, 1.0), Candidate(a, 0.9)]
        preferences = [preference("category", "shirts"), preference("material", "cotton")]
        ranked = rank_candidates(candidates, preferences)
        self.assertEqual(ranked[0].product.parent_asin, "a")
        self.assertEqual(candidates[0].score, 1.0)
        self.assertIn("evidence", ranked[0].route_scores)
        self.assertEqual(evidence_score(a, []), 0.0)


class ConstraintRankingTest(unittest.TestCase):
    @staticmethod
    def candidates(rows):
        return [Candidate(product_from_dict({"parent_asin": identifier, **row}), score,
                          {"sparse": score}) for identifier, score, row in rows]

    def test_product_guard_demotes_compatible_accessory_but_preserves_unknown(self):
        candidates = self.candidates([
            ("laces", 5.0, {"title": "Replacement shoe laces", "categories": ["Shoe Accessories"]}),
            ("unknown", 2.0, {"title": "Handmade athletic essential"}),
            ("shoes", 1.0, {"title": "Running shoes", "categories": ["Shoes"]}),
        ])
        ranked = rank_product_compatibility(candidates, [preference("category", "sneakers", hard=True)])
        self.assertEqual([item.product.parent_asin for item in ranked], ["unknown", "shoes", "laces"])
        self.assertNotIn("object_penalty", ranked[0].route_scores)
        self.assertEqual(rank_product_compatibility(ranked, [preference("category", "sneakers", hard=True)]),
                         ranked)

    def test_scoped_material_is_a_hard_constraint_only_when_proven(self):
        body = product_from_dict({"parent_asin": "body", "title": "Leather body bag"})
        handles = product_from_dict({"parent_asin": "handles", "title": "Cotton body bag with leather handles"})
        unknown = product_from_dict({"parent_asin": "unknown", "title": "Daily bag"})
        scoped = Preference("material", "leather", 1, "leather body", hard=True, scope="body")
        self.assertGreater(preference_evidence(body, scoped), 0)
        self.assertLess(preference_evidence(handles, scoped), 0)
        self.assertEqual(preference_evidence(unknown, scoped), 0)

    def test_budget_violations_demoted_but_unknown_prices_retained(self):
        candidates = self.candidates([
            ("expensive", 20.0, {"price": 80}),
            ("unknown", 2.0, {}),
            ("from_low", 1.0, {"price": "from $20"}),
            ("affordable", 0.5, {"price": 30}),
            ("from_high", 19.0, {"price": "from $80"}),
        ])
        ranked = rank_constraints(candidates, [preference("budget", "<= 50", hard=True)])
        self.assertEqual([item.product.parent_asin for item in ranked],
                         ["unknown", "from_low", "affordable", "expensive", "from_high"])
        self.assertEqual([item.score for item in ranked[:3]], [2.0, 1.0, 0.5])
        self.assertEqual({item.product.parent_asin for item in ranked},
                         {item.product.parent_asin for item in candidates})
        self.assertTrue(all(left.score >= right.score for left, right in zip(ranked, ranked[1:])))

    def test_budget_floor_and_range_require_confirmed_out_of_bounds_price(self):
        for value in (">= 30", "30-50"):
            with self.subTest(value=value):
                candidates = self.candidates([
                    ("cheap", 10.0, {"price": 20}),
                    ("from_low", 2.0, {"price": "from $20"}),
                    ("unknown", 1.0, {}),
                    ("in_range", 0.0, {"price": 40}),
                ])
                ranked = rank_constraints(candidates, [preference("budget", value, hard=True)])
                self.assertEqual([item.product.parent_asin for item in ranked],
                                 ["from_low", "unknown", "in_range", "cheap"])

    def test_excluded_material_does_not_penalize_faux_leather_or_missing_metadata(self):
        candidates = self.candidates([
            ("leather", 30.0, {"details": {"Material": "Leather"}}),
            ("faux", 3.0, {"details": {"Material": "Faux leather"}}),
            ("free", 2.0, {"details": {"Material": "Leather-free"}}),
            ("unknown", 1.0, {}),
        ])
        ranked = rank_constraints(candidates, [preference("material", "leather", -1, hard=True)])
        self.assertEqual([item.product.parent_asin for item in ranked],
                         ["faux", "free", "unknown", "leather"])
        self.assertEqual([item.score for item in ranked[:3]], [3.0, 2.0, 1.0])
        ranked = rank_constraints(candidates, [preference("material", "faux leather", -1, hard=True)])
        self.assertEqual([item.product.parent_asin for item in ranked],
                         ["leather", "free", "unknown", "faux"])

    def test_negated_feature_does_not_contradict_an_exclusion(self):
        candidates = self.candidates([
            ("waterproof", 20.0, {"title": "Waterproof jacket"}),
            ("not_waterproof", 2.0, {"title": "Not waterproof jacket"}),
            ("unknown", 1.0, {"title": "Jacket"}),
        ])
        ranked = rank_constraints(candidates, [preference("feature", "waterproof", -1, hard=True)])
        self.assertEqual([item.product.parent_asin for item in ranked],
                         ["not_waterproof", "unknown", "waterproof"])

    def test_avoidance_warning_is_not_penalized_for_matching_user_exclusion(self):
        candidates = self.candidates([
            ("soakable", 20.0, {"features": ["Suitable for soaking."]}),
            ("warning", 2.0, {"features": ["Please avoid soaking the strap."]}),
            ("unknown", 1.0, {}),
        ])
        ranked = rank_constraints(candidates, [preference("other", "soaking", -1, hard=True)])
        self.assertEqual([item.product.parent_asin for item in ranked], ["warning", "unknown", "soakable"])
        self.assertEqual([item.score for item in ranked[:2]], [2.0, 1.0])

    def test_explicit_lack_contradicts_hard_requirement_but_missing_facts_do_not(self):
        for attribute, value, lacking, supported in (
                ("feature", "waterproof", "Not waterproof jacket", "Waterproof jacket"),
                ("material", "leather", "Faux leather bag", "Leather bag")):
            candidates = self.candidates([
                ("lacking", 20.0, {"title": lacking}),
                ("unknown", 2.0, {}),
                ("supported", 1.0, {"title": supported}),
            ])
            with self.subTest(value=value):
                ranked = rank_constraints(candidates, [preference(attribute, value, hard=True)])
                self.assertEqual([item.product.parent_asin for item in ranked],
                                 ["unknown", "supported", "lacking"])
                self.assertEqual([item.score for item in ranked[:2]], [2.0, 1.0])

    def test_unconfirmed_requirements_soft_budgets_and_inactive_exclusions_are_neutral(self):
        candidates = self.candidates([
            ("z", 2.0, {"price": 80, "details": {"Material": "Wool"}}),
            ("a", 2.0, {}),
        ])
        inactive = preference("material", "wool", -1)
        inactive.active = False
        for preferences in ([], [preference("budget", "<= 50")], [inactive],
                            [preference("material", "cotton", hard=True)],
                            [preference("material", "any", polarity=0, hard=True)]):
            with self.subTest(preferences=preferences):
                self.assertEqual(rank_constraints(candidates, preferences), candidates)

    def test_preserves_input_objects_product_ids_and_stable_ties(self):
        candidates = self.candidates([
            ("z_bad", 3.0, {"price": 80}),
            ("a_bad", 3.0, {"price": 80}),
            ("z_unknown", 2.0, {}),
            ("a_unknown", 2.0, {}),
        ])
        ranked = rank_constraints(candidates, [preference("budget", "<= 50", hard=True)])
        self.assertEqual([item.product.parent_asin for item in ranked],
                         ["z_unknown", "a_unknown", "z_bad", "a_bad"])
        self.assertEqual([item.score for item in candidates], [3.0, 3.0, 2.0, 2.0])
        self.assertTrue(all(item.route_scores == {"sparse": item.score} for item in candidates))
        originals = {item.product.parent_asin: item.product for item in candidates}
        self.assertTrue(all(item.product is originals[item.product.parent_asin] for item in ranked))
        self.assertEqual(ranked[0].score, ranked[1].score)
        self.assertEqual(ranked[2].score, ranked[3].score)

    def test_penalty_is_recomputed_without_accumulating_and_can_be_retracted(self):
        candidates = self.candidates([
            ("expensive", 20.0, {"price": 80}),
            ("unknown", 1.0, {}),
        ])
        preferences = [preference("budget", "<= 50", hard=True)]
        ranked = rank_constraints(candidates, preferences)
        self.assertEqual(rank_constraints(ranked, preferences), ranked)
        self.assertEqual(rank_constraints(ranked, []), candidates)
        contrast = [Candidate(item.product, item.score + 2.0, {**item.route_scores, "contrast": 2.0})
                    for item in ranked]
        reranked = rank_constraints(contrast, preferences)
        self.assertEqual([item.score for item in reranked], [item.score + 2.0 for item in ranked])
        self.assertEqual(rank_constraints(reranked, preferences), reranked)

    def test_replaced_score_scale_gets_fresh_penalty(self):
        candidates = self.candidates([
            ("expensive", 20.0, {"price": 80}),
            ("unknown", 1.0, {}),
        ])
        preferences = [preference("budget", "<= 50", hard=True)]
        initial = rank_constraints(candidates, preferences)
        replacement = [Candidate(item.product, 1.0 if item.product.price is not None else 0.5,
                                 {key: value for key, value in item.route_scores.items()
                                  if key != "constraint_penalty"}) for item in initial]
        ranked = rank_constraints(replacement, preferences)
        self.assertEqual([item.product.parent_asin for item in ranked], ["unknown", "expensive"])
        self.assertEqual([item.score for item in ranked], [0.5, -0.5])
        self.assertEqual(rank_constraints(ranked, preferences), ranked)

    def test_fractional_penalties_reach_an_exact_fixed_point(self):
        candidates = self.candidates([
            ("low_expensive", -4.9, {"price": 80}),
            ("unknown", -4.9, {}),
            ("high_expensive", 8.4, {"price": 80}),
        ])
        preferences = [preference("budget", "<= 50", hard=True)]
        ranked = rank_constraints(candidates, preferences)
        self.assertEqual(rank_constraints(ranked, preferences), ranked)


class GroupedConstraintRankingTest(unittest.TestCase):
    @staticmethod
    def choices(hard=True):
        return [preference("material", value, hard=hard, alternative_group="material-choice")
                for value in ("cotton", "linen")]

    @staticmethod
    def candidates(features):
        return ConstraintRankingTest.candidates([
            ("known", 20.0, {"title": "Shirt", "features": features}),
            ("unknown", 1.0, {"title": "Shirt"}),
        ])

    def test_all_nine_evidence_combinations(self):
        for cotton in (-1, 0, 1):
            for linen in (-1, 0, 1):
                features = [f"{value.title()} fabric." if signal > 0 else f"{value.title()}-free."
                            for value, signal in (("cotton", cotton), ("linen", linen)) if signal]
                candidates = self.candidates(features)
                preferences = self.choices()
                with self.subTest(cotton=cotton, linen=linen):
                    signals = [preference_evidence(candidates[0].product, item) for item in preferences]
                    self.assertEqual([(signal > 0) - (signal < 0) for signal in signals], [cotton, linen])
                    ranked = rank_constraints(candidates, preferences)
                    known = next(item for item in ranked if item.product.parent_asin == "known")
                    contradicted = cotton == linen == -1
                    self.assertEqual("constraint_penalty" in known.route_scores, contradicted)
                    self.assertEqual(ranked[0].product.parent_asin, "unknown" if contradicted else "known")
                    self.assertEqual(rank_constraints(ranked, preferences), ranked)
                    self.assertEqual(candidates[0].score, 20.0)
                    self.assertNotIn("constraint_penalty", candidates[0].route_scores)

    def test_supported_soft_member_satisfies_group_with_a_hard_member(self):
        candidates = self.candidates(["Cotton-free.", "Linen fabric."])
        preferences = self.choices()
        preferences[1].hard = False
        self.assertEqual(rank_constraints(candidates, preferences), candidates)

    def test_soft_group_is_not_a_hard_requirement(self):
        candidates = self.candidates(["Cotton-free.", "Linen-free."])
        self.assertEqual(rank_constraints(candidates, self.choices(hard=False)), candidates)

    def test_inactive_and_neutral_members_cannot_rescue_a_contradiction(self):
        candidates = self.candidates(["Cotton-free.", "Linen fabric."])
        for removal in ("inactive", "neutral"):
            preferences = self.choices()
            if removal == "inactive":
                preferences[1].active = False
            else:
                preferences[1].polarity = 0
            with self.subTest(removal=removal):
                self.assertEqual(rank_constraints(candidates, preferences)[0].product.parent_asin, "unknown")

    def test_inactive_hard_member_cannot_harden_a_live_soft_group(self):
        candidates = self.candidates(["Cotton fabric.", "Linen-free."])
        preferences = self.choices()
        preferences[0].active = False
        preferences[1].hard = False
        self.assertEqual(rank_constraints(candidates, preferences), candidates)

    def test_negative_requirements_are_independent_of_group_metadata(self):
        candidates = self.candidates(["Cotton fabric.", "Linen-free."])
        for preferences in (
            self.choices() + [preference("material", "cotton", -1, hard=True,
                                         alternative_group="material-choice")],
            [preference("material", value, -1, hard=True, alternative_group="material-choice")
             for value in ("cotton", "linen")],
        ):
            with self.subTest(preferences=preferences):
                self.assertEqual(rank_constraints(candidates, preferences)[0].product.parent_asin, "unknown")

    def test_group_identity_is_scoped_to_attribute(self):
        candidates = self.candidates(["Cotton-free.", "Blue."])
        preferences = [preference("material", "cotton", hard=True, alternative_group="choice"),
                       preference("color", "blue", hard=True, alternative_group="choice")]
        self.assertEqual(rank_constraints(candidates, preferences)[0].product.parent_asin, "unknown")

    def test_distinct_groups_in_one_attribute_remain_independent(self):
        candidates = self.candidates(["Cotton fabric.", "Wool-free.", "Silk-free."])
        preferences = self.choices() + [
            preference("material", value, hard=True, alternative_group="second-choice")
            for value in ("wool", "silk")]
        self.assertEqual(rank_constraints(candidates, preferences)[0].product.parent_asin, "unknown")

    def test_overlapping_values_do_not_merge_distinct_groups(self):
        candidates = self.candidates(["Cotton-free.", "Linen fabric.", "Wool-free."])
        preferences = self.choices() + [
            preference("material", value, hard=True, alternative_group="second-choice")
            for value in ("cotton", "wool")]
        self.assertEqual(rank_constraints(candidates, preferences)[0].product.parent_asin, "unknown")

    def test_ungrouped_conjunction_is_not_weakened(self):
        candidates = self.candidates(["Cotton fabric.", "Linen-free."])
        preferences = [preference("material", value, hard=True) for value in ("cotton", "linen")]
        self.assertEqual(rank_constraints(candidates, preferences)[0].product.parent_asin, "unknown")
        self.assertEqual(rank_constraints(candidates, self.choices()), candidates)

    def test_revised_group_removes_a_stored_penalty(self):
        candidates = self.candidates(["Cotton-free.", "Linen-free."])
        preferences = self.choices()
        ranked = rank_constraints(candidates, preferences)
        preferences.append(preference("material", "silk", hard=True, alternative_group="material-choice"))
        self.assertEqual(rank_constraints(ranked, preferences), candidates)

    def test_score_scale_replacement_reapplies_only_proven_group_penalties(self):
        candidates = self.candidates(["Cotton-free.", "Linen-free."])
        preferences = self.choices()
        initial = rank_constraints(candidates, preferences)
        replacement = [Candidate(item.product, 1.0 if item.product.parent_asin == "known" else 0.5,
                                 {"neural": 1.0}) for item in initial]
        ranked = rank_constraints(replacement, preferences)
        self.assertEqual([item.product.parent_asin for item in ranked], ["unknown", "known"])
        self.assertEqual([item.score for item in ranked], [0.5, -0.5])
        self.assertEqual(rank_constraints(ranked, preferences), ranked)

    def test_feature_alternatives_use_one_evidence_contribution(self):
        product = product_from_dict({"parent_asin": "a", "title": "Jacket",
                                     "features": ["Waterproof.", "Not insulated."]})
        ordinary = [preference("feature", value, hard=True) for value in ("waterproof", "insulated")]
        grouped = [preference("feature", value, hard=True, alternative_group="feature-choice")
                   for value in ("waterproof", "insulated")]
        individual = [evidence_score(product, [item]) for item in ordinary]
        self.assertAlmostEqual(evidence_score(product, ordinary), sum(individual))
        self.assertAlmostEqual(evidence_score(product, grouped), max(individual))

    def test_explicit_feature_groups_do_not_absorb_other_requirements(self):
        product = product_from_dict({"parent_asin": "a", "title": "Jacket",
                                     "features": ["Waterproof.", "Not insulated.", "No pockets."]})
        choices = [preference("feature", value, hard=True, alternative_group="feature-choice")
                   for value in ("waterproof", "insulated")]
        separate = preference("feature", "pockets", hard=True)
        exclusion = preference("feature", "waterproof", -1, alternative_group="feature-choice")
        self.assertAlmostEqual(evidence_score(product, choices + [separate, exclusion]),
                               evidence_score(product, [choices[0]]) + evidence_score(product, [separate])
                               + evidence_score(product, [exclusion]))


if __name__ == "__main__":
    unittest.main()
