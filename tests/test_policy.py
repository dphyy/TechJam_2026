import copy
import math
import unittest
from unittest.mock import patch

from mercury.config import Config
from mercury.policy import choose_policy
from mercury.ranking import rank_candidates
from mercury.state import SessionState
from mercury.types import Candidate, FacetEvidence, IntentDecision, Preference, Product


ATTRIBUTES = ("category", "material", "color", "size", "style", "brand",
              "budget", "feature", "use_case")


def candidate(index, score=None, facets=None, confidence=None, price=None, lower_bound=False):
    facets = facets or {}
    confidence = confidence or {}
    evidence = tuple(
        FacetEvidence(attribute, value, "details", confidence.get(attribute, 1.0))
        for attribute, values in facets.items() for value in values
    )
    product = Product(f"P{index:03d}", f"Item {index}", {"title": f"Item {index}"},
                      facets, evidence, price, lower_bound)
    return Candidate(product, 1.0 - index * 0.03 if score is None else score)


def material_pool(count=12):
    return [candidate(index, facets={"material": ("cotton" if index < 10 else "wool",)})
            for index in range(count)]


class PolicyTest(unittest.TestCase):
    def test_semantic_other_goals_skip_paraphrases_of_the_same_question(self):
        state = SessionState({})
        config = Config(question_policy="other", other_question_limit=4,
                        semantic_question_goals=True)
        decisions = []
        for turn in range(1, 4):
            decision = choose_policy(state, material_pool(), config, turn, 10)
            decisions.append(decision)
            state.record_question(decision.ask_attribute, decision.question_goal)
        self.assertEqual([decision.question_goal for decision in decisions], [
            "other:open_detail", "other:must_have_or_dealbreaker", "other:priority",
        ])
        self.assertEqual(len({decision.message for decision in decisions}), 3)

    def test_positive_value_gate_does_not_ask_to_consume_a_turn(self):
        state = SessionState({})
        state.update("I am exploring.", 1)
        intent = IntentDecision("browsing", .1, .9, 0, True)
        decision = choose_policy(
            state, [], Config(question_policy="intent", require_positive_question_value=True),
            1, 10, intent=intent,
        )
        self.assertIsNone(decision.ask_attribute)
        self.assertEqual(decision.diagnostics["decision"], "question_value_below_turn_cost")

    def test_generic_discovery_prompts_do_not_repeat_verbatim(self):
        state = SessionState({})
        prompts = []
        config = Config(other_question_limit=9)
        for turn, message in enumerate(("A shirt", "Cotton", "Blue"), 1):
            state.update(message, turn)
            decision = choose_policy(state, material_pool(), config, turn, 10)
            self.assertEqual(decision.ask_attribute, "other")
            prompts.append(decision.message)
            state.record_question("other")
        self.assertEqual(len(prompts), len(set(prompts)))
        state.update("Nothing more to add", 4)
        self.assertNotEqual(choose_policy(state, material_pool(), config, 4, 10).ask_attribute, "other")

    def test_other_question_limit_is_bounded(self):
        for value in (-1, 10, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Config(other_question_limit=value)

    def test_default_asks_useful_other_question_and_shows_full_slate(self):
        decision = choose_policy(SessionState({}), material_pool(), Config(), 1, 10)
        self.assertEqual(decision.ask_attribute, "other")
        self.assertEqual(decision.slate_size, 10)
        self.assertIn("?", decision.message)
        self.assertNotIn("P000", decision.message)

    def test_discriminating_policy_asks_one_well_supported_top30_split(self):
        pool = [candidate(index, facets={"color": ("blue" if index < 15 else "red",)})
                for index in range(30)]
        state = SessionState({})
        config = Config(question_policy="discriminating")
        first = choose_policy(state, pool, config, 1, 10)
        self.assertEqual(first.ask_attribute, "color")
        self.assertEqual(first.diagnostics["decision"], "well_supported_top30_split")
        state.record_question(first.ask_attribute)
        second = choose_policy(state, pool, config, 2, 10)
        self.assertEqual(second.ask_attribute, "other")

    def test_discriminating_policy_falls_back_when_facets_are_sparse(self):
        pool = [candidate(index, facets={"color": ("blue",)} if index < 10 else {})
                for index in range(30)]
        decision = choose_policy(
            SessionState({}), pool, Config(question_policy="discriminating"), 1, 10,
        )
        self.assertEqual(decision.ask_attribute, "other")
        self.assertEqual(decision.diagnostics["decision"], "bounded_other_fallback")

    def test_slate_respects_available_candidates_and_requested_cap(self):
        for top_k, count, expected in ((50, 12, 10), (3, 12, 3), (10, 2, 2),
                                       (0, 12, 0), (-1, 12, 0), (10, 0, 0)):
            with self.subTest(top_k=top_k, count=count):
                decision = choose_policy(SessionState({}), material_pool(count), Config(), 1, top_k)
                self.assertEqual(decision.slate_size, expected)

    def test_final_turn_restores_full_slate_and_does_not_ask_unusable_question(self):
        for question in ("other", "schedule", "entropy", "rank_value", "none"):
            for slate in ("fixed", "gap", "lookahead"):
                with self.subTest(question=question, slate=slate):
                    config = Config(question_policy=question, slate_policy=slate, slate_size=0)
                    decision = choose_policy(SessionState({}), material_pool(), config, 10, 10)
                    self.assertEqual(decision.slate_size, 10)
                    self.assertIsNone(decision.ask_attribute)
                    self.assertNotIn("?", decision.message)

    def test_final_turn_still_obeys_requested_cap(self):
        decision = choose_policy(SessionState({}), material_pool(), Config(slate_size=1), 10, 4)
        self.assertEqual(decision.slate_size, 4)

    def test_explicit_abstention_is_bounded_to_one_consecutive_turn(self):
        for policy in ("fixed", "gap", "lookahead"):
            config = Config(slate_size=0, slate_policy=policy)
            first = choose_policy(SessionState({}), material_pool(), config, 1, 10)
            second = choose_policy(SessionState({}), material_pool(), config, 2, 10, 1)
            self.assertEqual(first.slate_size, 0)
            self.assertEqual(second.slate_size, 10)

    def test_no_candidates_cannot_be_recovered_by_abstention_guard(self):
        decision = choose_policy(SessionState({}), [], Config(slate_size=0), 3, 10, 1)
        self.assertEqual(decision.slate_size, 0)
        self.assertEqual(decision.ask_attribute, "other")

    def test_none_policy_does_not_hide_a_question_in_message(self):
        decision = choose_policy(SessionState({}), material_pool(), Config(question_policy="none"), 1, 10)
        self.assertIsNone(decision.ask_attribute)
        self.assertNotIn("?", decision.message)
        self.assertEqual(decision.slate_size, 10)

    def test_intent_policy_asks_product_type_for_broad_browsing(self):
        intent = IntentDecision("browsing", .1, .9, 0, True, ("over_general",))
        decision = choose_policy(SessionState({}), material_pool(), Config(question_policy="intent"),
                                 1, 10, intent=intent)
        self.assertEqual(decision.ask_attribute, "category")
        self.assertEqual(decision.diagnostics["decision"], "browsing_product_type")
        self.assertEqual(decision.diagnostics["ask_turn_cost"], .02)

    def test_intent_policy_does_not_repeat_no_preference_slot(self):
        state = SessionState({})
        state.record_question("category")
        state.update("I do not have a category preference.", 1)
        intent = IntentDecision("browsing", .1, .9, 0, True)
        decision = choose_policy(state, material_pool(), Config(question_policy="intent"),
                                 2, 10, intent=intent)
        self.assertNotEqual(decision.ask_attribute, "category")

    def test_intent_policy_never_asks_on_final_turn(self):
        intent = IntentDecision("browsing", .1, .9, 0, True)
        decision = choose_policy(SessionState({}), material_pool(), Config(question_policy="intent"),
                                 10, 10, intent=intent)
        self.assertIsNone(decision.ask_attribute)

    def test_schedule_avoids_known_negative_neutral_asked_and_unproductive_fields(self):
        state = SessionState({})
        state.preferences = [Preference("category", "shirts", 1, "shirts"),
                             Preference("material", "wool", 1, "no wool", polarity=-1),
                             Preference("color", "any", 1, "any", polarity=0)]
        state.asked_counts["size"] = 1
        state.unproductive_attributes.add("style")
        decision = choose_policy(state, material_pool(), Config(question_policy="schedule"), 2, 10)
        self.assertEqual(decision.ask_attribute, "brand")

    def test_inactive_preferences_do_not_block_questions(self):
        state = SessionState({})
        state.preferences.append(Preference("category", "shirts", 1, "shirts", active=False))
        decision = choose_policy(state, [], Config(question_policy="schedule"), 2, 10)
        self.assertEqual(decision.ask_attribute, "category")

    def test_profile_is_not_treated_as_current_answer(self):
        decision = choose_policy(SessionState({"preferred_category": "shirts"}), [],
                                 Config(question_policy="schedule"), 1, 10)
        self.assertEqual(decision.ask_attribute, "category")

    def test_recovery_after_uninformative_other_answer_changes_question_and_restores_slate(self):
        state = SessionState({})
        state.update("A shirt.", 1)
        state.record_question("other")
        state.update("Nothing to add.", 2)
        decision = choose_policy(state, material_pool(), Config(slate_size=2), 2, 10)
        self.assertNotIn(decision.ask_attribute, ("other", "category", None))
        self.assertEqual(decision.slate_size, 10)

    def test_no_preference_answer_is_not_repeated_even_when_state_changed(self):
        state = SessionState({})
        state.record_question("material")
        state.update("No preference.", 2)
        decision = choose_policy(state, material_pool(), Config(question_policy="entropy", slate_size=1), 2, 10)
        self.assertNotEqual(decision.ask_attribute, "material")
        self.assertEqual(decision.slate_size, 10)

    def test_entropy_accounts_for_unknown_catalog_values(self):
        pool = [candidate(0, facets={"material": ("cotton",), "color": ("blue",)}),
                candidate(1, facets={"color": ("blue",)})]
        decision = choose_policy(SessionState({}), pool, Config(question_policy="entropy"), 1, 10)
        self.assertEqual(decision.ask_attribute, "material")
        self.assertAlmostEqual(decision.diagnostics["facet_entropy"]["material"], 1.0)
        self.assertAlmostEqual(decision.diagnostics["unknown_mass"]["material"], 0.5)

    def test_entropy_does_not_turn_completely_missing_field_into_disagreement(self):
        decision = choose_policy(SessionState({}), [candidate(0), candidate(1)],
                                 Config(question_policy="entropy"), 1, 10)
        self.assertEqual(decision.ask_attribute, "other")
        self.assertTrue(all(value == 0.0 for value in decision.diagnostics["facet_entropy"].values()))

    def test_entropy_splits_multivalue_product_mass_without_double_counting(self):
        pool = [candidate(0, facets={"color": ("red", "blue", "red")}), candidate(1)]
        decision = choose_policy(SessionState({}), pool, Config(question_policy="entropy"), 1, 10)
        self.assertAlmostEqual(decision.diagnostics["facet_entropy"]["color"], 1.5)
        self.assertAlmostEqual(decision.diagnostics["unknown_mass"]["color"], 0.5)

    def test_price_lower_bounds_do_not_become_exact_budget_answers(self):
        pool = [candidate(0, price=10, lower_bound=True), candidate(1, price=80, lower_bound=True)]
        decision = choose_policy(SessionState({}), pool, Config(question_policy="entropy"), 1, 10)
        self.assertEqual(decision.diagnostics["facet_entropy"]["budget"], 0.0)
        self.assertEqual(decision.diagnostics["unknown_mass"]["budget"], 1.0)

    def test_exact_catalog_prices_can_make_budget_question_useful(self):
        pool = [candidate(0, price=20), candidate(1, price=80), candidate(2)]
        decision = choose_policy(SessionState({}), pool, Config(question_policy="entropy"), 1, 10)
        self.assertEqual(decision.ask_attribute, "budget")

    def test_rank_value_prefers_rank_recovery_over_raw_facet_entropy(self):
        pool = [candidate(index, facets={"color": (f"shade-{index}",),
                                        "material": ("cotton" if index < 10 else "wool",)},
                          confidence={"color": 0.001}) for index in range(12)]
        entropy = choose_policy(SessionState({}), pool, Config(question_policy="entropy"), 1, 10)
        value = choose_policy(SessionState({}), pool, Config(question_policy="rank_value"), 1, 10)
        self.assertEqual(entropy.ask_attribute, "color")
        self.assertEqual(value.ask_attribute, "material")
        self.assertGreater(value.diagnostics["question_scores"]["material"],
                           value.diagnostics["question_scores"]["color"])
        self.assertEqual(value.slate_size, 10)

    def test_rank_value_explicitly_labels_heuristic_weights_and_uncertainty(self):
        pool = material_pool() + [candidate(12)]
        decision = choose_policy(SessionState({}), pool, Config(question_policy="rank_value"), 1, 10)
        self.assertIn("heuristic", decision.diagnostics["weight_model"])
        self.assertIn("not_calibrated", decision.diagnostics["weight_model"])
        model = decision.diagnostics["outcome_models"]["material"]
        outcomes = model["outcomes"]
        self.assertTrue({"unknown", "no_preference", "outside_pool"} <= {o["kind"] for o in outcomes})
        self.assertAlmostEqual(sum(outcome["weight"] for outcome in outcomes), 1.0)
        for outcome in outcomes:
            if outcome["kind"] != "value":
                self.assertEqual(outcome["expected_rr_gain"], 0.0)
        self.assertGreater(model["expected_rr_gain"], 0.0)
        self.assertGreater(model["expected_recovery_gain"], 0.0)
        self.assertEqual(decision.diagnostics["reference_slate_size"], 10)

    def test_rank_value_has_no_fictional_ask_cost_and_keeps_same_turn_recommendations(self):
        with patch("mercury.policy.rank_candidates", wraps=rank_candidates) as rerank:
            decision = choose_policy(SessionState({}), material_pool(), Config(question_policy="rank_value"), 1, 10)
        self.assertTrue(rerank.called)
        self.assertEqual(decision.ask_attribute, "material")
        self.assertEqual(decision.slate_size, 10)
        self.assertEqual(decision.diagnostics["ask_turn_cost"], 0.0)

    def test_rank_value_bounds_simulations_and_keeps_unmodeled_answer_mass(self):
        pool = [candidate(index, facets={attribute: (f"value-{index}",) for attribute in ATTRIBUTES
                                        if attribute != "budget"}) for index in range(100)]
        with patch("mercury.policy.rank_candidates", wraps=rank_candidates) as rerank:
            decision = choose_policy(SessionState({}), pool, Config(question_policy="rank_value"), 1, 10)
        self.assertLessEqual(rerank.call_count, 9 * 4)
        self.assertTrue(all(len(call.args[0]) <= 40 for call in rerank.call_args_list))
        self.assertEqual(decision.diagnostics["pool_size"], 40)
        self.assertEqual(decision.diagnostics["unmodeled_candidates"], 60)
        for model in decision.diagnostics["outcome_models"].values():
            self.assertLessEqual(sum(o["kind"] == "value" for o in model["outcomes"]), 4)
            self.assertAlmostEqual(sum(o["weight"] for o in model["outcomes"]), 1.0)
        self.assertTrue(any(o["kind"] == "unmodeled_value" and o["weight"] > 0
                            for o in decision.diagnostics["outcome_models"]["material"]["outcomes"]))

    def test_zero_rank_gain_falls_back_to_other_instead_of_using_entropy(self):
        pool = [candidate(index, score=100 - index * 10,
                          facets={"material": ("cotton" if index < 10 else "wool",)})
                for index in range(12)]
        decision = choose_policy(SessionState({}), pool, Config(question_policy="rank_value"), 1, 10)
        self.assertEqual(decision.ask_attribute, "other")
        self.assertEqual(decision.diagnostics["question_scores"]["material"], 0.0)

    def test_rank_weights_are_not_conditioned_on_prior_recommendation_failure(self):
        before = choose_policy(SessionState({}), material_pool(), Config(question_policy="rank_value"), 1, 10)
        later_state = SessionState({})
        later_state.update("Keep looking.", 2)
        after = choose_policy(later_state, material_pool(), Config(question_policy="rank_value"), 2, 10)
        self.assertEqual(before.diagnostics["question_scores"], after.diagnostics["question_scores"])

    def test_gap_slate_is_opt_in_and_never_automatically_empty(self):
        pool = [candidate(index, score=10 - index * 0.01 if index < 3 else -index * 0.01)
                for index in range(12)]
        fixed = choose_policy(SessionState({}), pool, Config(), 1, 10)
        gap = choose_policy(SessionState({}), pool, Config(slate_policy="gap"), 1, 10)
        self.assertEqual(fixed.slate_size, 10)
        self.assertEqual(gap.slate_size, 3)
        self.assertEqual(fixed.ask_attribute, gap.ask_attribute)

    def test_flat_or_nonfinite_gaps_keep_full_slate(self):
        for scores in ([1.0] * 12, [math.nan] * 12, [math.inf] * 12):
            pool = [candidate(index, score=score) for index, score in enumerate(scores)]
            decision = choose_policy(SessionState({}), pool, Config(slate_policy="gap"), 1, 10)
            self.assertEqual(decision.slate_size, 10)

    def test_lookahead_shortens_only_when_answer_reranks_preserve_prefix(self):
        state = SessionState({})
        state.preferences.append(Preference("category", "shirts", 1, "shirts"))
        pool = [candidate(index, score=10 - index * 0.01 if index < 3 else -index * 0.01,
                          facets={"material": ("cotton" if index < 3 else "wool",)})
                for index in range(12)]
        decision = choose_policy(state, pool, Config(question_policy="schedule", slate_policy="lookahead"), 1, 10)
        self.assertEqual(decision.ask_attribute, "material")
        self.assertEqual(decision.slate_size, 3)

    def test_lookahead_keeps_full_slate_when_plausible_answer_changes_prefix(self):
        state = SessionState({})
        state.preferences.append(Preference("category", "shirts", 1, "shirts"))
        pool = [candidate(index, score=0.31 - index * 0.001 if index < 3 else 0.0,
                          facets={"material": ("cotton" if index < 3 else "wool",)})
                for index in range(12)]
        decision = choose_policy(state, pool, Config(question_policy="schedule", slate_policy="lookahead"), 1, 10)
        self.assertEqual(decision.slate_size, 10)

    def test_question_selection_is_independent_of_slate_policy(self):
        for question in ("other", "schedule", "entropy", "rank_value", "none"):
            selected = [choose_policy(SessionState({}), material_pool(),
                                      Config(question_policy=question, slate_policy=slate), 1, 10).ask_attribute
                        for slate in ("fixed", "gap", "lookahead")]
            self.assertEqual(len(set(selected)), 1)

    def test_question_messages_match_only_the_selected_attribute(self):
        words = {"category": "item", "material": "material", "color": "color", "size": "size",
                 "style": "style", "brand": "brand", "budget": "budget", "feature": "feature",
                 "use_case": "use"}
        for attribute in ATTRIBUTES:
            state = SessionState({})
            state.unproductive_attributes = set(ATTRIBUTES) - {attribute}
            decision = choose_policy(state, material_pool(), Config(question_policy="schedule"), 1, 10)
            self.assertEqual(decision.ask_attribute, attribute)
            self.assertIn(words[attribute], decision.message.lower())
            self.assertIn("?", decision.message)

    def test_exhausted_questions_can_stop_without_repeating_refused_other(self):
        state = SessionState({})
        state.unproductive_attributes = {*ATTRIBUTES, "other"}
        for question in ("other", "schedule", "entropy", "rank_value"):
            decision = choose_policy(state, material_pool(), Config(question_policy=question), 2, 10)
            self.assertIsNone(decision.ask_attribute)
            self.assertNotIn("?", decision.message)
            self.assertEqual(decision.slate_size, 10)

    def test_policy_does_not_mutate_state_candidates_or_scores(self):
        state = SessionState({})
        state.update("A shirt.", 1)
        pool = material_pool()
        snapshot = copy.deepcopy((state.__dict__, pool))
        choose_policy(state, pool, Config(question_policy="rank_value", slate_policy="lookahead"), 1, 10)
        self.assertEqual((state.__dict__, pool), snapshot)


if __name__ == "__main__":
    unittest.main()
