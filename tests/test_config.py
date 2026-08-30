import unittest

from mercury.config import Config


class ConfigTest(unittest.TestCase):
    def test_defaults_are_offline_and_conservative(self):
        config = Config()
        self.assertFalse(config.dense)
        self.assertFalse(config.role_evidence)
        self.assertFalse(config.composition_evidence)
        self.assertFalse(config.source_alias_retrieval)
        self.assertFalse(config.neural_rerank)
        self.assertEqual(config.slate_policy, "fixed")
        self.assertEqual(config.slate_size, 10)

    def test_rejects_unknown_keys_and_unsafe_bounds(self):
        for value in ({"scenario_type": "buying"}, {"slate_size": 11},
                      {"candidate_limit": 0}, {"question_policy": "oracle"},
                      {"intent_routing": False}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Config.from_dict(value)

    def test_round_trip(self):
        config = Config.from_dict({"question_policy": "rank_value", "contrast": True, "composition_evidence": True})
        self.assertEqual(Config.from_dict(config.to_dict()), config)

    def test_routed_retrieval_weights_are_bounded(self):
        self.assertTrue(Config(routed_retrieval=True).routed_retrieval)
        for key in ("buying_scoped_weight", "buying_dense_weight", "browsing_dense_weight",
                    "mixed_dense_weight", "browsing_scenario_weight", "mixed_scoped_weight",
                    "intent_object_weight", "intent_slot_weight", "intent_hard_weight",
                    "intent_buying_language_weight", "intent_browsing_language_weight",
                    "intent_use_case_weight", "intent_unresolved_weight",
                    "intent_sparse_request_weight"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                Config.from_dict({key: 1.1})

    def test_intent_rule_defaults_match_the_grouped_cv_artifact(self):
        config = Config()
        self.assertEqual((config.router_buying_threshold, config.router_browsing_threshold), (.5, .5))
        self.assertEqual(
            (config.intent_object_weight, config.intent_slot_weight, config.intent_hard_weight,
             config.intent_buying_language_weight, config.intent_browsing_language_weight,
             config.intent_use_case_weight, config.intent_unresolved_weight,
             config.intent_sparse_request_weight),
            (.2, .25, 0.0, .2, .5, .25, .25, .5),
        )

    def test_product_guard_is_explicitly_gated(self):
        self.assertFalse(Config().product_guard)
        self.assertTrue(Config(product_guard=True).product_guard)
        with self.assertRaises(ValueError):
            Config.from_dict({"product_guard": 1})

    def test_component_scope_parsing_is_explicitly_gated(self):
        self.assertFalse(Config().scoped_preferences)
        self.assertTrue(Config(scoped_preferences=True).scoped_preferences)

    def test_structured_rerank_is_explicitly_gated(self):
        self.assertFalse(Config().structured_rerank)
        self.assertTrue(Config(structured_rerank=True).structured_rerank)

    def test_intent_clarification_controls_are_validated(self):
        config = Config(question_policy="intent", over_general_cutoff=True)
        self.assertEqual(config.question_policy, "intent")
        self.assertTrue(config.over_general_cutoff)
        with self.assertRaises(ValueError):
            Config(question_turn_cost=-0.1)

    def test_runtime_adaptation_controls_are_gated(self):
        config = Config(profile_prior=True, soft_preference_decay=True)
        self.assertTrue(config.profile_prior)
        self.assertTrue(config.soft_preference_decay)
        with self.assertRaises(ValueError):
            Config(profile_weight=2)

    def test_soft_price_weight_is_bounded(self):
        self.assertEqual(Config().soft_price_weight, .02)
        for value in (-.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Config(soft_price_weight=value)

    def test_retrieval_sufficiency_controls_are_gated_and_bounded(self):
        config = Config(retrieval_sufficiency_gate=True, insufficient_action="minimal_probe",
                        max_deferred_turns=1, minimal_probe_limit=30,
                        minimum_retrieval_specificity=.4)
        self.assertTrue(config.retrieval_sufficiency_gate)
        self.assertEqual(config.insufficient_action, "minimal_probe")
        self.assertFalse(Config().retrieval_sufficiency_gate)
        for values in ({"insufficient_action": "skip"}, {"max_deferred_turns": 0},
                       {"minimal_probe_limit": 0}, {"minimum_retrieval_specificity": 1.1},
                       {"retrieval_sufficiency_gate": 1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Config.from_dict(values)

    def test_compute_cascade_has_a_hard_d60_and_session_ceiling(self):
        config = Config(compute_cascade=True, rerank_limit=30,
                        cascade_max_rerank_limit=60, cascade_max_turns=2)
        self.assertTrue(config.compute_cascade)
        for values in ({"compute_cascade": 1},
                       {"compute_cascade": True, "cascade_max_rerank_limit": 120},
                       {"compute_cascade": True, "rerank_limit": 61},
                       {"compute_cascade": True, "cascade_max_turns": 11},
                       {"cascade_threshold": -1}, {"cascade_previous_margin_threshold": -1},
                       {"cascade_previous_margin_threshold": 101}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Config.from_dict(values)

    def test_multi_hypothesis_retrieval_is_gated_and_shares_one_budget(self):
        config = Config(multi_hypothesis_retrieval=True, max_intent_hypotheses=2,
                        hypothesis_candidate_budget=120)
        self.assertTrue(config.multi_hypothesis_retrieval)
        for values in ({"multi_hypothesis_retrieval": 1},
                       {"multi_hypothesis_retrieval": True, "max_intent_hypotheses": 3},
                       {"multi_hypothesis_retrieval": True, "hypothesis_candidate_budget": 121}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Config.from_dict(values)

    def test_semantic_dialogue_controls_are_explicit_booleans(self):
        config = Config(semantic_question_goals=True, require_positive_question_value=True)
        self.assertTrue(config.semantic_question_goals)
        self.assertTrue(config.require_positive_question_value)
        for key in ("semantic_question_goals", "require_positive_question_value"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                Config.from_dict({key: 1})

    def test_evidence_arms_are_separately_configured(self):
        with self.assertRaises(ValueError):
            Config(role_evidence=True, composition_evidence=True)

    def test_source_alias_route_is_broad_only_and_boolean(self):
        self.assertTrue(Config(source_alias_retrieval=True).source_alias_retrieval)
        for value in (1, "true", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Config(source_alias_retrieval=value)
        for mode in ("field_union", "factored"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                Config(source_alias_retrieval=True, retrieval_mode=mode)

    def test_margin_fusion_is_gated_and_conservative(self):
        config = Config(neural_rerank=True, neural_weight=.75, neural_margin_fusion=True,
                        neural_low_margin_weight=.50, neural_margin_threshold=1.0)
        self.assertTrue(config.neural_margin_fusion)
        self.assertEqual(config.neural_low_margin_weight, .50)
        for values in (
            {"neural_margin_fusion": 1},
            {"slate_reset_on_override": 1},
            {"neural_low_margin_weight": -0.1},
            {"neural_margin_threshold": -0.1},
            {"neural_margin_threshold": float("inf")},
            {"neural_margin_fusion": True, "neural_weight": .5, "neural_low_margin_weight": .75},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Config.from_dict(values)

    def test_progressive_frontier_requires_neural_seen_aware_slates(self):
        config = Config(neural_rerank=True, seen_aware_slate=True,
                        progressive_frontier_rerank=True)
        self.assertTrue(config.progressive_frontier_rerank)
        for values in (
            {"seen_aware_slate": 1},
            {"progressive_frontier_rerank": 1},
            {"progressive_frontier_rerank": True, "seen_aware_slate": True},
            {"progressive_frontier_rerank": True, "neural_rerank": True},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Config.from_dict(values)

    def test_page_local_reranking_has_separate_bounded_budgets(self):
        config = Config(
            neural_rerank=True, page_local_rerank=True,
            page_local_rerank_limit=10, page_local_max_batches=2,
            page_local_max_pairs=20, page_local_budget_seconds=.25,
        )
        self.assertTrue(config.page_local_rerank)
        for values in (
            {"page_local_rerank": 1},
            {"page_local_rerank": True},
            {"neural_rerank": True, "page_local_rerank": True,
             "seen_aware_slate": True},
            {"neural_rerank": True, "page_local_rerank": True,
             "page_local_rerank_limit": 11},
            {"neural_rerank": True, "page_local_rerank": True,
             "page_local_rerank_limit": 10, "page_local_max_pairs": 9},
            {"page_local_budget_seconds": 0},
            {"page_local_budget_seconds": float("inf")},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Config.from_dict(values)

    def test_neural_logit_cache_is_bounded_and_requires_the_reranker(self):
        config = Config(
            neural_rerank=True, neural_logit_cache=True,
            neural_logit_cache_size=8192,
        )
        self.assertTrue(config.neural_logit_cache)
        self.assertEqual(Config.from_dict(config.to_dict()), config)
        for values in (
            {"neural_logit_cache": 1},
            {"neural_logit_cache": True},
            {"neural_logit_cache_size": 0},
            {"neural_logit_cache_size": 10001},
            {"neural_logit_cache_size": True},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Config.from_dict(values)

    def test_neural_batch_size_is_restricted_to_registered_matrix_values(self):
        for batch_size in (16, 30, 32):
            with self.subTest(batch_size=batch_size):
                config = Config(neural_batch_size=batch_size)
                self.assertEqual(config.neural_batch_size, batch_size)
                self.assertEqual(Config.from_dict(config.to_dict()), config)
        for batch_size in (0, 15, 31, 33, 16.0, True, "30"):
            with self.subTest(batch_size=batch_size), self.assertRaises(ValueError):
                Config(neural_batch_size=batch_size)

    def test_catalog_vocabulary_is_explicit_and_path_bound(self):
        config = Config(catalog_vocabulary=True, catalog_vocabulary_path="models/vocabulary.json")
        self.assertTrue(config.catalog_vocabulary)
        for values in ({"catalog_vocabulary": 1}, {"catalog_vocabulary_path": ""}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Config.from_dict(values)

    def test_alternatives_are_explicitly_opt_in(self):
        self.assertEqual(Config().alternatives_mode, "off")
        for mode in ("off", "parse", "grouped"):
            with self.subTest(mode=mode):
                config = Config.from_dict({"alternatives_mode": mode})
                self.assertEqual(config.alternatives_mode, mode)
                self.assertEqual(Config.from_dict(config.to_dict()), config)

    def test_rejects_invalid_alternatives_mode(self):
        for mode in ("automatic", "or", "", None, False, 1):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                Config.from_dict({"alternatives_mode": mode})

    def test_rerank_admission_is_explicit_and_bounded(self):
        self.assertEqual(Config().rerank_admission, "prefix")
        for mode in ("prefix", "stratified", "cover", "fusion", "linear"):
            with self.subTest(mode=mode):
                self.assertEqual(Config(rerank_admission=mode).rerank_admission, mode)
        with self.assertRaises(ValueError):
            Config(rerank_admission="oracle")
        with self.assertRaises(ValueError):
            Config(admission_model_path="")

    def test_retrieval_mode_is_explicit_and_bounded(self):
        self.assertEqual(Config().retrieval_mode, "broad")
        for mode in ("broad", "field_union", "factored"):
            with self.subTest(mode=mode):
                self.assertEqual(Config(retrieval_mode=mode).retrieval_mode, mode)
        with self.assertRaises(ValueError):
            Config(retrieval_mode="target_lookup")

    def test_rerank_document_mode_is_explicit_and_bounded(self):
        self.assertEqual(Config().rerank_document_mode, "head")
        for mode in ("head", "lexical", "protected"):
            with self.subTest(mode=mode):
                self.assertEqual(Config(rerank_document_mode=mode).rerank_document_mode, mode)
        with self.assertRaises(ValueError):
            Config(rerank_document_mode="generated_summary")

    def test_grouped_alternatives_require_ledger_state(self):
        for state_mode in ("latest", "history"):
            with self.subTest(state_mode=state_mode), self.assertRaises(ValueError):
                Config(state_mode=state_mode, alternatives_mode="grouped")
            for alternatives_mode in ("off", "parse"):
                self.assertEqual(Config(state_mode=state_mode, alternatives_mode=alternatives_mode).state_mode,
                                 state_mode)

    def test_reranker_model_is_explicit_and_bounded(self):
        self.assertEqual(Config().reranker_model, "reranker")
        for name in ("reranker", "bge_reranker_base"):
            with self.subTest(name=name):
                config = Config(reranker_model=name)
                self.assertEqual(config.reranker_model, name)
                self.assertEqual(Config.from_dict(config.to_dict()), config)
        with self.assertRaises(ValueError):
            Config(reranker_model="target_lookup")

    def test_turn_budget_is_optional_and_non_negative(self):
        self.assertEqual(Config().turn_budget_seconds, 0.0)
        config = Config(turn_budget_seconds=2.5)
        self.assertEqual(config.turn_budget_seconds, 2.5)
        self.assertEqual(Config.from_dict(config.to_dict()), config)
        for value in (-1.0, float("inf"), float("nan"), "2.5", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Config(turn_budget_seconds=value)

    def test_slate_paging_is_opt_in_and_bounded(self):
        self.assertEqual(Config().slate_paging_first_turn, 0)
        for turn in (0, 1, 5, 10):
            with self.subTest(turn=turn):
                config = Config(slate_paging_first_turn=turn)
                self.assertEqual(config.slate_paging_first_turn, turn)
                self.assertEqual(Config.from_dict(config.to_dict()), config)
        for value in (-1, 11, 1.5, "5", True, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Config(slate_paging_first_turn=value)
