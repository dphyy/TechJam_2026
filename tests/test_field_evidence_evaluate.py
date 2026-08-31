from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from experiments.field_evidence_evaluate import (
    ADJUSTMENT_KEY, EVIDENCE_CONFIG, FieldEvidenceAgent, PhraseRanker,
    apply_score_deltas, run_experiment, validate_control,
)
from mercury.agent import Agent
from mercury.config import Config
from mercury.field_evidence import FieldEvidenceResult, FieldWitness
from mercury.types import Candidate


class DeterministicRanker:
    def __init__(self, *args, **kwargs):
        self.asset_identity = "a" * 64
        self.backend_identity = "b" * 64
        self.prompt_tokens = 0
        self.calls = []

    def rank(self, query, candidates, limit, weight, *args, **kwargs):
        self.calls.append((query, [item.product.parent_asin for item in candidates], limit, weight))
        self.prompt_tokens += min(limit, len(candidates))
        result = []
        for index, candidate in enumerate(candidates):
            parts = dict(candidate.route_scores)
            if index < limit:
                parts.update(neural_rank=index + 1, neural_logit=1.0 - index / 100)
            result.append(Candidate(candidate.product, 1.0 - index / 1000, parts))
        return result


class FieldEvidenceEvaluateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.catalog = self.root / "catalog.jsonl"
        rows = [{"parent_asin": f"item-{index:03d}", "title": "Cotton shirt",
                 "categories": ["Shirts"],
                 "features": ["ratchet buckle"] if index == 5 else ["standard closure"]}
                for index in range(150)]
        self.catalog.write_text("\n".join(json.dumps(row) for row in rows))
        self.config = Config.load(Path("configs/selected.json"))
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.config.to_dict()))
        self.agents = []

    def tearDown(self):
        for agent in self.agents:
            agent.close()
        self.temp.cleanup()

    def agent(self, arm=None):
        with patch("mercury.neural.NeuralRanker", DeterministicRanker):
            agent = (Agent(self.catalog, self.config) if arm is None
                     else FieldEvidenceAgent(self.catalog, self.config, arm))
        self.agents.append(agent)
        agent.reset("opaque", {})
        return agent

    def test_off_matches_unwrapped_behavior_prefix_membership_identity_and_cache(self):
        plain, off = self.agent(), self.agent("off")
        for turn, message in enumerate(("I need a cotton shirt with ratchet buckle.",
                                        "I need a cotton shirt with ratchet buckle."), 1):
            with patch("experiments.field_evidence_evaluate.field_phrase_evidence") as evidence:
                actual = off.respond("opaque", message, turn, 10)
            expected = plain.respond("opaque", message, turn, 10)
            evidence.assert_not_called()
            self.assertEqual(actual, expected)
            for key in ("query", "routes", "stage_ids", "rerank_prefix_ids", "ranked_ids",
                        "cache_hit", "effective_capabilities", "fallbacks"):
                self.assertEqual(off.last_diagnostics[key], plain.last_diagnostics[key], key)
        self.assertTrue(off.last_diagnostics["cache_hit"])
        self.assertEqual(off.reranker.inner.calls, plain.reranker.calls)

    def test_scoring_preserves_120_members_and_30_prefix_and_applies_after_replacement(self):
        plain, scored = self.agent(), self.agent("scoring_only")
        message = "I need a cotton shirt with ratchet buckle."
        plain.respond("opaque", message, 1, 10)
        response = scored.respond("opaque", message, 1, 10)
        before, after = plain.last_diagnostics, scored.last_diagnostics
        self.assertEqual(before["stage_ids"]["candidate_limited"], after["stage_ids"]["candidate_limited"])
        self.assertEqual(before["rerank_prefix_ids"], after["rerank_prefix_ids"])
        self.assertEqual(len(after["stage_ids"]["candidate_limited"]), 120)
        self.assertEqual(len(after["rerank_prefix_ids"]), 30)
        self.assertEqual(set(before["ranked_ids"]), set(after["ranked_ids"]))
        self.assertEqual(plain.reranker.calls, scored.reranker.inner.calls)
        receipt = after["field_evidence"]
        self.assertEqual(receipt["score_application_count"], 1)
        self.assertEqual(len(response["recommendations"]), 10)
        self.assertEqual(set(receipt["score_deltas"]), {"item-005"})
        baseline_scores = {item.product.parent_asin: item.score for item in plain._last_candidates["opaque"]}
        changed_scores = {item.product.parent_asin: item.score for item in scored._last_candidates["opaque"]}
        for identifier in baseline_scores:
            self.assertAlmostEqual(changed_scores[identifier] - baseline_scores[identifier],
                                   receipt["score_deltas"].get(identifier, 0.0))

    def test_cached_scoring_is_not_applied_twice(self):
        agent = self.agent("scoring_only")
        message = "I need a cotton shirt with ratchet buckle."
        agent.respond("opaque", message, 1, 10)
        before = [(item.product.parent_asin, item.score) for item in agent._last_candidates["opaque"]]
        agent.respond("opaque", message, 2, 10)
        self.assertTrue(agent.last_diagnostics["cache_hit"])
        self.assertEqual(agent.last_diagnostics["field_evidence"]["score_application_count"], 0)
        self.assertEqual(before, [(item.product.parent_asin, item.score)
                                  for item in agent._last_candidates["opaque"]])
        self.assertEqual(len(agent.reranker.inner.calls), 1)

    def test_model_identity_and_token_accounting_remain_delegated(self):
        agent = self.agent("scoring_only")
        self.assertIsInstance(agent.reranker, PhraseRanker)
        self.assertEqual(agent.reranker.asset_identity, agent.reranker.inner.asset_identity)
        agent.respond("opaque", "I need a cotton shirt.", 1, 10)
        before = agent.last_diagnostics["effective_capabilities"]["identity_sha256"]
        agent.reranker.inner.backend_identity = "c" * 64
        response = agent.respond("opaque", "I need a cotton shirt.", 2, 10)
        self.assertFalse(agent.last_diagnostics["cache_hit"])
        self.assertNotEqual(before, agent.last_diagnostics["effective_capabilities"]["identity_sha256"])
        self.assertEqual(response["usage"]["prompt_tokens"], 30)

    def test_admission_reports_tail_dropped_by_unchanged_candidate_cap(self):
        agent = self.agent("admission_only")
        base = [Candidate(product, 1.0 - index / 1000, {"sparse": 1.0})
                for index, product in enumerate(agent.catalog.products) if product.parent_asin != "item-005"][:120]
        base_ids = [item.product.parent_asin for item in base]
        with patch.object(Agent, "_retrieve", return_value=(base, {"sparse": base_ids}, {"sparse": 1.0})):
            agent.respond("opaque", "I need a cotton shirt with ratchet buckle.", 1, 10)
        receipt = agent.last_diagnostics["field_evidence"]
        self.assertEqual(receipt["admitted_ids"], ["item-005"])
        self.assertEqual(receipt["admitted_stage_survival"]["retrieved"], ["item-005"])
        self.assertEqual(receipt["admitted_stage_survival"]["candidate_limited"], [])
        self.assertEqual(receipt["admitted_stage_survival"]["final_ranked"], [])
        self.assertEqual(receipt["score_application_count"], 0)
        self.assertEqual(agent.last_diagnostics["stage_ids"]["candidate_limited"], base_ids)

    def test_delta_is_idempotent_and_requires_a_supported_bounded_witness(self):
        agent = self.agent("off")
        product = agent.catalog.products[0]
        witness = FieldWitness("ratchet buckle", "features", "ratchet buckle", 0, 14, 1, 0.5)
        evidence = FieldEvidenceResult([product.parent_asin], score_deltas={product.parent_asin: 0.04},
                                       witnesses={product.parent_asin: witness})
        once = apply_score_deltas([Candidate(product, 2.0)], evidence)
        twice = apply_score_deltas(once, evidence)
        self.assertEqual(once, twice)
        self.assertEqual(once[0].score, 2.04)
        self.assertEqual(once[0].route_scores[ADJUSTMENT_KEY], 0.04)
        for delta in (0.081, float("nan"), -0.01):
            evidence.score_deltas[product.parent_asin] = delta
            with self.assertRaises(ValueError):
                apply_score_deltas(once, evidence)
        evidence.score_deltas[product.parent_asin] = 0.04
        evidence.witnesses.clear()
        with self.assertRaises(ValueError):
            apply_score_deltas(once, evidence)

    def test_quoted_negation_produces_no_adjustment_or_membership_change(self):
        plain, scored = self.agent(), self.agent("scoring_only")
        message = 'I need a cotton shirt, not "ratchet buckle".'
        expected = plain.respond("opaque", message, 1, 10)
        actual = scored.respond("opaque", message, 1, 10)
        self.assertEqual(actual, expected)
        self.assertEqual(scored.last_diagnostics["field_evidence"]["score_deltas"], {})

    def test_full_width_policy_is_preserved_through_all_ten_turns(self):
        agent = self.agent("scoring_only")
        for turn in range(1, 11):
            response = agent.respond("opaque", "I need a cotton shirt with ratchet buckle.", turn, 10)
            self.assertEqual(len(response["recommendations"]), 10)
            self.assertEqual(agent.config.slate_policy, "fixed")
            self.assertEqual(agent.config.slate_size, 10)

    def test_dynamic_control_options_are_rejected(self):
        for changes in ({"candidate_limit": 119}, {"rerank_limit": 31}, {"slate_size": 9},
                        {"slate_policy": "gap"}, {"turn_budget_seconds": 1.0},
                        {"routed_retrieval": True}, {"page_local_rerank": True},
                        {"constraint_check_stage": "pre"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_control(replace(self.config, **changes))

    def test_runtime_extensions_have_no_dataset_or_outcome_access(self):
        for runtime in (FieldEvidenceAgent, PhraseRanker, apply_score_deltas):
            code = inspect.getsource(runtime)
            for forbidden in ("ground_truth", "sample_id", "load_jsonl", "dataset", "technical_score"):
                self.assertNotIn(forbidden, code)

    def test_missing_neural_backend_remains_a_fallback_without_score_application(self):
        with patch("mercury.neural.NeuralRanker", side_effect=OSError("unavailable")):
            agent = FieldEvidenceAgent(self.catalog, self.config, "scoring_only")
        self.agents.append(agent)
        agent.reset("opaque", {})
        agent.respond("opaque", "I need a cotton shirt with ratchet buckle.", 1, 10)
        receipt = agent.last_diagnostics["field_evidence"]
        self.assertFalse(receipt["neural_available"])
        self.assertEqual(receipt["score_application_count"], 0)
        self.assertIn("neural_rerank", agent.last_diagnostics["fallbacks"])

    def test_harness_reserves_registration_before_inference_and_outputs_are_create_only(self):
        dataset = self.root / "dataset.jsonl"
        dataset.write_text(json.dumps({"sample_id": "authored", "scenario_type": "buying",
                                       "user_profile": {}, "ground_truth": {"parent_asin": "item-005"}}))
        output = self.root / "output"

        class RegisteredRanker(DeterministicRanker):
            def rank(inner, *args, **kwargs):
                registration = json.loads((output / "registration.json").read_text())
                self.assertEqual(registration["evidence_config"]["score_cap"], 0.08)
                return super(RegisteredRanker, inner).rank(*args, **kwargs)

        with patch("mercury.neural.NeuralRanker", RegisteredRanker):
            report = run_experiment(self.config_path, self.catalog, dataset, output)
        self.assertEqual([run["name"] for run in report["runs"]], ["off", "scoring_only"])
        self.assertTrue((output / "suite" / "report.json").is_file())
        self.assertTrue((output / "field_traces.json").is_file())
        before = (output / "registration.json").read_bytes()
        with self.assertRaises(FileExistsError):
            run_experiment(self.config_path, self.catalog, dataset, output)
        self.assertEqual((output / "registration.json").read_bytes(), before)
        self.assertEqual(EVIDENCE_CONFIG.score_cap, 0.08)

    def test_failed_evaluation_leaves_consumed_registration_and_error_receipt(self):
        dataset = self.root / "dataset.jsonl"
        dataset.write_text("{}\n")
        output = self.root / "failed"
        with patch("experiments.field_evidence_evaluate.suite.evaluate_suite", side_effect=RuntimeError("fail")):
            with self.assertRaises(RuntimeError):
                run_experiment(self.config_path, self.catalog, dataset, output)
        self.assertTrue((output / "registration.json").is_file())
        self.assertEqual(json.loads((output / "error.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
