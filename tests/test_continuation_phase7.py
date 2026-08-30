import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.causal_attribution import attribute_session, paired_diff, timing_reconciliation
from experiments.robustness_matrix_v2_prepare import build_matrix, power_calculation
from mercury.agent import Agent
from mercury.config import Config


def product(index: int, category: str) -> dict:
    return {
        "parent_asin": f"P{index:04d}",
        "title": f"Distinct item model{index}",
        "categories": ["Catalog", category],
        "features": [f"feature-{index}"],
        "rating_number": index,
    }


class ContinuationMatrixTest(unittest.TestCase):
    def test_matrix_excludes_prior_targets_families_categories_and_annotation_groups(self):
        products = [product(index, f"category-{index // 3}") for index in range(90)]
        public = [{"sample_id": "public", "ground_truth": {"parent_asin": "P0000"}}]
        consumed = [[
            {"sample_id": "old-1", "ground_truth": {"parent_asin": "P0003"}},
            {"sample_id": "old-2", "ground_truth": {"parent_asin": "P0006"}},
        ]]
        prior = [[{"author_family": "old-author", "user_family": "old-user"}]]
        counts = {"training": 12, "screening": 6, "confirmation": 3}
        with patch("experiments.robustness_matrix_v2_prepare.SPLIT_COUNTS", counts):
            matrix = build_matrix(products, public, consumed, prior, seed="test-v2")
        selected = {
            row["ground_truth"]["parent_asin"]
            for rows in matrix["datasets"].values() for row in rows
        }
        self.assertTrue(selected.isdisjoint({"P0000", "P0003", "P0006"}))
        selected_categories = {
            products[int(identifier[1:])]["categories"][-1] for identifier in selected
        }
        self.assertTrue(selected_categories.isdisjoint({"category-0", "category-1", "category-2"}))
        self.assertEqual(matrix["audit"]["cross_split_target_overlap"], 0)
        self.assertTrue(all(value == 0 for value in matrix["audit"]["cross_split_group_overlap"].values()))

    def test_power_receipt_is_bounded_and_records_effective_rows(self):
        receipt = power_calculation(160)
        self.assertEqual(receipt["effective_rows_at_75_percent"], 120)
        self.assertGreaterEqual(receipt["approximate_power"], 0)
        self.assertLessEqual(receipt["approximate_power"], 1)


class CausalAttributionTest(unittest.TestCase):
    def test_attributes_retrieval_admission_ranking_and_paging(self):
        result = {"sample_id": "x", "hit": False}
        cases = {
            "retrieval": [{"diagnostics": {"semantic_state_signature": [["category", "shirt"]],
                                                "retrieved_ids": [], "stage_ids": {}}}],
            "admission": [{"diagnostics": {"retrieved_ids": ["T"], "stage_ids": {
                "admission_selected": ["A"], "final_ranked": ["A"], "returned_page": ["A"],
            }}}],
            "ranking": [{"diagnostics": {"retrieved_ids": ["T"], "stage_ids": {
                "admission_selected": ["T"], "neural_ranked": [str(index) for index in range(11)] + ["T"],
                "guarded_after_rerank": [str(index) for index in range(11)] + ["T"],
                "final_ranked": [str(index) for index in range(11)] + ["T"],
                "returned_page": ["A"],
            }}}],
            "paging": [{"diagnostics": {"retrieved_ids": ["T"], "stage_ids": {
                "admission_selected": ["T"], "neural_ranked": ["T"],
                "guarded_after_rerank": ["T"], "final_ranked": ["T"],
                "returned_page": ["A"],
            }}}],
        }
        for expected, trace in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(attribute_session(trace, result, "T")["stage"], expected)

    def test_reconciles_component_timings_and_pairs_gains_losses(self):
        traces = [[{"latency_seconds": 0.10, "diagnostics": {
            "component_latency_seconds": {"parsing": 0.02, "retrieval": 0.03, "policy": 0.04},
        }}]]
        self.assertEqual(timing_reconciliation(traces)["reconciled_fraction"], 1.0)
        control = {"sessions": [
            {"sample_id": "a", "hit": False, "stage": "ranking", "scenario": "buying"},
            {"sample_id": "b", "hit": True, "stage": "success", "scenario": "browsing"},
        ]}
        candidate = {"sessions": [
            {"sample_id": "a", "hit": True, "stage": "success", "scenario": "buying"},
            {"sample_id": "b", "hit": False, "stage": "paging", "scenario": "browsing"},
        ]}
        diff = paired_diff(control, candidate)
        self.assertEqual((diff["gained_count"], diff["lost_count"], diff["net_hits"]), (1, 1, 0))


class PhaseSevenDiagnosticsTest(unittest.TestCase):
    def test_agent_emits_signatures_stage_memberships_and_nonoverlapping_timings(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            catalog.write_text("\n".join(json.dumps({
                "parent_asin": str(index), "title": f"Blue cotton shirt {index}",
                "categories": ["Shirts"],
            }) for index in range(12)), encoding="utf-8")
            agent = Agent(catalog, Config(neural_rerank=False, evidence_ranking=True))
            try:
                agent.reset("phase7", {})
                agent.respond("phase7", "I need a blue cotton shirt", 1, 10)
                diagnostics = agent.last_diagnostics
                self.assertEqual(len(diagnostics["semantic_state_sha256"]), 64)
                self.assertEqual(len(diagnostics["retrieval_plan_sha256"]), 64)
                self.assertIn("admission_selected", diagnostics["stage_ids"])
                self.assertIn("returned_page", diagnostics["stage_ids"])
                expected = {
                    "parsing_and_state", "intent_planning", "retrieval", "pre_neural_ranking",
                    "admission", "neural", "post_neural_ranking", "policy",
                    "page_local_rerank", "response_assembly",
                }
                self.assertEqual(set(diagnostics["component_latency_seconds"]), expected)
                self.assertTrue(all(value >= 0 for value in diagnostics["component_latency_seconds"].values()))
            finally:
                agent.close()


if __name__ == "__main__":
    unittest.main()
