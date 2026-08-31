from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import agent as public_entrypoint
from experiments.submission_evaluate import (
    ObservedAgent, _aggregate, _public_agent, evaluate_submission, source_receipt, verified_metrics,
)
from mercury.agent import Agent as PreviousAgent
from mercury.model_assets import file_sha256


class SubmissionEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog, self.dataset = self.root / "catalog.jsonl", self.root / "dataset.jsonl"
        rows = [{"parent_asin": "A", "categories": ["shirts"], "title": "Blue cotton shirt",
                 "features": ["cotton", "blue"]},
                {"parent_asin": "B", "categories": ["shirts"], "title": "Red wool shirt",
                 "features": ["wool", "red"]}]
        self.catalog.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.dataset.write_text(json.dumps({"sample_id": "do-not-export-this-session",
            "scenario_type": "buying", "user_profile": {}, "ground_truth": {"parent_asin": "A"}}) + "\n")

    def evaluate(self, name="evaluation", full_width=False):
        output = self.root / name
        report = evaluate_submission(self.catalog, self.dataset, output, full_width=full_width)
        return report, output / "report.json"

    def test_actual_public_default_and_raw_control_are_separately_bound(self):
        report, path = self.evaluate()
        self.assertTrue(report["valid"])
        self.assertEqual(report["result"]["sample_count"], 1)
        self.assertEqual(report["measurement"]["errors"], 0)
        self.assertEqual(report["measurement"]["startup_fallbacks"], [])
        certified = verified_metrics(path, file_sha256(path), file_sha256(self.catalog))
        self.assertEqual(certified["metrics"]["hit_rate_at_10"], 1.0)
        self.assertNotIn("do-not-export-this-session", json.dumps(certified))
        self.assertIn("agent.py", report["source_hashes"])
        self.assertIn("starter/agent.py", report["source_hashes"])
        full, full_path = self.evaluate("raw", full_width=True)
        self.assertTrue(full["valid"])
        self.assertEqual(set(full["measurement"]["widths"]), {2})
        with self.assertRaisesRegex(ValueError, "identity"):
            verified_metrics(full_path, file_sha256(full_path), file_sha256(self.catalog))

    def test_public_import_cannot_silently_evaluate_a_different_backend(self):
        with patch.object(public_entrypoint, "Agent", PreviousAgent):
            with self.assertRaisesRegex(ValueError, "entry point"):
                _public_agent()

    def test_full_width_reports_retrieval_shortfall_without_unrelated_padding(self):
        rows = [{"parent_asin": "A", "title": "Cotton shirt", "categories": ["shirts"]}]
        rows.extend({"parent_asin": f"hammer{index}", "title": "Steel hammer", "categories": ["tools"]}
                    for index in range(11))
        self.catalog.write_text("".join(json.dumps(row) + "\n" for row in rows))
        report, _ = self.evaluate(full_width=True)
        self.assertTrue(report["valid"])
        self.assertEqual(report["measurement"]["widths"], {1: 1})
        self.assertEqual(report["measurement"]["candidate_shortfall_turns"], 1)

    def test_safe_index_fallback_is_recorded_but_cannot_certify_healthy_execution(self):
        from mercury.lexical.constraint_index import default_catalog_index_path

        default_catalog_index_path(self.catalog).write_bytes(b"invalid catalog index")
        output = self.root / "fallback"
        with self.assertRaisesRegex(RuntimeError, "verification failed"):
            evaluate_submission(self.catalog, self.dataset, output)
        report = json.loads((output / "report.json").read_text())
        self.assertFalse(report["valid"])
        self.assertEqual(report["measurement"]["errors"], 0)
        self.assertIn("catalog_index_rebuilt", report["measurement"]["startup_fallbacks"])

    def test_report_tampering_and_stale_runtime_are_rejected(self):
        report, path = self.evaluate()
        digest = file_sha256(path)
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "checksum"):
            verified_metrics(path, digest, file_sha256(self.catalog))
        edits = []
        for key, value in (("valid", 1), ("catalog_sha256", "bad"), ("mode", "full_width")):
            item = deepcopy(report)
            item[key] = value
            edits.append(item)
        for key, value in (("errors", False), ("fallback_turns", 1), ("source_changed", "false"),
                           ("startup_fallbacks", ["missing"])):
            item = deepcopy(report)
            item["measurement"][key] = value
            edits.append(item)
        for key in ("agent.py", "starter/agent.py", "mercury/lexical/retrieval.py"):
            item = deepcopy(report)
            item["source_hashes"][key] = "bad"
            edits.append(item)
        item = deepcopy(report)
        item["result"]["recommended_technical_score"] = .123
        edits.append(item)
        item = deepcopy(report)
        item["result"]["sessions"][0]["reciprocal_rank"] = .125
        edits.append(item)
        for index, item in enumerate(edits):
            with self.subTest(index=index):
                path.write_text(json.dumps(item))
                with self.assertRaises(ValueError):
                    verified_metrics(path, file_sha256(path), file_sha256(self.catalog))

    def test_invalid_responses_are_not_hidden_by_official_exception_handling(self):
        output = self.root / "failed"
        with patch.object(public_entrypoint.Agent, "respond", side_effect=RuntimeError("transient")):
            with self.assertRaisesRegex(RuntimeError, "verification failed"):
                evaluate_submission(self.catalog, self.dataset, output)
        report = json.loads((output / "report.json").read_text())
        self.assertFalse(report["valid"])
        self.assertEqual(report["measurement"]["errors"], 10)

    def test_changed_sources_cannot_certify_a_run(self):
        original = source_receipt()
        changed = {**original, "agent.py": "changed"}
        with patch("experiments.submission_evaluate.source_receipt", side_effect=(original, changed)):
            with self.assertRaisesRegex(RuntimeError, "verification failed"):
                evaluate_submission(self.catalog, self.dataset, self.root / "changed")

    def test_create_only_and_empty_dataset(self):
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "keep"
        sentinel.write_text("untouched")
        with self.assertRaises(FileExistsError):
            evaluate_submission(self.catalog, self.dataset, output)
        self.assertEqual(sentinel.read_text(), "untouched")
        self.dataset.write_text("")
        with self.assertRaisesRegex(ValueError, "empty"):
            evaluate_submission(self.catalog, self.dataset, self.root / "empty")
        self.assertFalse((self.root / "empty").exists())

    def test_observer_rejects_illegal_ids_and_false_width_receipts(self):
        good = {"message": "Choose a shirt", "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "recommendations": [{"parent_asin": "A", "score": 1.0}]}
        diagnostics = {"state_committed": True,
                       "stage_receipts": {"ranked_prefix": {"available": True, "complete": True,
                                                             "count": 1, "ids": ["A"]},
                                          "question_context": {"available": True, "count": 1}}}
        for response, receipt in (
            ({**good, "recommendations": [{"parent_asin": "outside", "score": 1.0}]}, diagnostics),
            ({**good, "recommendations": [{"parent_asin": "A", "score": float("nan")}]}, diagnostics),
            (good, {**diagnostics, "stage_receipts": {"ranked_prefix": {"available": True, "ids": ["B"]}}}),
            (good, {"state_committed": True}),
            ({**good, "ask_attribute": ["color"]}, diagnostics),
            ({**good, "usage": {"prompt_tokens": False, "completion_tokens": 0}}, diagnostics),
        ):
            inner = SimpleNamespace(reset=lambda *args: None, respond=lambda *args: response,
                                    last_diagnostics=receipt)
            observed = ObservedAgent(inner, {"A"}, True)
            observed.reset("s", {})
            with self.assertRaises(ValueError):
                observed.respond("s", "shirt", 1, 10)
            self.assertEqual(observed.errors, 1)

    def test_aggregate_rejects_invalid_outcomes(self):
        row = {"sample_id": "s", "hit": True, "best_rank": 2, "first_hit_turn": 2, "reciprocal_rank": .5}
        self.assertEqual(_aggregate([row])["recommended_technical_score"], .83)
        for rows in ([], [row, row], [{**row, "best_rank": True}], [{**row, "hit": 1}],
                     [{**row, "first_hit_turn": 0}], [{**row, "reciprocal_rank": float("nan")} ]):
            with self.assertRaises(ValueError):
                _aggregate(rows)


if __name__ == "__main__":
    unittest.main()
