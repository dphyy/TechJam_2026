from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.evaluate_suite import SuiteSpec, evaluate_suite, markdown_report, parse_config_spec, write_report


class EvaluateSuiteTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        catalog = root / "catalog.jsonl"
        dataset = root / "dataset.jsonl"
        rows = [
            {"parent_asin": "A", "title": "Blue cotton running shoe", "categories": ["Clothing", "Shoes"],
             "features": ["cotton", "running"], "details": {"color": "blue"}},
            {"parent_asin": "B", "title": "Black leather boot", "categories": ["Clothing", "Boots"],
             "features": ["leather"], "details": {"color": "black"}},
        ]
        samples = [{"sample_id": "s1", "scenario_type": "buying", "user_profile": {},
                    "ground_truth": {"parent_asin": "A"}}]
        catalog.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        dataset.write_text("".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8")
        return catalog, dataset

    def test_parse_candidate_spec_requires_safe_name(self) -> None:
        spec = parse_config_spec("candidate=configs/selected.json")
        self.assertEqual(spec.name, "candidate")
        self.assertEqual(spec.kind, "config")
        with self.assertRaises(ValueError):
            parse_config_spec("../bad=configs/selected.json")
        with self.assertRaises(ValueError):
            parse_config_spec("missing-separator")

    def test_suite_report_contains_metrics_latency_fallbacks_and_scenarios(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog, dataset = self._fixture(root)
            report = evaluate_suite([SuiteSpec("baseline", "baseline")], catalog, dataset)
            self.assertEqual(report["schema"], "mercury-evaluation-suite-v1")
            metrics = report["runs"][0]["metrics"]
            for key in ("hit_rate_at_10", "mrr", "mttc", "technical_score",
                        "latency_p50_seconds", "latency_p95_seconds",
                        "fallback_turns", "token_usage", "scenario_metrics"):
                self.assertIn(key, metrics)
            self.assertIn("buying", metrics["scenario_metrics"])
            self.assertIn("suite_max_rss_bytes", report)
            self.assertIn("baseline", markdown_report(report))

    def test_write_report_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog, dataset = self._fixture(root)
            report = evaluate_suite([SuiteSpec("baseline", "baseline")], catalog, dataset)
            output = root / "report"
            write_report(report, output)
            self.assertTrue((output / "report.json").is_file())
            self.assertTrue((output / "report.md").is_file())
            with self.assertRaises(FileExistsError):
                write_report(report, output)

    def test_write_report_cleans_up_after_a_write_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report"
            report = {
                "catalog_sha256": "catalog",
                "dataset_sha256": "dataset",
                "source_changed_during_run": False,
                "runs": [],
                "interpretation": "fixture",
            }
            with patch.object(Path, "write_text", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    write_report(report, output)
            self.assertFalse(output.exists())

    def test_evaluation_closes_an_agent_after_failure(self) -> None:
        class ClosableAgent:
            closed = False

            def close(self) -> None:
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog, dataset = self._fixture(root)
            agent = ClosableAgent()
            with patch("experiments.evaluate_suite._agent_for", return_value=(agent, None)), \
                    patch("experiments.evaluate_suite.evaluate", side_effect=RuntimeError("failed")):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    evaluate_suite([SuiteSpec("baseline", "baseline")], catalog, dataset)
            self.assertTrue(agent.closed)

    def test_evaluation_rejects_changed_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog, dataset = self._fixture(root)
            with patch("experiments.evaluate_suite.file_sha256", side_effect=("catalog-before", "dataset-before", "catalog-after", "dataset-after")):
                with self.assertRaisesRegex(RuntimeError, "changed during evaluation"):
                    evaluate_suite([SuiteSpec("baseline", "baseline")], catalog, dataset)

    def test_duplicate_specs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog, dataset = self._fixture(root)
            with self.assertRaises(ValueError):
                evaluate_suite([SuiteSpec("same", "baseline"), SuiteSpec("same", "baseline")], catalog, dataset)


if __name__ == "__main__":
    unittest.main()
