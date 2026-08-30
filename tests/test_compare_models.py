import json
import tempfile
import unittest
from pathlib import Path

from experiments.compare_models import compare_model_runs


class ModelComparisonTests(unittest.TestCase):
    def write_run(self, root, name, *, pairs=30, dataset="same", source="same", weight=.75):
        run = root / name
        run.mkdir()
        session = {"sample_id": "s", "hit": True, "reciprocal_rank": 1.0, "first_hit_turn": 1}
        (run / "result.json").write_text(json.dumps({"sessions": [session], "reported_token_usage": {}}))
        (run / "manifest.json").write_text(json.dumps({
            "dataset_sha256": dataset, "source_hashes": {"agent.py": source},
            "config": {"artifact_dir": str(root / name / "assets"), "neural_weight": weight},
            "max_rss_bytes": 100,
        }))
        (run / "diagnostics.json").write_text(json.dumps({"p95_seconds": .1}))
        (run / "traces.json").write_text(json.dumps([[
            {"diagnostics": {"neural_scores": {"scored_pairs": pairs}}},
        ]]))
        return run

    def test_comparison_enforces_same_source_data_config_and_pair_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = self.write_run(root, "control")
            candidate = self.write_run(root, "candidate", pairs=60)
            report = compare_model_runs(control, candidate)
            self.assertEqual(report["max_observed_pairs"], {"control": 30, "candidate": 60})
            excessive = self.write_run(root, "excessive", pairs=61)
            with self.assertRaisesRegex(ValueError, "pair cap"):
                compare_model_runs(control, excessive)
            changed = self.write_run(root, "changed", weight=.5)
            with self.assertRaisesRegex(ValueError, "Only the local model"):
                compare_model_runs(control, changed)


if __name__ == "__main__":
    unittest.main()
