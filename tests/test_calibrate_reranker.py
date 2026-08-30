import json
import tempfile
import unittest
from pathlib import Path

from experiments.calibrate_reranker import calibrate_run, fit_platt, grouped_calibration
from mercury.model_assets import file_sha256


class RerankerCalibrationTests(unittest.TestCase):
    def test_grouped_calibration_is_deterministic_and_finite(self):
        records = [{"group": f"g{i}", "margin": (i % 5) / 4, "label": int(i % 3 != 0)}
                   for i in range(30)]
        first = grouped_calibration(records, 3)
        second = grouped_calibration(records, 3)
        self.assertEqual(first, second)
        self.assertEqual(first["out_of_fold_metrics"]["count"], 30)
        self.assertGreaterEqual(first["out_of_fold_metrics"]["expected_calibration_error"], 0)

    def test_invalid_calibration_data_is_rejected(self):
        for records in ([], [{"group": "a", "margin": -1, "label": 1}],
                        [{"group": "a", "margin": 1, "label": 2}]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                fit_platt(records)

    def test_run_extraction_writes_create_only_calibration_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            samples, traces = [], []
            for index in range(10):
                target = f"T{index}"
                samples.append({"sample_id": f"s{index}", "user_group_id": f"u{index}",
                                "ground_truth": {"parent_asin": target}})
                recommendations = [{"parent_asin": target}] if index % 2 else [{"parent_asin": "other"}]
                traces.append([{"turn": 1, "response": {"recommendations": recommendations},
                                "diagnostics": {"neural_scores": {"logit_margin": index / 10}}}])
            dataset = root / "dataset.jsonl"
            dataset.write_text("".join(json.dumps(sample) + "\n" for sample in samples))
            (run / "traces.json").write_text(json.dumps(traces))
            (run / "manifest.json").write_text(json.dumps({
                "dataset_sha256": file_sha256(dataset), "reserved_evaluation": False,
            }))
            output = root / "calibration"
            report = calibrate_run(run, dataset, output, folds=2)
            self.assertEqual(report["record_count"], 10)
            self.assertTrue((output / "calibration.json").is_file())
            with self.assertRaises(FileExistsError):
                calibrate_run(run, dataset, output, folds=2)

    def test_calibration_rejects_mismatched_or_reserved_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            dataset = root / "dataset.jsonl"
            dataset.write_text('{"sample_id":"s","ground_truth":{"parent_asin":"T"}}\n')
            (run / "traces.json").write_text("[]")
            manifest = run / "manifest.json"
            manifest.write_text(json.dumps({"dataset_sha256": "wrong", "reserved_evaluation": False}))
            with self.assertRaisesRegex(ValueError, "does not match"):
                calibrate_run(run, dataset, root / "mismatch", folds=2)
            manifest.write_text(json.dumps({
                "dataset_sha256": file_sha256(dataset), "reserved_evaluation": True,
            }))
            with self.assertRaisesRegex(ValueError, "Reserved"):
                calibrate_run(run, dataset, root / "reserved", folds=2)


if __name__ == "__main__":
    unittest.main()
