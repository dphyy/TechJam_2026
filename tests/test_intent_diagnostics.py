import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.intent_dataset import load_authored, prepare, validate_splits
from experiments.intent_diagnostics import STRUCTURAL_FEATURES, evaluate_sealed, fit, structural_vector
from experiments.tune_intent_rules import verify_protocol_inputs


class _FakeEncoder:
    def encode(self, texts, **_kwargs):
        markers = (("need", "must", "find"), ("ideas", "explore", "browse"),
                   ("maybe", "open", "though"), ("actually", "change", "forget"))
        rows = []
        for text in texts:
            lowered = text.lower()
            vector = np.asarray([sum(lowered.count(word) for word in words) for words in markers]
                                + [len(lowered.split()) / 20], dtype=np.float64)
            norm = np.linalg.norm(vector)
            rows.append(vector / norm if norm else vector)
        return np.asarray(rows)


class IntentDatasetTests(unittest.TestCase):
    def test_authored_source_and_exact_grouped_split(self):
        source = Path("data/intent_authored_v1.json")
        _, groups = load_authored(source)
        self.assertEqual(len(groups), 60)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "split"
            manifest = prepare(source, root)
            self.assertEqual(manifest["summary"]["train"]["rows"], 84)
            self.assertEqual(manifest["summary"]["validation"]["rows"], 18)
            self.assertEqual(manifest["summary"]["sealed_test"]["rows"], 18)
            splits = {
                name: [json.loads(line) for line in (root / filename).read_text().splitlines()]
                for name, filename in manifest["files"].items()
            }
            validate_splits(splits)
            with self.assertRaises(FileExistsError):
                prepare(source, root)

    def test_split_validation_rejects_group_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "split"
            manifest = prepare(Path("data/intent_authored_v1.json"), root)
            splits = {
                name: [json.loads(line) for line in (root / filename).read_text().splitlines()]
                for name, filename in manifest["files"].items()
            }
            leaked = dict(splits["train"][0])
            leaked["sample_id"] = "new_sample"
            splits["validation"].append(leaked)
            with self.assertRaisesRegex(ValueError, "allocation|leakage"):
                validate_splits(splits)

    def test_rule_tuner_accepts_only_manifested_train_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "split"
            prepare(Path("data/intent_authored_v1.json"), root)
            manifest = verify_protocol_inputs(root / "train.jsonl", root / "validation.jsonl")
            self.assertEqual(manifest["protocol"], "intent-authored-v1")
            with self.assertRaisesRegex(ValueError, "manifested validation"):
                verify_protocol_inputs(root / "train.jsonl", root / "sealed-test.jsonl")
            (root / "validation.jsonl").write_text("tampered\n")
            with self.assertRaisesRegex(ValueError, "frozen manifest hash"):
                verify_protocol_inputs(root / "train.jsonl", root / "validation.jsonl")


class IntentDiagnosticTests(unittest.TestCase):
    def test_structural_features_are_fixed_and_finite(self):
        row = {"history": ["I wanted a red dress."], "message": "Actually make it a blue cotton dress."}
        values = structural_vector(row)
        self.assertEqual(values.shape, (len(STRUCTURAL_FEATURES),))
        self.assertTrue(np.isfinite(values).all())
        self.assertGreater(values[STRUCTURAL_FEATURES.index("known_object")], 0)
        self.assertGreater(values[STRUCTURAL_FEATURES.index("preference_removed_count")], 0)

    def test_fit_freezes_all_baselines_and_sealed_test_is_single_use(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            data_root = temporary / "data"
            prepare(Path("data/intent_authored_v1.json"), data_root)
            fit_root = temporary / "fit"
            freeze = fit(data_root, fit_root, temporary / "unused-model", encoder=_FakeEncoder())
            self.assertEqual(set(freeze["models"]),
                             {"rules_only", "structural", "semantic_linear", "hybrid_linear"})
            self.assertTrue(all("macro_f1" in report for report in freeze["validation_reports"].values()))
            report = evaluate_sealed(fit_root / "model-freeze.json", temporary / "test-report",
                                     temporary / "unused-model", encoder=_FakeEncoder())
            self.assertEqual(report["count"], 18)
            self.assertFalse(report["runtime_routing_changed"])
            self.assertTrue((data_root / "sealed-test-consumed.json").is_file())
            with self.assertRaises(FileExistsError):
                evaluate_sealed(fit_root / "model-freeze.json", temporary / "second-report",
                                temporary / "unused-model", encoder=_FakeEncoder())
            self.assertFalse((temporary / "second-report").exists())


if __name__ == "__main__":
    unittest.main()
