import json
import tempfile
import unittest
from pathlib import Path

from experiments.dataset_status import audit_dataset
from mercury.model_assets import file_sha256


class DatasetStatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "final-sealed.jsonl"
        self.dataset.write_text('{"sample_id": "example"}\n')
        self.digest = file_sha256(self.dataset)
        self.receipts = self.root / "runs"
        self.receipts.mkdir()

    def write(self, name, value):
        path = self.receipts / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def audit(self, dataset=None):
        return audit_dataset(dataset or self.dataset, [self.receipts])

    def test_consumed_marker_overrides_stale_manifest_and_survives_rename(self):
        self.write("manifest.json", {"reserved_status": "sealed", "reserved_sha256": self.digest})
        self.write("run/reserved-consumed.json", {"reserved_sha256": self.digest})
        renamed = self.dataset.with_name("fresh-validation.jsonl")
        renamed.write_bytes(self.dataset.read_bytes())
        report = self.audit(renamed)
        self.assertEqual(report["status"], "consumed")
        self.assertEqual(len(report["evidence"]), 1)
        self.assertFalse(report["untouched_holdout_verified"])

    def test_completed_old_pipeline_run_counts_for_new_pipeline(self):
        self.write("old/report.json", {"dataset_sha256": self.digest,
                                      "runs": [{"name": "neural", "metrics": {
                                          "sample_count": 40, "hit_rate_at_10": 0.9,
                                      }}]})
        self.assertEqual(self.audit()["status"], "consumed")

    def test_submission_report_is_recognized_without_dataset_path(self):
        self.write("selected-report.json", {"dataset_sha256": self.digest,
                                            "result": {"sessions": [{}]}})
        self.assertEqual(self.audit()["status"], "consumed")

    def test_empty_outcome_placeholder_does_not_claim_completion(self):
        self.write("report.json", {"dataset_sha256": self.digest, "sessions": []})
        self.assertEqual(self.audit()["status"], "unknown")

    def test_interrupted_registration_cannot_be_reported_as_untouched_or_completed(self):
        self.write("failed/registration.json", {"dataset_sha256": self.digest})
        report = self.audit()
        self.assertEqual(report["status"], "attempt_recorded")
        self.assertFalse(report["evidence"][0]["completed"])
        self.assertFalse(report["untouched_holdout_verified"])

    def test_name_and_preparation_metadata_do_not_prove_unseen_or_consumed(self):
        self.write("manifest.json", {"dataset_sha256": self.digest,
                                     "validation_outcomes_accessed": False})
        self.write("freeze-report.json", {"reserved_sha256": self.digest})
        report = self.audit()
        self.assertEqual(report["status"], "unknown")
        self.assertEqual(report["evidence"], [])

    def test_same_path_with_different_bytes_does_not_reuse_old_receipt(self):
        self.write("old/report.json", {"dataset": str(self.dataset),
                                      "dataset_sha256": "0" * 64, "sessions": [{}]})
        self.assertEqual(self.audit()["status"], "unknown")

    def test_ledger_events_override_stale_sealed_status(self):
        self.write("consumption-ledger.json", {"entries": {"final": {
            "status": "sealed", "events": [{"dataset_sha256": self.digest}],
        }}})
        self.assertEqual(self.audit()["status"], "consumed")

    def test_nested_manifest_hash_is_not_evaluation_evidence(self):
        self.write("report.json", {"source_hashes": {"dataset_sha256": self.digest}, "sessions": [{}]})
        self.assertEqual(self.audit()["status"], "unknown")

    def test_bad_receipt_or_missing_root_cannot_produce_clean_audit(self):
        path = self.write("report.json", {})
        path.write_text('{"dataset_sha256":')
        report = audit_dataset(self.dataset, [self.receipts, self.root / "missing"])
        self.assertEqual(report["status"], "unknown")
        self.assertFalse(report["scan_complete"])
        self.assertEqual(len(report["warnings"]), 2)

    def test_overlapping_roots_are_deduplicated_and_audit_is_read_only(self):
        path = self.write("old/report.json", {"dataset_sha256": self.digest, "sessions": [{}]})
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        report = audit_dataset(self.dataset, [self.receipts, path.parent])
        self.assertEqual(len(report["evidence"]), 1)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
