from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.normalize_consumed_ids import normalize


class NormalizeConsumedIdsTest(unittest.TestCase):
    def test_prefixes_only_sample_ids_and_records_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            row = {"sample_id": "shared", "ground_truth": {"parent_asin": "P1"}}
            first.write_text(json.dumps(row) + "\n", encoding="utf-8")
            second.write_text(json.dumps(row) + "\n", encoding="utf-8")
            output = root / "normalized"
            manifest = normalize([first, second], output)
            left = json.loads((output / "first.jsonl").read_text())
            right = json.loads((output / "second.jsonl").read_text())
            self.assertEqual(left["sample_id"], "first__shared")
            self.assertEqual(right["sample_id"], "second__shared")
            self.assertEqual(left["ground_truth"], row["ground_truth"])
            self.assertEqual(manifest["files"]["first.jsonl"]["row_count"], 1)

    def test_refuses_to_overwrite_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            source.write_text('{"sample_id":"one"}\n', encoding="utf-8")
            output = root / "output"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                normalize([source], output)


if __name__ == "__main__":
    unittest.main()
