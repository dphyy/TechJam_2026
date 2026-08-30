import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.freeze import freeze_configs


class FreezeTest(unittest.TestCase):
    def test_freeze_binds_config_source_and_data_and_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, reserved, output = root / "config.json", root / "reserved", root / "freeze.json"
            config.write_text(json.dumps({"evidence_ranking": False}))
            reserved.write_text("opaque test data")
            with patch("experiments.freeze.source_hashes", return_value={"agent.py": "hash"}):
                value = freeze_configs([config], reserved, output, "development-only choice")
                self.assertEqual(value["source_hashes"], {"agent.py": "hash"})
                self.assertFalse(value["configs"][0]["evidence_ranking"])
                self.assertEqual(len(value["reserved_sha256"]), 64)
                with self.assertRaises(FileExistsError):
                    freeze_configs([config], reserved, output, "cannot tune again")
                with self.assertRaises(ValueError):
                    freeze_configs([config, config], reserved, root / "duplicate", "duplicate")
                with self.assertRaises(ValueError):
                    freeze_configs([config] * 3, reserved, root / "three", "too many")


if __name__ == "__main__":
    unittest.main()
