from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.private_like_validate import DEFAULT_DATASET, run_private_like


class PrivateLikeValidationTest(unittest.TestCase):
    def test_default_fixture_covers_declared_risk_groups(self) -> None:
        pack = json.loads(DEFAULT_DATASET.read_text(encoding="utf-8"))
        groups = {case["group"] for case in pack["cases"]}
        self.assertEqual(pack["schema"], "cycle2-capability-fixtures-v1")
        self.assertTrue({
            "ordinary vague query", "intent override", "explicit negation", "no preference",
            "alternatives", "body versus component", "accessory versus primary object",
            "sparse metadata", "negative feedback",
        }.issubset(groups))
        self.assertTrue(all(any(turn["assertions"] for turn in case["turns"]) for case in pack["cases"]))

    def test_runner_uses_private_like_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            config = Path(directory) / "config.json"
            config.write_text("{}\n", encoding="utf-8")
            with patch("experiments.private_like_validate.run_capabilities", return_value={"ok": True}) as runner:
                self.assertEqual(run_private_like(config, output), {"ok": True})
            args = runner.call_args.args
            kwargs = runner.call_args.kwargs
            self.assertEqual(args[:4], (DEFAULT_DATASET, config, output, "development"))
            self.assertEqual(kwargs["provenance"]["schema"], "private-like-engineering-validation-v1")


if __name__ == "__main__":
    unittest.main()
