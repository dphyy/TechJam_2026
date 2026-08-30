import json
import unittest
from pathlib import Path

from experiments.metamorphic_validate import evaluate_pack, validate_pack
from mercury.config import Config


ROOT = Path(__file__).resolve().parents[1]


class MetamorphicValidateTest(unittest.TestCase):
    def test_authored_pack_is_valid_and_runs_legally(self):
        pack = json.loads((ROOT / "data/metamorphic_robustness_v1.json").read_text(encoding="utf-8"))
        self.assertIs(validate_pack(pack), pack)
        result = evaluate_pack(pack, Config(neural_rerank=False, evidence_ranking=True))
        self.assertEqual(result["case_count"], 7)
        legal = [
            check["passed"]
            for case in result["cases"]
            for check in case["checks"]
            if check["property"] == "legal_output"
        ]
        self.assertTrue(legal)
        self.assertTrue(all(legal))

    def test_rejects_unknown_property(self):
        pack = {
            "schema": "mercury-metamorphic-dialogues-v1",
            "cases": [{
                "id": "bad",
                "catalog": [{"parent_asin": "A", "title": "Hat", "categories": ["Hats"]}],
                "variants": [["A hat"], ["Hat please"]],
                "properties": ["target_specific_answer"],
            }],
        }
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            validate_pack(pack)


if __name__ == "__main__":
    unittest.main()
