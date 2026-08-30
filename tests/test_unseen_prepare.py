import json
import tempfile
import unittest
from pathlib import Path

from experiments.unseen_prepare import prepare, validate_splits
from mercury.model_assets import file_sha256


class UnseenPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.catalog = self.root / "catalog.jsonl"
        categories = ["Shirts", "Shoes", "Bags", "Jewelry", "Hats"]
        with self.catalog.open("x") as handle:
            for index in range(70):
                row = {"parent_asin": f"P{index:03}", "title": f"Product {index}",
                       "categories": [categories[index % len(categories)]], "features": [f"feature {index}"]}
                handle.write(json.dumps(row) + "\n")
        self.excluded = self.root / "public.jsonl"
        self.excluded.write_text(json.dumps({"ground_truth": {"parent_asin": "P000"}}) + "\n")

    def test_prepare_is_deterministic_disjoint_and_create_only(self):
        first = self.root / "first"
        second = self.root / "second"
        manifest = prepare(self.catalog, self.excluded, first, 20, 20, 7)
        prepare(self.catalog, self.excluded, second, 20, 20, 7)
        self.assertEqual(file_sha256(first / "development.jsonl"), file_sha256(second / "development.jsonl"))
        self.assertEqual(file_sha256(first / "final-sealed.jsonl"), file_sha256(second / "final-sealed.jsonl"))
        self.assertEqual(manifest["development_scenarios"],
                         {"boundary": 1, "browsing": 8, "buying": 8, "intent_override": 3})
        development = [json.loads(line) for line in (first / "development.jsonl").read_text().splitlines()]
        final = [json.loads(line) for line in (first / "final-sealed.jsonl").read_text().splitlines()]
        self.assertNotIn("P000", {row["ground_truth"]["parent_asin"] for row in development + final})
        self.assertTrue(all(row["difficulty_bucket"] == "unseen" for row in development + final))
        validate_splits(development, final, {"P000"})
        with self.assertRaises(FileExistsError):
            prepare(self.catalog, self.excluded, first, 20, 20, 7)

    def test_validation_rejects_target_and_user_leakage(self):
        def sample(name, user, target, scenario):
            return {
                "sample_id": name, "user_group_id": user, "scenario_type": scenario,
                "category_bucket": "clothing", "ground_truth": {"parent_asin": target},
            }
        scenarios = ["buying"] * 8 + ["browsing"] * 8 + ["intent_override"] * 3 + ["boundary"]
        development = [sample(f"d{i}", f"du{i}", f"dt{i}", scenario) for i, scenario in enumerate(scenarios)]
        final = [sample(f"f{i}", f"fu{i}", f"ft{i}", scenario) for i, scenario in enumerate(scenarios)]
        validate_splits(development, final, set())
        final[0]["ground_truth"]["parent_asin"] = "dt0"
        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_splits(development, final, set())


if __name__ == "__main__":
    unittest.main()
