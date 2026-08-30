import json
import tempfile
import unittest
from pathlib import Path

from experiments.robustness_matrix_prepare import (
    SPLIT_COUNTS,
    build_matrix,
    lock_matrix,
    metadata_strata,
    record_consumption,
    verify_lock,
)


def product(index: int) -> dict:
    value = index
    letters = ""
    while True:
        letters = chr(ord("a") + value % 26) + letters
        value = value // 26 - 1
        if value < 0:
            break
    return {
        "parent_asin": f"P{index:04d}",
        "title": f"Distinct {letters} tote",
        "categories": ["Root", f"Category {index}"],
        "features": ["feature", f"attribute {index}"],
        "details": {"Department": "unisex"},
        "description": [f"Description {index}"],
        "price": 10 + index % 50,
        "rating_number": index % 300,
    }


def target(identifier: str, sample: str) -> dict:
    return {
        "sample_id": sample,
        "scenario_type": "buying",
        "ground_truth": {"parent_asin": identifier},
        "user_profile": {},
    }


class RobustnessMatrixPrepareTest(unittest.TestCase):
    def setUp(self):
        self.products = [product(index) for index in range(900)]
        self.public = [target("P0000", "public-1")]
        self.consumed = [[target("P0001", "consumed-1")]]

    def test_matrix_is_group_disjoint_and_deterministic(self):
        first = build_matrix(self.products, self.public, self.consumed, "fixed-seed")
        second = build_matrix(self.products, self.public, self.consumed, "fixed-seed")
        self.assertEqual(first, second)
        self.assertEqual(first["audit"]["counts"], SPLIT_COUNTS)
        self.assertEqual(first["audit"]["cross_split_target_overlap"], 0)
        self.assertFalse(any(first["audit"]["cross_split_group_overlap"].values()))
        all_targets = {
            row["ground_truth"]["parent_asin"]
            for rows in first["datasets"].values()
            for row in rows
        }
        self.assertNotIn("P0000", all_targets)
        self.assertNotIn("P0001", all_targets)

    def test_metadata_strata_preserve_unknown_and_detect_conflicts(self):
        sparse = {"title": "Tiny hat", "categories": ["Women", "Hats"], "price": None}
        self.assertTrue({"missing_price", "short_title", "sparse_features"} <= set(metadata_strata(sparse, 1)))
        conflict = {
            "title": "Mens leather belt",
            "categories": ["Women", "Shoes"],
            "details": {"Department": "mens"},
            "features": ["leather"],
            "price": 12,
        }
        self.assertIn("contradictory_fields", metadata_strata(conflict, 2))
        self.assertIn("near_duplicate_document", metadata_strata(conflict, 2))

    def test_lock_is_idempotent_and_tamper_evident(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            public = root / "public.jsonl"
            consumed = root / "consumed.jsonl"
            output = root / "matrix"
            catalog.write_text("".join(json.dumps(row) + "\n" for row in self.products), encoding="utf-8")
            public.write_text(json.dumps(self.public[0]) + "\n", encoding="utf-8")
            consumed.write_text(json.dumps(self.consumed[0][0]) + "\n", encoding="utf-8")
            initial = lock_matrix(catalog, public, [consumed], output, "fixed-seed")
            self.assertEqual(initial, lock_matrix(catalog, public, [consumed], output, "fixed-seed"))
            self.assertTrue(verify_lock(catalog, public, [consumed], output)["verified"])
            with self.assertRaisesRegex(ValueError, "before screening"):
                record_consumption(output, "confirmation", "too early")
            screening = record_consumption(output, "screening", "frozen candidate", None)
            self.assertEqual(screening["entries"]["screening"]["status"], "consumed")
            confirmation = record_consumption(output, "confirmation", "passing finalist", None)
            self.assertEqual(confirmation["entries"]["confirmation"]["status"], "consumed")
            with (output / "screening.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(ValueError, "hash drift"):
                verify_lock(catalog, public, [consumed], output)


if __name__ == "__main__":
    unittest.main()
