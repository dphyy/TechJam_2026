import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from experiments.cycle3_prepare import (
    build_pack,
    lock_pack,
    title_key,
    verify_lock,
)


def products(count: int = 380) -> list[dict]:
    return [
        {
            "parent_asin": f"P{index:04d}",
            "title": "cotton shirt " + chr(97 + index // 26) + chr(97 + index % 26),
            "categories": ["Clothing", "Shirts"],
            "features": ["cotton"],
        }
        for index in range(count)
    ]


def sample(identifier: str, target: str) -> dict:
    return {"sample_id": identifier, "ground_truth": {"parent_asin": target}}


class Cycle3PrepareTest(unittest.TestCase):
    def test_title_key_normalizes_loose_families(self):
        self.assertEqual(title_key("Example BLUE Shirt 42", loose=True), "example shirt")
        self.assertEqual(title_key("Example Red Shirt 10", loose=True), "example shirt")
        self.assertNotEqual(title_key("Example BLUE Shirt 42"), title_key("Example Red Shirt 10"))

    def test_deterministic_family_disjoint_pack_and_distributions(self):
        rows = products()
        rows += [
            {**rows[0], "parent_asin": "PUBLIC-FAMILY", "title": rows[0]["title"] + " blue 44"},
            {**rows[1], "parent_asin": "CONSUMED-FAMILY", "title": rows[1]["title"] + " red 12"},
            {**rows[2], "parent_asin": "ELIGIBLE-DUPLICATE", "title": rows[2]["title"] + " navy"},
        ]
        pack = build_pack(
            rows,
            [sample("public", "P0000")],
            [[sample("consumed", "P0001")]],
        )
        self.assertEqual(
            pack,
            build_pack(
                list(reversed(rows)),
                [sample("public", "P0000")],
                [[sample("consumed", "P0001")]],
            ),
        )
        by_id = {row["parent_asin"]: row for row in rows}
        all_rows = pack["screening"] + pack["confirmation"] + pack["validation"]
        targets = [row["ground_truth"]["parent_asin"] for row in all_rows]
        self.assertEqual(len(targets), 320)
        self.assertEqual(len(set(targets)), 320)
        self.assertNotIn("P0000", targets)
        self.assertNotIn("PUBLIC-FAMILY", targets)
        self.assertNotIn("P0001", targets)
        self.assertNotIn("CONSUMED-FAMILY", targets)
        self.assertEqual(len({title_key(by_id[target]["title"], loose=True) for target in targets}), 320)
        self.assertEqual(len({"P0002", "ELIGIBLE-DUPLICATE"} & set(targets)), 1)
        expected = {"buying": 64, "browsing": 64, "intent_override": 24, "boundary": 8}
        for split in ("screening", "confirmation", "validation"):
            expected_split = expected if split == "screening" else {
                key: value // 2 for key, value in expected.items()
            }
            self.assertEqual(
                Counter(row["scenario_type"] for row in pack[split]), expected_split
            )
            self.assertTrue(all(row["user_profile"] == {} for row in pack[split]))
            self.assertTrue(all("intent_card" not in row and "behavior" not in row for row in pack[split]))
        self.assertEqual(pack["audit"]["public_target_overlap"], 0)
        self.assertEqual(pack["audit"]["consumed_target_overlap"], 0)
        self.assertEqual(pack["audit"]["cross_split_loose_title_overlap"], 0)

    def test_all_consumed_datasets_are_inputs_and_bad_rows_fail(self):
        rows = products()
        with self.assertRaisesRegex(ValueError, "Duplicate consumed sample ID"):
            build_pack(
                rows,
                [sample("public", "P0000")],
                [[sample("same", "P0001")], [sample("same", "P0002")]],
            )
        with self.assertRaisesRegex(ValueError, "missing from catalog"):
            build_pack(rows, [sample("public", "P0000")], [[sample("other", "MISSING")]])

    def test_lock_is_idempotent_binds_every_input_and_hides_validation_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = root / "catalog.jsonl"
            public = root / "public.jsonl"
            consumed = root / "consumed.jsonl"
            catalog.write_text("".join(json.dumps(row) + "\n" for row in products()))
            public.write_text(json.dumps(sample("public", "P0000")) + "\n")
            consumed.write_text(json.dumps(sample("consumed", "P0001")) + "\n")
            output = root / "pack"
            first = lock_pack(catalog, public, [consumed], output)
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            self.assertEqual(first, lock_pack(catalog, public, [consumed], output))
            self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})
            self.assertEqual(first["counts"], {"screening": 160, "confirmation": 80, "validation": 80})
            self.assertFalse(first["validation_outcomes_accessed"])
            encoded_manifest = (output / "manifest.json").read_text()
            self.assertNotIn("validation_ids", first)
            for row in build_pack(products(), [sample("public", "P0000")], [[sample("consumed", "P0001")]])["validation"]:
                self.assertNotIn(row["ground_truth"]["parent_asin"], encoded_manifest)
            self.assertEqual(first["source_sha256"]["consumed_datasets"], {"consumed.jsonl": self._digest(consumed)})
            (output / "validation.jsonl").write_text("tampered\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                lock_pack(catalog, public, [consumed], output)

    def test_existing_unlocked_directory_seed_and_input_drift_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = root / "catalog.jsonl"
            public = root / "public.jsonl"
            consumed = root / "consumed.jsonl"
            catalog.write_text("".join(json.dumps(row) + "\n" for row in products()))
            public.write_text(json.dumps(sample("public", "P0000")) + "\n")
            consumed.write_text(json.dumps(sample("consumed", "P0001")) + "\n")
            output = root / "pack"
            output.mkdir()
            (output / "keep.txt").write_text("keep")
            with self.assertRaisesRegex(ValueError, "without a lock"):
                lock_pack(catalog, public, [consumed], output)
            self.assertEqual((output / "keep.txt").read_text(), "keep")
            (output / "keep.txt").unlink()
            output.rmdir()
            lock_pack(catalog, public, [consumed], output)
            with self.assertRaisesRegex(ValueError, "differs"):
                lock_pack(catalog, public, [consumed], output, seed="other-seed")
            with consumed.open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                lock_pack(catalog, public, [consumed], output)

    def test_read_only_verification_supports_commit_drift_with_preserved_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = root / "catalog.jsonl"
            public = root / "public.jsonl"
            consumed = root / "consumed.jsonl"
            snapshot = root / "cycle3_prepare-original.py"
            snapshot.write_bytes(Path("experiments/cycle3_prepare.py").read_bytes())
            catalog.write_text("".join(json.dumps(row) + "\n" for row in products()))
            public.write_text(json.dumps(sample("public", "P0000")) + "\n")
            consumed.write_text(json.dumps(sample("consumed", "P0001")) + "\n")
            output = root / "pack"
            manifest = lock_pack(catalog, public, [consumed], output)
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            with patch("experiments.cycle3_prepare.subprocess.check_output", return_value="f" * 40 + "\n"), patch(
                "experiments.cycle3_prepare.build_pack", side_effect=AssertionError("must not regenerate")
            ):
                receipt = verify_lock(catalog, public, [consumed], output, snapshot)
            self.assertTrue(receipt["verified"])
            self.assertTrue(receipt["commit_changed"])
            self.assertEqual(receipt["original_source_commit"], manifest["source_commit"])
            self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})

    def test_verification_rejects_tampered_data_inputs_and_snapshot(self):
        for changed, expected in (
            ("validation.jsonl", "validation data"),
            ("catalog.jsonl", "catalog source"),
            ("public.jsonl", "public_dataset source"),
            ("consumed.jsonl", "consumed_dataset source"),
            ("cycle3_prepare-original.py", "preparation_script source"),
        ):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                catalog = root / "catalog.jsonl"
                public = root / "public.jsonl"
                consumed = root / "consumed.jsonl"
                snapshot = root / "cycle3_prepare-original.py"
                snapshot.write_bytes(Path("experiments/cycle3_prepare.py").read_bytes())
                catalog.write_text("".join(json.dumps(row) + "\n" for row in products()))
                public.write_text(json.dumps(sample("public", "P0000")) + "\n")
                consumed.write_text(json.dumps(sample("consumed", "P0001")) + "\n")
                output = root / "pack"
                lock_pack(catalog, public, [consumed], output)
                manifest_bytes = (output / "manifest.json").read_bytes()
                target = output / changed if changed == "validation.jsonl" else root / changed
                with target.open("ab") as handle:
                    handle.write(b"\n")
                with self.assertRaisesRegex(ValueError, expected):
                    verify_lock(catalog, public, [consumed], output, snapshot)
                self.assertEqual((output / "manifest.json").read_bytes(), manifest_bytes)

    @staticmethod
    def _digest(path: Path) -> str:
        import hashlib

        return hashlib.sha256(path.read_bytes()).hexdigest()
