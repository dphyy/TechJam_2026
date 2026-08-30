import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from experiments.cycle2_prepare import build_pack, lock_pack, title_key, verify_lock


def products(count=90):
    return [{"parent_asin": f"P{i:04d}",
             "title": "cotton shirt " + chr(97 + i // 26) + chr(97 + i % 26),
             "categories": ["Clothing", "Shirts"], "features": ["cotton"]}
            for i in range(count)]


def old_samples():
    return [{"sample_id": "old-one", "ground_truth": {"parent_asin": "P0000"}}]


class Cycle2PrepareTest(unittest.TestCase):
    def test_fresh_cli_generation_then_default_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = root / "catalog.jsonl"
            old = root / "old.jsonl"
            catalog.write_text("".join(json.dumps(row) + "\n" for row in products()))
            old.write_text("".join(json.dumps(row) + "\n" for row in old_samples()))
            output = root / "pack"
            command = [sys.executable, "-m", "experiments.cycle2_prepare",
                       "--catalog", str(catalog), "--old-dataset", str(old), "--output", str(output)]
            subprocess.run(command, capture_output=True, text=True, check=True)
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            verified = subprocess.run(command + ["--verify-lock"], capture_output=True, text=True)
            self.assertEqual(verified.returncode, 0, verified.stderr)
            self.assertTrue(json.loads(verified.stdout)["verified"])
            self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})
            provenance = root / "provenance"
            provenance.mkdir()
            (provenance / "cycle2_prepare-original.py").write_text("older unrelated generator\n")
            verified = subprocess.run(command + ["--verify-lock"], capture_output=True, text=True)
            self.assertEqual(verified.returncode, 0, verified.stderr)
            self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})

    def test_family_key_normalizes_case_colors_and_numbers(self):
        self.assertEqual(title_key("Example BLUE Shirt 42", loose=True), "example shirt")
        self.assertEqual(title_key("Example Red Shirt 10", loose=True), "example shirt")
        self.assertNotEqual(title_key("Example BLUE Shirt 42"), title_key("Example Red Shirt 10"))

    def test_pack_is_deterministic_and_target_family_disjoint(self):
        rows = products()
        rows.append({**rows[0], "parent_asin": "OLD-FAMILY", "title": rows[0]["title"] + " blue 44"})
        rows.append({**rows[2], "parent_asin": "NEW-FAMILY", "title": rows[2]["title"] + " red 12"})
        pack = build_pack(rows, old_samples())
        self.assertEqual(pack, build_pack(list(reversed(rows)), old_samples()))
        by_id = {row["parent_asin"]: row for row in rows}
        all_rows = pack["development"] + pack["validation"]
        targets = [row["ground_truth"]["parent_asin"] for row in all_rows]
        self.assertEqual(len(targets), 64)
        self.assertEqual(len(set(targets)), 64)
        self.assertNotIn("P0000", targets)
        self.assertNotIn("OLD-FAMILY", targets)
        self.assertEqual(len({title_key(by_id[target]["title"], loose=True) for target in targets}), 64)
        for split in ("development", "validation"):
            self.assertEqual(Counter(row["scenario_type"] for row in pack[split]),
                             {"buying": 12, "browsing": 12, "intent_override": 6, "boundary": 2})
            self.assertTrue(all(row["user_profile"] == {} for row in pack[split]))
            self.assertTrue(all("intent_card" not in row and "behavior" not in row for row in pack[split]))
        self.assertEqual(pack["audit"]["old_target_overlap"], 0)
        self.assertEqual(pack["audit"]["cross_split_loose_title_overlap"], 0)

    def test_duplicate_catalog_identifier_is_rejected(self):
        rows = products()
        with self.assertRaisesRegex(ValueError, "Duplicate catalog"):
            build_pack(rows + [rows[0]], old_samples())

    def test_invalid_catalog_and_old_rows_are_rejected(self):
        for bad in (None, {}, {"parent_asin": ""}, {"parent_asin": 123}):
            with self.subTest(row=bad), self.assertRaises(ValueError):
                build_pack([bad], old_samples())
        with self.assertRaisesRegex(ValueError, "Duplicate old sample"):
            build_pack(products(), old_samples() * 2)
        with self.assertRaisesRegex(ValueError, "missing from catalog"):
            build_pack(products(), [{"sample_id": "old", "ground_truth": {"parent_asin": "MISSING"}}])
        with self.assertRaisesRegex(ValueError, "ground truth"):
            build_pack(products(), [{"sample_id": "old", "ground_truth": {}}])

    def test_insufficient_family_count_fails_without_partial_pack(self):
        with self.assertRaisesRegex(ValueError, "64 eligible"):
            build_pack(products(60), old_samples())

    def test_empty_titles_are_excluded(self):
        rows = products()
        rows += [{"parent_asin": "EMPTY", "title": "", "categories": ["Clothing"]},
                 {"parent_asin": "COLOR-ONLY", "title": "blue 42", "categories": ["Clothing"]}]
        pack = build_pack(rows, old_samples())
        self.assertEqual(pack["audit"]["ineligible_title_count"], 2)

    def test_lock_is_idempotent_and_manifest_binds_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = root / "catalog.jsonl"
            old = root / "old.jsonl"
            catalog.write_text("".join(json.dumps(row) + "\n" for row in products()))
            old.write_text("".join(json.dumps(row) + "\n" for row in old_samples()))
            output = root / "pack"
            first = lock_pack(catalog, old, output)
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            self.assertEqual(first, lock_pack(catalog, old, output))
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})
            self.assertFalse(first["validation_outcomes_accessed"])
            self.assertNotIn("validation_ids", first)
            (output / "validation.jsonl").write_text("tampered\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                lock_pack(catalog, old, output)

    def test_existing_unlocked_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = root / "catalog.jsonl"
            old = root / "old.jsonl"
            catalog.write_text("".join(json.dumps(row) + "\n" for row in products()))
            old.write_text("".join(json.dumps(row) + "\n" for row in old_samples()))
            output = root / "pack"
            output.mkdir()
            (output / "keep.txt").write_text("keep")
            with self.assertRaisesRegex(ValueError, "without a lock"):
                lock_pack(catalog, old, output)
            self.assertEqual((output / "keep.txt").read_text(), "keep")

    def test_changed_input_or_seed_cannot_replace_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = root / "catalog.jsonl"
            old = root / "old.jsonl"
            catalog.write_text("".join(json.dumps(row) + "\n" for row in products()))
            old.write_text("".join(json.dumps(row) + "\n" for row in old_samples()))
            output = root / "pack"
            lock_pack(catalog, old, output)
            with self.assertRaisesRegex(ValueError, "differs"):
                lock_pack(catalog, old, output, seed="another-seed")
            with catalog.open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                lock_pack(catalog, old, output)

    def test_jsonl_unicode_separator_inside_title_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = products()
            rows[1]["title"] += "\u2028distinctive"
            catalog = root / "catalog.jsonl"
            old = root / "old.jsonl"
            catalog.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            old.write_text("".join(json.dumps(row) + "\n" for row in old_samples()))
            self.assertEqual(lock_pack(catalog, old, root / "pack")["counts"],
                             {"development": 32, "validation": 32})

    def test_verify_lock_accepts_commit_only_drift_without_regeneration_or_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = root / "catalog.jsonl"
            old = root / "old.jsonl"
            original = root / "preparation.py"
            original.write_bytes(Path("experiments/cycle2_prepare.py").read_bytes())
            catalog.write_text("".join(json.dumps(row) + "\n" for row in products()))
            old.write_text("".join(json.dumps(row) + "\n" for row in old_samples()))
            output = root / "pack"
            manifest = lock_pack(catalog, old, output)
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with patch("experiments.cycle2_prepare.subprocess.check_output", return_value="f" * 40 + "\n"), \
                    patch("experiments.cycle2_prepare.build_pack", side_effect=AssertionError("must not regenerate")):
                receipt = verify_lock(catalog, old, output, original)
            self.assertTrue(receipt["verified"])
            self.assertTrue(receipt["commit_changed"])
            self.assertEqual(receipt["original_source_commit"], manifest["source_commit"])
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_verify_lock_rejects_tampered_data_catalog_and_creation_source(self):
        for changed, expected in (("validation.jsonl", "validation data"),
                                  ("catalog.jsonl", "catalog source"),
                                  ("old.jsonl", "old_dataset source"),
                                  ("preparation.py", "preparation_script source")):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                catalog = root / "catalog.jsonl"
                old = root / "old.jsonl"
                original = root / "preparation.py"
                original.write_bytes(Path("experiments/cycle2_prepare.py").read_bytes())
                catalog.write_text("".join(json.dumps(row) + "\n" for row in products()))
                old.write_text("".join(json.dumps(row) + "\n" for row in old_samples()))
                output = root / "pack"
                lock_pack(catalog, old, output)
                manifest_bytes = (output / "manifest.json").read_bytes()
                target = output / changed if changed == "validation.jsonl" else root / changed
                with target.open("ab") as handle:
                    handle.write(b"\n")
                with self.assertRaisesRegex(ValueError, expected):
                    verify_lock(catalog, old, output, original)
                self.assertEqual((output / "manifest.json").read_bytes(), manifest_bytes)

    def test_verify_lock_requires_creation_source_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = root / "catalog.jsonl"
            old = root / "old.jsonl"
            catalog.write_text("".join(json.dumps(row) + "\n" for row in products()))
            old.write_text("".join(json.dumps(row) + "\n" for row in old_samples()))
            output = root / "pack"
            lock_pack(catalog, old, output)
            with self.assertRaisesRegex(ValueError, "Missing preparation_script"):
                verify_lock(catalog, old, output, root / "missing.py")

    def test_default_verification_cannot_substitute_a_different_current_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = root / "catalog.jsonl"
            old = root / "old.jsonl"
            catalog.write_text("".join(json.dumps(row) + "\n" for row in products()))
            old.write_text("".join(json.dumps(row) + "\n" for row in old_samples()))
            output = root / "pack"
            manifest = lock_pack(catalog, old, output)
            manifest["source_sha256"]["preparation_script"] = "0" * 64
            (output / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "preserved preparation_script"):
                verify_lock(catalog, old, output)
