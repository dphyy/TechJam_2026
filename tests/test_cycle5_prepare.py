import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from experiments.cycle5_prepare import (
    BAND_EDGES,
    SPLITS,
    allocate,
    band_of,
    build_pack,
    derive_quotas,
    lock_pack,
    rating_number,
    verify_lock,
)


def products(count: int = 900) -> list[dict]:
    """A catalog whose review counts span every band, thinning toward the top."""
    rows = []
    for index in range(count):
        # Deterministic spread: most rows are unpopular, a few are very popular.
        if index % 25 == 0:
            ratings = 30000 + index
        elif index % 7 == 0:
            ratings = 6000 + index
        elif index % 3 == 0:
            ratings = 1200 + index
        elif index % 2 == 0:
            ratings = 150 + index
        else:
            ratings = index % 120
        rows.append(
            {
                "parent_asin": f"P{index:04d}",
                "title": "cotton shirt " + chr(97 + index // 26) + chr(97 + index % 26),
                "categories": ["Clothing", "Shirts"],
                "features": ["cotton"],
                "rating_number": ratings,
            }
        )
    return rows


def sample(identifier: str, target: str) -> dict:
    return {"sample_id": identifier, "ground_truth": {"parent_asin": target}}


def public_samples(rows: list[dict], count: int = 40) -> list[dict]:
    # Draw a popularity-skewed reference, mirroring the released-public shape.
    ordered = sorted(rows, key=lambda row: -row["rating_number"])
    return [sample(f"public_{index}", row["parent_asin"]) for index, row in enumerate(ordered[:count])]


class BandingTest(unittest.TestCase):
    def test_band_edges_are_contiguous_and_half_open(self):
        self.assertEqual(band_of(0), 0)
        self.assertEqual(band_of(4), 0)
        self.assertEqual(band_of(5), 1)
        self.assertEqual(band_of(99), 1)
        self.assertEqual(band_of(100), 2)
        self.assertEqual(band_of(999), 2)
        self.assertEqual(band_of(1000), 3)
        self.assertEqual(band_of(4999), 3)
        self.assertEqual(band_of(5000), 4)
        self.assertEqual(band_of(19999), 4)
        self.assertEqual(band_of(20000), 5)
        self.assertEqual(band_of(10 ** 9), 5)

    def test_rating_number_rejects_unusable_values(self):
        for value in (None, "12", True, float("nan"), float("inf"), -3, [4]):
            self.assertEqual(rating_number({"rating_number": value}), 0)
        self.assertEqual(rating_number({"rating_number": 7}), 7)
        self.assertEqual(rating_number({}), 0)


class QuotaTest(unittest.TestCase):
    def test_quotas_sum_to_total_and_track_shares(self):
        quotas = derive_quotas([2, 8, 41, 41, 60, 48], 320)
        self.assertEqual(sum(quotas), 320)
        self.assertEqual(quotas[0], 3)
        self.assertGreater(quotas[4], quotas[2])

    def test_quotas_reject_degenerate_input(self):
        with self.assertRaises(ValueError):
            derive_quotas([0, 0], 10)
        with self.assertRaises(ValueError):
            derive_quotas([1, 1], 0)

    def test_shortfall_cascades_into_the_next_lower_band(self):
        taken, shortfall = allocate([10, 10, 10], [10, 16, 4])
        self.assertEqual(sum(taken), 30)
        self.assertEqual(taken[2], 4)
        self.assertEqual(shortfall[2], 6)
        self.assertEqual(taken[1], 16)
        self.assertEqual(taken[0], 10)

    def test_allocation_refuses_an_impossible_total(self):
        with self.assertRaises(ValueError):
            allocate([10, 10], [2, 2])


class Cycle5PackTest(unittest.TestCase):
    def setUp(self):
        self.rows = products()
        self.public = public_samples(self.rows)
        self.consumed = [[sample("consumed_0", "P0001")]]
        self.pack = build_pack(self.rows, self.public, self.consumed)

    def test_build_is_deterministic_under_input_order(self):
        self.assertEqual(self.pack, build_pack(list(reversed(self.rows)), self.public, self.consumed))

    def test_split_sizes_and_scenario_mix_are_fixed(self):
        for name, scenarios in SPLITS.items():
            self.assertEqual(len(self.pack[name]), len(scenarios))
            self.assertEqual(
                Counter(row["scenario_type"] for row in self.pack[name]), Counter(scenarios)
            )

    def test_no_overlap_with_public_consumed_or_between_splits(self):
        for key, value in self.pack["audit"].items():
            if key.endswith("overlap"):
                self.assertEqual(value, 0, key)

    def test_every_split_receives_a_share_of_the_popular_bands(self):
        by_id = {row["parent_asin"]: row for row in self.rows}
        top = {}
        for name in SPLITS:
            bands = [band_of(rating_number(by_id[row["ground_truth"]["parent_asin"]])) for row in self.pack[name]]
            top[name] = sum(band >= 3 for band in bands) / len(bands)
        self.assertGreater(min(top.values()), 0.0)
        # Dealing on one continuous cycle keeps the splits comparable.
        self.assertLess(max(top.values()) - min(top.values()), 0.15)

    def test_audit_records_the_reference_distribution_and_shortfall(self):
        bands = self.pack["audit"]["popularity_bands"]
        self.assertEqual(len(bands), len(BAND_EDGES))
        self.assertEqual(sum(entry["selected"] for entry in bands.values()), 320)
        self.assertEqual(
            sum(entry["released_public_targets"] for entry in bands.values()), len(self.public)
        )
        for entry in bands.values():
            self.assertLessEqual(entry["selected"], entry["eligible_families"])
        self.assertEqual(
            self.pack["audit"]["band_shortfall_total"],
            sum(entry["unfilled_quota_served_elsewhere"] for entry in bands.values()),
        )

    def test_draw_is_popularity_matched_not_uniform(self):
        by_id = {row["parent_asin"]: row for row in self.rows}
        chosen = [
            rating_number(by_id[row["ground_truth"]["parent_asin"]])
            for name in SPLITS
            for row in self.pack[name]
        ]
        catalog = [rating_number(row) for row in self.rows]
        chosen.sort()
        catalog.sort()
        self.assertGreater(chosen[len(chosen) // 2], catalog[len(catalog) // 2])

    def test_requires_public_and_consumed_inputs(self):
        with self.assertRaises(ValueError):
            build_pack(self.rows, [], self.consumed)
        with self.assertRaises(ValueError):
            build_pack(self.rows, self.public, [])


class Cycle5LockTest(unittest.TestCase):
    def _write(self, directory: Path) -> tuple[Path, Path, Path]:
        rows = products()
        catalog = directory / "catalog.jsonl"
        catalog.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        public = directory / "public.jsonl"
        public.write_text(
            "".join(json.dumps(row) + "\n" for row in public_samples(rows)), encoding="utf-8"
        )
        consumed = directory / "consumed.jsonl"
        consumed.write_text(json.dumps(sample("consumed_0", "P0001")) + "\n", encoding="utf-8")
        return catalog, public, consumed

    def test_lock_is_idempotent_and_verifies(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            catalog, public, consumed = self._write(directory)
            output = directory / "pack"
            first = lock_pack(catalog, public, [consumed], output)
            second = lock_pack(catalog, public, [consumed], output)
            self.assertEqual(first, second)
            self.assertEqual(verify_lock(catalog, public, [consumed], output)["verified"], True)

    def test_verify_rejects_tampered_data(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            catalog, public, consumed = self._write(directory)
            output = directory / "pack"
            lock_pack(catalog, public, [consumed], output)
            path = output / "screening.jsonl"
            path.write_text(path.read_text(encoding="utf-8").replace("cycle5_screening_0001", "tampered"), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_lock(catalog, public, [consumed], output)

    def test_lock_refuses_a_directory_without_a_manifest(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            catalog, public, consumed = self._write(directory)
            output = directory / "pack"
            output.mkdir()
            with self.assertRaises(ValueError):
                lock_pack(catalog, public, [consumed], output)


if __name__ == "__main__":
    unittest.main()
