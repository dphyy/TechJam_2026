import json
import tempfile
import unittest
from pathlib import Path

from mercury.catalog import Catalog
from mercury.contrast import ContrastIndex, compile_contrasts
from mercury.types import Candidate, Preference


class ContrastTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "catalog.jsonl"
        rows = [
            {"parent_asin": "a", "title": "Blue cotton hiking shirt", "features": ["Breathable"]},
            {"parent_asin": "b", "title": "Red polyester hiking shirt"},
            {"parent_asin": "c", "title": "Red polyester hiking shirt"},
            {"parent_asin": "d", "title": "Blue wool winter coat"},
        ]
        self.path.write_text("\n".join(map(json.dumps, rows)))
        self.catalog = Catalog(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_compilation_is_bounded_and_grounded(self):
        rows = compile_contrasts(self.catalog, neighbor_limit=2)
        self.assertEqual(set(rows), {"a", "b", "c", "d"})
        for identifier, row in rows.items():
            self.assertLessEqual(len(row["neighbors"]), 2)
            self.assertNotIn(identifier, row["neighbors"])
            product = self.catalog.by_id[identifier]
            for item in row["differences"]:
                self.assertIn(item["value"], product.facets[item["attribute"]])
                self.assertGreaterEqual(item["weight"], 0)
                self.assertLessEqual(item["weight"], 1)
                self.assertTrue(item["source"])

    def test_identical_products_are_never_collapsed_or_invented_as_different(self):
        rows = compile_contrasts(self.catalog, neighbor_limit=1)
        self.assertEqual(rows["b"]["neighbors"], ["c"])
        self.assertEqual(rows["c"]["neighbors"], ["b"])
        self.assertEqual(rows["b"]["differences"], [])
        self.assertEqual(rows["c"]["differences"], [])

    def test_index_version_and_catalog_hash_are_checked(self):
        from mercury.contrast import write_contrasts
        write_contrasts(self.catalog, self.root / "contrast")
        index = ContrastIndex(self.catalog, self.root / "contrast")
        preferences = [Preference("material", "cotton", 1, "cotton")]
        ranked = index.rank([Candidate(p, 1) for p in self.catalog.products], preferences, 0.2)
        self.assertEqual(ranked[0].product.parent_asin, "a")
        manifest_path = self.root / "contrast" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["catalog_sha256"] = "changed"
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            ContrastIndex(self.catalog, self.root / "contrast")


if __name__ == "__main__":
    unittest.main()
