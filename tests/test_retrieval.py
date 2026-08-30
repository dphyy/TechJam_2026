import json
import tempfile
import unittest
from pathlib import Path

from mercury.catalog import Catalog
from mercury.retrieval import SparseIndex, fuse_routes


class RetrievalTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        path = Path(self.temp.name) / "catalog.jsonl"
        rows = [
            {"parent_asin": "shirt", "title": "Blue cotton shirt", "categories": ["Shirts"]},
            {"parent_asin": "dress", "title": "Blue cotton dress", "categories": ["Dresses"]},
            {"parent_asin": "shoes", "title": "Waterproof running shoes", "categories": ["Shoes"]},
            {"parent_asin": "copy", "title": "Blue cotton shirt", "categories": ["Shirts"]},
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows))
        self.catalog = Catalog(path)
        self.index = SparseIndex(self.catalog)

    def tearDown(self):
        self.index.close()
        self.temp.cleanup()

    def test_stemming_and_title_matching(self):
        result = self.index.search("cotton shirts", 10)
        self.assertEqual(set(result[:2]), {"shirt", "copy"})

    def test_safe_query_and_empty_query(self):
        self.assertEqual(self.index.search('" OR * : ()', 10), [])
        self.assertEqual(self.index.search("", 10), [])
        self.assertEqual(self.index.search("shoes", 0), [])

    def test_numeric_specifications_are_retained_as_search_terms(self):
        from mercury.retrieval import terms
        self.assertEqual(terms("80% cotton 3 button closure"), ["80", "cotton", "3", "button", "closure"])

    def test_identical_products_keep_both_ids(self):
        result = self.index.search("blue cotton shirt", 10)
        self.assertIn("shirt", result)
        self.assertIn("copy", result)

    def test_category_scoped_route(self):
        result = self.index.search("blue cotton", 10, categories=["shirts"])
        self.assertEqual(set(result), {"shirt", "copy"})

    def test_fusion_is_deterministic_and_deduplicated(self):
        scores = fuse_routes({"sparse": ["shirt", "dress", "shirt"], "dense": ["dress", "copy"]},
                             {"sparse": 0.5, "dense": 0.5})
        self.assertEqual(scores[0][0], "dress")
        self.assertEqual(len(scores), 3)
        self.assertGreater(scores[0][1], scores[-1][1])


if __name__ == "__main__":
    unittest.main()
