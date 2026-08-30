import json
import tempfile
import unittest
from pathlib import Path

from mercury.catalog import Catalog
from mercury.state import SessionState
from mercury.vocabulary import CatalogVocabulary


class CatalogVocabularyTest(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path, str]:
        catalog = root / "catalog.jsonl"
        catalog.write_text(json.dumps({
            "parent_asin": "A", "title": "Trail gaiters", "categories": ["Outdoor", "Gaiters"],
        }) + "\n", encoding="utf-8")
        catalog_hash = Catalog(catalog).sha256
        model = root / "vocabulary.json"
        model.write_text(json.dumps({
            "schema": "mercury-catalog-vocabulary-v1",
            "version": "test-v1",
            "catalog_sha256": catalog_hash,
            "minimum_support": 5,
            "minimum_confidence": .8,
            "taxonomy": [{"canonical": "trail gaiters", "role": "object", "support": 9,
                          "method": "category_path"}],
            "aliases": [{
                "alias": "trail gaiter", "attribute": "category", "canonical": "trail gaiters",
                "support": 9, "confidence": 1.0, "method": "category_path",
            }],
        }), encoding="utf-8")
        return catalog, model, catalog_hash

    def test_longest_unoccupied_alias_is_proposed_with_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog, model, catalog_hash = self.fixture(Path(directory))
            vocabulary = CatalogVocabulary(model, catalog_hash)
            state = SessionState({}, "ledger", "grouped", False, vocabulary)
            state.update("I need a trail-gaiter.", 1)
            matching = [item for item in state.active_preferences() if item.value == "trail gaiters"]
            self.assertEqual(len(matching), 1)
            self.assertEqual(matching[0].attribute, "category")
            self.assertTrue(matching[0].source_kind.startswith("catalog_alias:test-v1:"))
            self.assertFalse(matching[0].hard)
            self.assertEqual(vocabulary.category_role("trail gaiters"), "object")
            self.assertEqual(catalog.name, "catalog.jsonl")

    def test_catalog_proposal_cannot_retire_conflicting_explicit_fact(self):
        with tempfile.TemporaryDirectory() as directory:
            _, model, catalog_hash = self.fixture(Path(directory))
            state = SessionState({}, "ledger", "grouped", False, CatalogVocabulary(model, catalog_hash))
            state.update("I need boots.", 1)
            state.update("Maybe trail gaiter options too.", 2)
            categories = {item.value for item in state.active_preferences() if item.attribute == "category"}
            self.assertIn("boots", categories)
            self.assertIn("trail gaiters", categories)

    def test_static_explicit_span_owns_overlap_and_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            _, model, catalog_hash = self.fixture(Path(directory))
            vocabulary = CatalogVocabulary(model, catalog_hash)
            self.assertEqual(vocabulary.find("trail gaiter", [(0, 12)]), [])
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                CatalogVocabulary(model, "different")


if __name__ == "__main__":
    unittest.main()
