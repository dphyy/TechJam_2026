import json
import tempfile
import unittest
from pathlib import Path

from mercury.agent import Agent
from mercury.catalog import Catalog
from mercury.config import Config
from mercury.state import SessionState
from mercury.vocabulary import CatalogVocabulary


class CatalogVocabularyV2Test(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path, str]:
        catalog = root / "catalog.jsonl"
        catalog.write_text("\n".join(json.dumps(row) for row in (
            {"parent_asin": "A", "title": "Trail gaiters", "categories": ["Outdoor", "Trail Gaiters"]},
            {"parent_asin": "B", "title": "Plain hiking item", "categories": ["Outdoor"]},
        )), encoding="utf-8")
        catalog_hash = Catalog(catalog).sha256
        model = root / "vocabulary.json"
        model.write_text(json.dumps({
            "schema": "mercury-catalog-vocabulary-v2",
            "version": "test-v2",
            "catalog_sha256": catalog_hash,
            "minimum_support": 5,
            "minimum_confidence": 0.8,
            "minimum_ambiguity_margin": 0.2,
            "state_minimum_confidence": 0.95,
            "state_minimum_ambiguity_margin": 0.5,
            "taxonomy": [{"canonical": "trail gaiters", "role": "object", "support": 9,
                          "method": "category_path"}],
            "aliases": [{
                "alias": "trail gaiter", "attribute": "category", "canonical": "trail gaiters",
                "support": 9, "confidence": 1.0, "ambiguity_margin": 1.0,
                "state_eligible": True, "role": "object", "method": "category_path",
            }],
        }), encoding="utf-8")
        return catalog, model, catalog_hash

    def test_cued_alias_uses_state_and_retrieval_lanes(self):
        with tempfile.TemporaryDirectory() as directory:
            _, model, catalog_hash = self.fixture(Path(directory))
            state = SessionState({}, "ledger", "grouped", False,
                                 CatalogVocabulary(model, catalog_hash), True)
            state.update("I need a trail-gaiter", 1)
            self.assertIn(
                ("category", "trail gaiters"),
                {(item.attribute, item.value) for item in state.active_preferences()},
            )
            self.assertEqual(state.vocabulary_expansion_query(), "trail gaiters")

    def test_uncued_alias_is_query_local_and_negation_suppresses_both_lanes(self):
        with tempfile.TemporaryDirectory() as directory:
            _, model, catalog_hash = self.fixture(Path(directory))
            vocabulary = CatalogVocabulary(model, catalog_hash)
            state = SessionState({}, "ledger", "grouped", False, vocabulary, True)
            state.update("Tell me some background about trail gaiter", 1)
            self.assertFalse(any(item.source_kind.startswith("catalog_alias:")
                                 for item in state.active_preferences()))
            self.assertEqual(state.vocabulary_expansion_query(), "trail gaiters")
            state.update("Please avoid trail gaiter", 2)
            self.assertFalse(any(item.source_kind.startswith("catalog_alias:")
                                 for item in state.active_preferences()))
            self.assertEqual(state.vocabulary_expansion_query(), "")

    def test_agent_exposes_bounded_catalog_expansion_route_without_persistent_fact(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog, model, _ = self.fixture(Path(directory))
            agent = Agent(catalog, Config(
                catalog_vocabulary=True, catalog_vocabulary_path=str(model),
                canonical_state_semantics=True, evidence_ranking=False,
                question_policy="none", slate_size=2,
            ))
            try:
                agent.reset("v2", {})
                agent.respond("v2", "Tell me about trail gaiter", 1, 2)
                diagnostics = agent.last_diagnostics
                self.assertEqual(diagnostics["vocabulary_expansion_query"], "trail gaiters")
                self.assertIn("catalog_expansion", diagnostics["routes"])
                self.assertTrue(diagnostics["catalog_vocabulary"]["dual_lane"])
                self.assertFalse(any(
                    item["source_kind"].startswith("catalog_alias:")
                    for item in diagnostics["preferences"]
                ))
            finally:
                agent.close()


if __name__ == "__main__":
    unittest.main()
