from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from demo.showcase import build_showcase
from mercury.agent import Agent
from mercury.catalog import product_from_dict
from mercury.config import Config
from mercury.types import Preference


class ShowcaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.catalog = self.root / "catalog.jsonl"
        rows = [
            {"parent_asin": "canvas", "title": "Blue canvas shoulder bag with adjustable strap",
             "categories": ["Bags"], "price": 39.0},
            {"parent_asin": "leather", "title": "Black leather shoulder bag with adjustable strap",
             "categories": ["Bags"], "price": 55.0},
            {"parent_asin": "other", "title": "Travel backpack", "categories": ["Backpacks"]},
            {"parent_asin": "red", "title": "Red cotton tote", "categories": ["Bags"]},
            {"parent_asin": "green", "title": "Green canvas handbag", "categories": ["Bags"]},
            {"parent_asin": "wallet", "title": "Blue leather wallet", "categories": ["Wallets"]},
        ]
        self.catalog.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        self.agent = Agent(self.catalog, Config(neural_rerank=False, question_policy="none", slate_size=3,
                                               slate_paging_first_turn=1,
                                               slate_reset_on_override=True))
        self.addCleanup(self.agent.close)

    def test_generates_portable_real_agent_evidence_report(self):
        output = self.root / "showcase"
        report = build_showcase(output, self.agent, results={"technical_score": 0.8})

        self.assertTrue((output / "index.html").is_file())
        self.assertTrue((output / "evidence.json").is_file())
        self.assertTrue(report["generated_from_real_agent"])
        self.assertEqual(len(report["turns"]), 5)
        self.assertGreaterEqual(report["turns"][4]["slate_page"], 1)
        self.assertEqual(report["turns"][1]["slate_page_reset"], "intent_override")
        self.assertTrue(any(item["value"] == "leather"
                            for item in report["turns"][1]["retracted_preferences"]))
        self.assertFalse(any(item["value"] == "adjustable"
                             for item in report["turns"][1]["retracted_preferences"]))
        statuses = {item["status"] for turn in report["turns"]
                    for product in turn["products"] for item in product["evidence"]}
        self.assertIn("supported", statuses)
        self.assertIn("contradicted", statuses)
        self.assertIn("unknown", statuses)
        page = (output / "index.html").read_text(encoding="utf-8")
        self.assertIn("See what Mercury remembered, changed, and proved.", page)
        self.assertIn("Technical score", page)
        self.assertIn("evidence.json", page)

    def test_refuses_to_overwrite_an_existing_showcase(self):
        output = self.root / "showcase"
        build_showcase(output, self.agent)

        with self.assertRaises(FileExistsError):
            build_showcase(output, self.agent)

    def test_negated_or_imitation_catalog_language_is_not_presented_as_support(self):
        from demo.showcase import _catalog_evidence

        product = product_from_dict({
            "parent_asin": "faux", "title": "Faux leather bag",
            "description": "A leather-free alternative.",
        })
        preferences = [
            Preference("material", "leather", 1, "leather"),
            Preference("material", "leather", 2, "no leather", polarity=-1),
        ]

        self.assertEqual(
            [item["status"] for item in _catalog_evidence(product, preferences)],
            ["unknown", "unknown"],
        )


if __name__ == "__main__":
    unittest.main()
