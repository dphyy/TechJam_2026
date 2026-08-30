import json
import tempfile
import unittest
from pathlib import Path

from mercury.agent import Agent
from mercury.config import Config


class RoutedRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "catalog.jsonl"
        rows = [
            {"parent_asin": "shirt", "title": "Blue cotton shirt", "categories": ["Shirts"]},
            {"parent_asin": "dress", "title": "Blue wedding dress", "categories": ["Dresses"]},
            {"parent_asin": "gift", "title": "Wedding gift jewelry", "categories": ["Jewelry"]},
        ]
        self.path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def agent(self, **values):
        return Agent(self.path, Config(evidence_ranking=False, routed_retrieval=True, **values))

    def test_buying_favors_scoped_sparse_route(self):
        agent = self.agent()
        agent.reset("s", {})
        agent.respond("s", "I need a blue cotton shirt.", 1, 10)
        self.assertEqual(agent.last_diagnostics["intent"]["mode"], "buying")
        self.assertAlmostEqual(agent.last_diagnostics["route_weights"]["sparse"], 0.45)
        self.assertAlmostEqual(agent.last_diagnostics["route_weights"]["scoped"], 0.55)
        agent.close()

    def test_browsing_adds_use_case_recovery_route(self):
        agent = self.agent()
        agent.reset("s", {})
        agent.respond("s", "I am exploring gift ideas for a wedding.", 1, 10)
        self.assertEqual(agent.last_diagnostics["intent"]["mode"], "browsing")
        self.assertIn("scenario_sparse", agent.last_diagnostics["routes"])
        self.assertEqual(agent.last_diagnostics["route_weights"]["scenario_sparse"], 0.25)
        agent.close()

    def test_dense_weight_depends_on_route_and_uses_semantic_context(self):
        class FakeDense:
            prompt_tokens = 0

            def __init__(self):
                self.queries = []

            def search(self, query, limit):
                self.queries.append((query, limit))
                return ["gift", "dress"]

        agent = self.agent()
        agent.dense = FakeDense()
        agent.reset("s", {})
        agent.respond("s", "I am exploring gift ideas for a wedding.", 1, 10)
        self.assertEqual(agent.last_diagnostics["route_weights"]["dense"], 0.55)
        self.assertIn("wedding", agent.dense.queries[0][0])
        self.assertNotIn("I am", agent.dense.queries[0][0])
        agent.close()

    def test_dense_failure_keeps_legal_sparse_slate(self):
        class BrokenDense:
            prompt_tokens = 0

            def search(self, query, limit):
                raise TimeoutError

        agent = self.agent()
        agent.dense = BrokenDense()
        agent.reset("s", {})
        response = agent.respond("s", "I am exploring gift ideas for a wedding.", 1, 10)
        identifiers = [item["parent_asin"] for item in response["recommendations"]]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertIn("dense", agent.last_diagnostics["fallbacks"])
        agent.close()


if __name__ == "__main__":
    unittest.main()
