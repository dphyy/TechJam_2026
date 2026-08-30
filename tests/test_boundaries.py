import ast
import socket
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import test_agent


class InferenceBoundaryTest(unittest.TestCase):
    def test_runtime_has_no_evaluator_or_label_dependencies(self):
        forbidden_imports = {"evaluator", "experiments"}
        forbidden_keys = {"sample_id", "scenario_type", "ground_truth", "intent_card", "behavior_for"}
        for path in Path("mercury").glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertFalse({alias.name.split(".")[0] for alias in node.names} & forbidden_imports, path)
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn((node.module or "").split(".")[0], forbidden_imports, path)
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.assertNotIn(node.value, forbidden_keys, path)

    def test_sparse_runtime_succeeds_when_network_is_denied(self):
        fixture = test_agent.AgentTest()
        fixture.setUp()
        try:
            with patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")), \
                 patch("socket.create_connection", side_effect=AssertionError("Network forbidden")):
                fixture.agent.reset("offline", {})
                result = fixture.agent.respond("offline", "A blue cotton shirt", 1, 10)
            self.assertEqual(len(result["recommendations"]), 10)
            self.assertEqual(fixture.agent.last_diagnostics["fallbacks"], [])
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
