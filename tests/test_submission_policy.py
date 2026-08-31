from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.presentation_evaluate import PresentationAgent, PRESETS
from mercury.lexical import Agent, AgentConfig, FULL_WIDTH_CONFIG, RecommendationPolicy
from mercury.lexical.diagnostics import signature


class SubmissionPolicyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.catalog = Path(self.temp.name) / "catalog.jsonl"
        self.catalog.write_text("".join(json.dumps({"parent_asin": str(index), "categories": ["shirts"],
                                                   "features": ["leather"]}) + "\n" for index in range(12)))
        self.base_config = AgentConfig(recommendation_policy=RecommendationPolicy(adaptive=False))

    def agent(self, config):
        agent = Agent(self.catalog, config=config)
        self.addCleanup(agent.close)
        agent.reset("s", {})
        return agent

    def test_safe_tentative_choice_matches_the_guarded_experiment(self):
        baseline = self.agent(self.base_config)
        trial = PresentationAgent(self.catalog, PRESETS["tentative_top1"], inner=baseline)
        self.addCleanup(trial.close)
        trial.reset("s", {})
        candidate = self.agent(replace(self.base_config, tentative_on_ambiguity=True))
        for turn, message in enumerate(("I'm looking for shirts.", "I prefer long sleeves."), 1):
            expected = trial.respond("s", message, turn, 10)
            actual = candidate.respond("s", message, turn, 10)
            self.assertEqual(actual, expected)
            receipt = candidate.last_diagnostics
            returned = [item["parent_asin"] for item in actual["recommendations"]]
            self.assertEqual(receipt["stage_receipts"]["returned"]["ids"], returned)
            self.assertEqual(receipt["stage_receipts"]["returned"]["sha256"], signature(tuple(returned)))
            if turn == 1:
                self.assertEqual(len(returned), 1)
                self.assertEqual(receipt["output_width"]["reason"], "tentative_ambiguity")
                self.assertFalse(receipt["output_width"]["ambiguity_deferred"])

    def test_known_violation_does_not_become_a_tentative_recommendation(self):
        agent = self.agent(replace(self.base_config, tentative_on_ambiguity=True))
        response = agent.respond("s", "I'm looking for shirts. No leather.", 1, 10)
        self.assertEqual(response["recommendations"], [])
        self.assertEqual(agent.last_diagnostics["output_width"]["reason"], "ambiguity_deferred")

    def test_full_width_remains_the_same_raw_prefix(self):
        control = self.agent(FULL_WIDTH_CONFIG)
        candidate = self.agent(replace(FULL_WIDTH_CONFIG, tentative_on_ambiguity=True))
        message = "I'm looking for shirts."
        self.assertEqual(candidate.respond("s", message, 1, 10), control.respond("s", message, 1, 10))
        self.assertEqual(candidate.last_diagnostics["output_width"]["reason"], "full_width")

    def test_failure_cannot_consume_ambiguity_or_commit_the_tentative_turn(self):
        agent = self.agent(replace(self.base_config, tentative_on_ambiguity=True))
        message = "I'm looking for shirts."
        with patch.object(agent, "_validate_response", side_effect=ValueError("rejected response")):
            with self.assertRaises(ValueError):
                agent.respond("s", message, 1, 10)
        self.assertEqual(agent._sessions["s"].last_turn, 0)
        self.assertNotIn("s", agent._ambiguity_deferred)
        response = agent.respond("s", message, 1, 10)
        self.assertEqual(len(response["recommendations"]), 1)
        with patch.object(agent.search, "search_with_context", side_effect=AssertionError("duplicate search")):
            self.assertEqual(agent.respond("s", message, 1, 10), response)
        self.assertTrue(agent.last_diagnostics["cache_hit"])

    def test_string_false_cannot_enable_tentative_policy(self):
        with self.assertRaises(ValueError):
            AgentConfig(tentative_on_ambiguity="false")


if __name__ == "__main__":
    unittest.main()
