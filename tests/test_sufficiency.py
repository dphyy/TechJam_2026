import unittest

from mercury.config import Config
from mercury.intent import decide_intent
from mercury.planning import build_retrieval_plan
from mercury.state import SessionState
from mercury.sufficiency import decide_retrieval_sufficiency


class RetrievalSufficiencyTests(unittest.TestCase):
    def decision(self, message, *, turn=1, deferred=0, action="minimal_probe", gate=True):
        state = SessionState({})
        state.update(message, turn)
        intent = decide_intent(state, message)
        plan = build_retrieval_plan(state, intent)
        config = Config(retrieval_sufficiency_gate=gate, insufficient_action=action,
                        question_policy="intent")
        return decide_retrieval_sufficiency(state, intent, plan, config, turn, deferred)

    def test_vague_request_uses_configured_bounded_action(self):
        self.assertEqual(self.decision("I am exploring gift ideas.").action, "minimal_probe")
        self.assertEqual(self.decision("I am exploring gift ideas.", action="clarify_first").action,
                         "clarify_first")

    def test_specific_final_productive_and_exhausted_cases_retrieve(self):
        self.assertEqual(self.decision("I need blue canvas bags with a zipper.").action, "retrieve")
        self.assertEqual(self.decision("I am exploring gift ideas.", turn=10).action, "retrieve")
        self.assertEqual(self.decision("I am exploring gift ideas.", deferred=1).action, "retrieve")
        self.assertEqual(self.decision("I am exploring gift ideas.", gate=False).action, "retrieve")


if __name__ == "__main__":
    unittest.main()
