import unittest

from mercury.intent import decide_intent
from mercury.state import SessionState


class IntentTest(unittest.TestCase):
    def decision(self, message, prior=()):
        state = SessionState({})
        for turn, text in enumerate(prior, 1):
            state.update(text, turn)
        state.update(message, len(prior) + 1)
        return decide_intent(state, message)

    def test_targeted_request_is_buying(self):
        decision = self.decision("I need black waterproof running shoes under $100.")
        self.assertEqual(decision.mode, "buying")
        self.assertGreaterEqual(decision.hard_constraint_count, 1)
        self.assertIn("explicit_object", decision.reasons)

    def test_open_use_case_is_browsing_and_over_general(self):
        decision = self.decision("I am exploring gift ideas for a wedding.")
        self.assertEqual(decision.mode, "browsing")
        self.assertTrue(decision.over_general)
        self.assertIn("use_case_without_object", decision.reasons)

    def test_product_with_exploratory_language_is_mixed(self):
        decision = self.decision("I am exploring ideas for an everyday bag.")
        self.assertEqual(decision.mode, "mixed")

    def test_override_uses_live_state_and_reports_reason(self):
        decision = self.decision("Actually switch that to blue canvas instead.",
                                 ("I need a black leather shoulder bag.",))
        self.assertEqual(decision.mode, "buying")
        self.assertIn("intent_override", decision.reasons)
        self.assertIn("explicit_object", decision.reasons)

    def test_rephrasing_preserves_route(self):
        variants = ("I need running shoes under $80.", "Running shoes, maximum budget $80.")
        self.assertEqual({self.decision(text).mode for text in variants}, {"buying"})


if __name__ == "__main__":
    unittest.main()
