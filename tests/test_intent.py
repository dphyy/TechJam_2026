import unittest

from mercury.intent import IntentWeights, decide_intent
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
        self.assertEqual(decision.event, "override")
        self.assertIn("intent_override", decision.reasons)
        self.assertIn("explicit_object", decision.reasons)

    def test_rephrasing_preserves_route(self):
        variants = ("I need running shoes under $80.", "Running shoes, maximum budget $80.")
        self.assertEqual({self.decision(text).mode for text in variants}, {"buying"})

    def test_need_ideas_is_browsing_not_buying_keyword_collision(self):
        decision = self.decision("I need ideas for a graduation gift.")
        self.assertEqual(decision.mode, "browsing")
        self.assertNotIn("direct_request", decision.reasons)

    def test_specific_product_for_a_gift_remains_buying(self):
        decision = self.decision("I need a silver necklace as a gift under $80.")
        self.assertEqual(decision.mode, "buying")
        self.assertIn("committed_language", decision.reasons)
        self.assertNotIn("browsing_language", decision.reasons)

    def test_tentative_object_with_firm_requirement_is_mixed(self):
        decision = self.decision("Maybe boots; whatever I choose must be waterproof.")
        self.assertEqual(decision.mode, "mixed")

    def test_correction_and_relaxation_are_separate_events(self):
        corrected = self.decision("Actually, no leather.", ("I need a leather shoulder bag.",))
        relaxed = self.decision("Forget the color; any color is fine.",
                                ("I need a red cotton shirt.",))
        self.assertEqual(corrected.event, "correction")
        self.assertEqual(relaxed.event, "relaxation")

    def test_low_confidence_mode_falls_back_to_mixed_for_actions(self):
        state = SessionState({})
        state.update("I am exploring bags.", 1)
        weights = IntentWeights(
            object=.6, slots=0, hard=0, buying_language=0, browsing_language=.1,
            use_case_without_object=0, unresolved=0, sparse_request=0,
        )
        decision = decide_intent(
            state, "I am exploring bags.", weights=weights, routing_confidence_threshold=.9,
        )
        self.assertEqual(decision.mode, "buying")
        self.assertEqual(decision.effective_mode, "mixed")
        self.assertIn("low_confidence_safe_fallback", decision.reasons)


if __name__ == "__main__":
    unittest.main()
