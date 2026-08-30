import unittest

from mercury.intent import decide_intent
from mercury.state import SessionState


class OverrideLanguageCapabilityTest(unittest.TestCase):
    """Target-disjoint language checks for semantic override behavior."""

    def transition(self, initial, followup, *, alternatives_mode="grouped"):
        state = SessionState({}, alternatives_mode=alternatives_mode)
        state.update(initial, 1)
        state.update(followup, 2)
        return state

    def test_implicit_replacements_are_detected_from_state_changes(self):
        cases = (
            ("A black leather bag.", "Blue works better.", "color", "black", "blue"),
            ("A black leather bag.", "Let's try canvas.", "material", "leather", "canvas"),
            ("A blue bag.", "Back to black.", "color", "blue", "black"),
            ("A leather bag.", "The backpack makes more sense.", "category", "bags", "backpacks"),
        )
        for initial, followup, attribute, old, new in cases:
            with self.subTest(followup=followup):
                state = self.transition(initial, followup)
                decision = state.last_override
                self.assertTrue(decision.detected)
                self.assertIn(attribute, decision.changed_attributes)
                self.assertIn((attribute, old), {(fact.attribute, fact.value) for fact in decision.retired})
                self.assertIn((attribute, new), {(fact.attribute, fact.value) for fact in decision.added})

    def test_negative_preference_flips_polarity(self):
        state = self.transition("A leather bag.", "I'm not feeling the leather.")
        decision = state.last_override
        self.assertTrue(decision.detected)
        self.assertIn("polarity_changed", decision.reasons)
        self.assertIn(("material", "leather", -1), {
            (fact.attribute, fact.value, fact.polarity) for fact in decision.added
        })

    def test_attribute_only_change_retires_the_old_value(self):
        for followup in ("Keep everything except the material.", "Same requirements, different material."):
            with self.subTest(followup=followup):
                state = self.transition("A black leather bag.", followup)
                self.assertTrue(state.last_override.detected)
                self.assertEqual(state.last_feedback.attribute, "material")
                self.assertNotIn("leather", state.query())

    def test_alternative_narrowing_is_an_override(self):
        state = self.transition("Cotton or linen shirts.", "Cotton only.")
        self.assertTrue(state.last_override.detected)
        self.assertIn(("material", "linen"), {
            (fact.attribute, fact.value) for fact in state.last_override.retired
        })

    def test_additive_preference_is_not_an_override(self):
        state = self.transition("A black bag.", "Also add pockets.")
        self.assertFalse(state.last_override.detected)
        self.assertIn("pockets", state.query())

    def test_protected_no_change_language_is_not_an_override(self):
        cases = (
            "Actually, black is still fine.",
            "Go with either black or blue.",
            "I don't need to change anything.",
            "Instead of changing the color, keep black.",
        )
        for followup in cases:
            with self.subTest(followup=followup):
                state = self.transition("A black bag.", followup)
                self.assertFalse(state.last_override.detected)
                self.assertNotIn("intent_override", decide_intent(state, followup).reasons)

    def test_explicit_zero_delta_restatement_remains_an_override(self):
        state = self.transition(
            "A blue cotton shirt.",
            "Actually, ignore my earlier preference. What I need is a blue cotton shirt.",
        )
        self.assertTrue(state.last_override.detected)
        self.assertEqual(state.last_override.retired, ())
        self.assertEqual(state.last_override.added, ())
        self.assertIn("explicit_correction_restatement", state.last_override.reasons)
        self.assertIn("intent_override", decide_intent(state, state.history[-1].text).reasons)

    def test_explicit_directive_without_a_parseable_replacement_is_an_override(self):
        state = self.transition(
            "A blue cotton shirt.",
            "Actually, ignore my earlier preference. What I need is: fabric.",
        )
        self.assertTrue(state.last_override.detected)
        self.assertFalse(state.last_update_informative)
        self.assertEqual(state.last_override.retired, ())
        self.assertEqual(state.last_override.added, ())
        self.assertIn("explicit_override_directive", state.last_override.reasons)
        self.assertIn("intent_override", decide_intent(state, state.history[-1].text).reasons)


if __name__ == "__main__":
    unittest.main()
