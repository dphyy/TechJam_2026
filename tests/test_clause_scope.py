import unittest

from mercury.state import SessionState


class ClauseScopeTest(unittest.TestCase):
    def states(self):
        for canonical in (False, True):
            yield SessionState(
                {}, alternatives_mode="grouped", scoped_preferences=True,
                canonical_state_semantics=canonical,
            )

    @staticmethod
    def values(state, attribute, polarity=1):
        return {p.value for p in state.active_preferences()
                if p.attribute == attribute and p.polarity == polarity}

    def test_independent_relaxation_preserves_required_material(self):
        for message in (
            "Cotton is essential and pockets are not required.",
            "Cotton is required and wool is not needed.",
            "I need cotton while pockets aren't necessary.",
            "Cotton is mandatory whereas pockets are no longer needed.",
        ):
            for state in self.states():
                with self.subTest(message=message, canonical=state.canonical_state_semantics):
                    state.update("A blue jacket with wool and pockets.", 1)
                    state.update(message, 2)
                    cotton = next(p for p in state.active_preferences()
                                  if p.attribute == "material" and p.value == "cotton")
                    self.assertEqual(cotton.polarity, 1)
                    self.assertTrue(cotton.hard)
                    relaxed_attribute, relaxed_value = (
                        ("material", "wool") if "wool" in message else ("feature", "pockets")
                    )
                    self.assertNotIn(relaxed_value, self.values(state, relaxed_attribute))
                    self.assertNotIn(relaxed_value, self.values(state, relaxed_attribute, -1))
                    self.assertEqual(self.values(state, "color"), {"blue"})
                    self.assertEqual(self.values(state, "category"), {"jackets"})

    def test_named_no_preference_preserves_independent_requirement(self):
        for message in (
            "Any color is fine and cotton is required.",
            "No color preference and the material must be cotton.",
            "I do not care about color and need cotton.",
            "Cotton is required and I have no color preference.",
            "I need cotton and have no color preference.",
        ):
            for state in self.states():
                with self.subTest(message=message, canonical=state.canonical_state_semantics):
                    state.update("A blue wool jacket.", 1)
                    state.update(message, 2)
                    self.assertEqual(self.values(state, "color", 0), {"any"})
                    self.assertEqual(self.values(state, "color"), set())
                    self.assertIn("cotton", self.values(state, "material"))
                    self.assertEqual(self.values(state, "material", 0), set())
                    self.assertTrue(next(p.hard for p in state.active_preferences()
                                         if p.value == "cotton"))
                    self.assertEqual(self.values(state, "category"), {"jackets"})

    def test_uncertain_field_does_not_swallow_independent_requirement(self):
        for message in (
            "I am not sure about color and need cotton.",
            "I do not know the size and the material must be cotton.",
        ):
            for state in self.states():
                with self.subTest(message=message, canonical=state.canonical_state_semantics):
                    state.update("A blue jacket.", 1)
                    state.update(message, 2)
                    self.assertEqual(self.values(state, "material"), {"cotton"})
                    self.assertEqual(self.values(state, "material", -1), set())
                    self.assertEqual(self.values(state, "color"), {"blue"})

    def test_compound_subject_relaxes_both_values(self):
        for message, attribute, expected in (
            ("Cotton and linen are not required.", "material", {"cotton", "linen"}),
            ("Pockets and a hood are no longer needed.", "feature", {"pockets", "hood"}),
        ):
            for state in self.states():
                with self.subTest(message=message, canonical=state.canonical_state_semantics):
                    state.update("A cotton and linen jacket with pockets and a hood.", 1)
                    state.update(message, 2)
                    self.assertTrue(expected.isdisjoint(self.values(state, attribute)))
                    self.assertTrue(expected.isdisjoint(self.values(state, attribute, -1)))
                    self.assertEqual(self.values(state, "category"), {"jackets"})

    def test_no_preference_can_still_name_multiple_fields(self):
        for state in self.states():
            state.update("A blue cotton jacket with pockets.", 1)
            state.update("No preference on color and material.", 2)
            self.assertEqual(self.values(state, "color", 0), {"any"})
            self.assertEqual(self.values(state, "material", 0), {"any"})
            self.assertEqual(self.values(state, "feature"), {"pockets"})

    def test_independent_relaxation_preserves_alternative_group(self):
        for state in self.states():
            state.update("A blue shirt with pockets.", 1)
            state.update("Cotton or linen are required and pockets are not required.", 2)
            materials = [p for p in state.active_preferences() if p.attribute == "material"]
            self.assertEqual({p.value for p in materials}, {"cotton", "linen"})
            self.assertEqual(len({p.alternative_group for p in materials}), 1)
            self.assertTrue(all(p.hard and p.polarity == 1 and p.alternative_group for p in materials))
            self.assertEqual(self.values(state, "feature"), set())

    def test_relaxation_preserves_owned_quantity_and_component(self):
        for state in self.states():
            state.update("A jacket with 3 pockets and a hood. Blue lining.", 1)
            state.update("Pockets are essential and a hood is not required.", 2)
            self.assertIn("3 pockets", state.query())
            self.assertEqual(self.values(state, "feature"), {"pockets"})
            self.assertTrue(next(p.hard for p in state.active_preferences() if p.value == "pockets"))
            blue = next(p for p in state.active_preferences() if p.value == "blue")
            self.assertEqual(blue.scope, "lining")

    def test_independent_negative_is_not_reinterpreted_as_relaxation(self):
        for state in self.states():
            state.update("No wool and pockets are not required.", 1)
            self.assertEqual(self.values(state, "material", -1), {"wool"})
            self.assertEqual(self.values(state, "material", 0), set())
            self.assertEqual(self.values(state, "feature", -1), set())

    def test_required_lists_finish_before_independent_relaxation(self):
        for message, attribute, expected in (
            ("I need cotton and linen and pockets are not required.", "material", {"cotton", "linen"}),
            ("I need cotton and linen while pockets aren't necessary.", "material", {"cotton", "linen"}),
            ("I need pockets and a hood and waterproof isn't required.", "feature", {"pockets", "hood"}),
        ):
            for state in self.states():
                with self.subTest(message=message, canonical=state.canonical_state_semantics):
                    state.update(message, 1)
                    self.assertEqual(self.values(state, attribute), expected)
                    self.assertTrue(all(p.hard for p in state.active_preferences()
                                        if p.attribute == attribute and p.polarity == 1))

    def test_relaxed_joint_subject_starts_after_complete_predicate(self):
        for state in self.states():
            state.update("Cotton is essential and pockets and a hood are not required.", 1)
            self.assertEqual(self.values(state, "material"), {"cotton"})
            self.assertEqual(self.values(state, "feature", 0), {"pockets", "hood"})
            self.assertEqual(self.values(state, "feature"), set())

    def test_three_independent_predicates_keep_their_scope(self):
        for message in (
            "Cotton is required and pockets are not needed and linen is essential.",
            "Cotton is required and linen is essential and pockets are not needed.",
            "No color preference and cotton is required and pockets are not needed.",
        ):
            for state in self.states():
                with self.subTest(message=message, canonical=state.canonical_state_semantics):
                    state.update(message, 1)
                    self.assertIn("cotton", self.values(state, "material"))
                    if "linen" in message:
                        self.assertIn("linen", self.values(state, "material"))
                    else:
                        self.assertEqual(self.values(state, "color", 0), {"any"})
                    self.assertEqual(self.values(state, "feature", 0), {"pockets"})


if __name__ == "__main__":
    unittest.main()
