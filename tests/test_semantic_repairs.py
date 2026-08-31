import unittest

from mercury.state import SessionState


class SemanticRepairTest(unittest.TestCase):
    @staticmethod
    def state(canonical=False):
        return SessionState({}, alternatives_mode="grouped", scoped_preferences=True,
                            canonical_state_semantics=canonical)

    @staticmethod
    def values(state, attribute, polarity=1):
        return {p.value for p in state.active_preferences()
                if p.attribute == attribute and p.polarity == polarity}

    def test_named_relaxation_is_independent_of_query_order_mode(self):
        for field, value in (("color", "blue"), ("material", "cotton")):
            for message in (
                f"I no longer have a {field} preference.",
                f"I no longer want any particular {field} preference.",
                f"The {field} is no longer important.",
            ):
                with self.subTest(message=message):
                    state = self.state()
                    state.update("I need a blue cotton shirt with pockets.", 1)
                    state.update(message, 2)
                    self.assertEqual(self.values(state, field), set())
                    self.assertEqual(self.values(state, field, 0), {"any"})
                    self.assertEqual(self.values(state, "feature"), {"pockets"})
                    self.assertEqual(self.values(state, "category"), {"shirts"})
                    self.assertFalse(next(p.active for p in state.preferences if p.value == value))

    def test_contextual_relaxation_uses_only_the_prompted_field(self):
        state = self.state()
        state.update("A blue cotton shirt.", 1)
        state.record_question("color")
        state.update("I no longer have a preference.", 2)
        self.assertEqual(self.values(state, "color", 0), {"any"})
        self.assertEqual(self.values(state, "material"), {"cotton"})

    def test_correction_directives_extend_through_multiple_comma_fragments(self):
        for message in (
            "Correction: make that blue, made of canvas, size 10.",
            "Actually blue, made of canvas, size 10.",
            "On second thought, blue, canvas material, size 10.",
        ):
            with self.subTest(message=message):
                state = self.state()
                state.update("I want a black leather travel bag with an adjustable strap. Size 8.", 1)
                state.update(message, 2)
                self.assertEqual(self.values(state, "color"), {"blue"})
                self.assertEqual(self.values(state, "material"), {"canvas"})
                self.assertEqual(self.values(state, "size"), {"10"})
                self.assertEqual(self.values(state, "category"), {"bags"})
                self.assertEqual(self.values(state, "use_case"), {"travel"})
                self.assertEqual(self.values(state, "feature"), {"adjustable"})
                self.assertNotIn("correction", state.query())
                self.assertTrue(state.last_override.detected)

    def test_additive_fragment_preserves_material_and_component(self):
        for message, materials in (
            ("Correction: blue, also cotton lining.", {"leather", "cotton"}),
            ("Actually blue, made of canvas, also cotton lining.", {"canvas", "cotton"}),
        ):
            with self.subTest(message=message):
                state = self.state()
                state.update("A black leather bag.", 1)
                state.update(message, 2)
                self.assertEqual(self.values(state, "material"), materials)
                self.assertEqual(self.values(state, "color"), {"blue"})
                self.assertEqual(next(p.scope for p in state.active_preferences()
                                      if p.value == "cotton"), "lining")

    def test_correction_header_keeps_original_provenance(self):
        state = self.state()
        state.update("A black leather bag.", 1)
        message = "Correction: blue, made of canvas."
        state.update(message, 2)
        self.assertEqual(state.history[-1].text, message)
        recorded = state.history[-1].preferences
        self.assertTrue(recorded)
        self.assertTrue(all(p.source_text == message and p.source_turn == 2 for p in recorded))
        self.assertTrue(all(p.source_text == message for p in state.active_preferences()
                            if p.source_turn == 2))

    def test_comma_directive_does_not_extend_across_sentence_boundary(self):
        state = self.state()
        state.update("A black leather bag.", 1)
        state.update("Actually blue, made of canvas. Cotton lining.", 2)
        self.assertEqual(self.values(state, "material"), {"canvas", "cotton"})
        self.assertEqual(self.values(state, "category"), {"bags"})

    def test_acceptance_predicates_do_not_add_an_unrelated_requirement(self):
        reference = self.state()
        reference.update("A cotton or linen summer shirt.", 1)
        for predicate in ("is suitable", "would be suitable", "seems acceptable"):
            with self.subTest(predicate=predicate):
                state = self.state()
                state.update(f"For summer, either cotton or linen {predicate} for the shirt.", 1)
                self.assertEqual(state.semantic_signature(), reference.semantic_signature())

    def test_descriptive_words_are_not_globally_removed(self):
        for canonical in (False, True):
            for message, expected in (
                ("A color correction filter.", "correction filter"),
                ("A cotton shirt suitable for screen printing.", "suitable"),
                ("A correction collar.", "correction collar"),
            ):
                with self.subTest(message=message, canonical=canonical):
                    state = self.state(canonical)
                    state.update(message, 1)
                    self.assertIn(expected, state.query())

    def test_query_order_remains_unchanged(self):
        left, right = self.state(), self.state()
        left.update("A cotton or linen summer shirt.", 1)
        right.update("A linen or cotton summer shirt.", 1)
        self.assertNotEqual(left.query(), right.query())
        self.assertEqual(left.semantic_signature(), right.semantic_signature())


if __name__ == "__main__":
    unittest.main()
