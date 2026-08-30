import unittest

from mercury.intent import decide_intent
from mercury.planning import build_retrieval_plan
from mercury.state import SessionState


class PlanningTest(unittest.TestCase):
    def plan(self, messages):
        state = SessionState({}, alternatives_mode="grouped", scoped_preferences=True)
        for turn, message in enumerate(messages, 1):
            state.update(message, turn)
        return state, build_retrieval_plan(state, decide_intent(state, messages[-1]))

    def test_plan_separates_positive_negative_and_search_context(self):
        state, plan = self.plan(["I need a blue canvas bag without leather."])
        self.assertIn("bags", plan.object_types)
        self.assertIn("leather", plan.negative_terms)
        self.assertNotIn("leather", plan.lexical_query)
        self.assertIn("Must not have: leather", plan.rerank_context)
        self.assertEqual(plan.lexical_query, state.query())

    def test_neutral_and_withdrawn_values_leave_all_search_contexts(self):
        _, plan = self.plan(["I need a blue cotton shirt.", "I do not have a color preference."])
        self.assertNotIn("blue", plan.positive_terms)
        self.assertNotIn("blue", plan.negative_terms)
        self.assertNotIn("blue", plan.rerank_context)
        self.assertIn("cotton", plan.positive_terms)

    def test_soft_negative_is_labeled_as_avoidance_not_a_hard_constraint(self):
        state, plan = self.plan(["I would prefer not to have leather."])
        leather = next(signal for signal in plan.soft_preferences if signal.value == "leather")
        self.assertEqual(leather.polarity, -1)
        self.assertFalse(leather.hard)
        self.assertIn("leather", plan.negative_terms)
        self.assertNotIn("leather", plan.lexical_query)
        self.assertIn("Prefer to avoid: leather", plan.rerank_context)
        self.assertNotIn("Must not have: leather", plan.rerank_context)
        self.assertEqual(plan.lexical_query, state.query())

    def test_plan_preserves_alternative_groups(self):
        _, plan = self.plan(["I need a cotton or linen shirt."])
        alternatives = [signal for signal in (*plan.hard_constraints, *plan.soft_preferences)
                        if signal.alternative_group]
        self.assertEqual({signal.value for signal in alternatives}, {"cotton", "linen"})
        self.assertEqual(len({signal.alternative_group for signal in alternatives}), 1)

    def test_component_scope_is_explicit(self):
        _, plan = self.plan(["I need a bag with a leather body, not just leather handles."])
        self.assertIn(("leather", "body"), {(signal.value, signal.scope) for signal in plan.scoped_features})

    def test_same_ledger_produces_same_plan(self):
        first_state, first = self.plan(["A waterproof canvas backpack for travel."])
        second = build_retrieval_plan(first_state, decide_intent(first_state, first_state.history[-1].text))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
