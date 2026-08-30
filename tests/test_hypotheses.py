import unittest

from mercury.hypotheses import build_intent_hypotheses
from mercury.types import RetrievalPlan


def plan(objects=(), positives=(), use_cases=(), query=""):
    return RetrievalPlan("mixed", objects, objects, positives, (), (), (), use_cases,
                         (), (), query, "Mode: mixed")


class IntentHypothesisTests(unittest.TestCase):
    def test_use_case_adds_at_most_one_alternative_to_open_request(self):
        hypotheses = build_intent_hypotheses(
            plan(positives=("wedding",), use_cases=("wedding",), query="wedding"), 2,
        )
        self.assertEqual(len(hypotheses), 2)
        self.assertEqual(hypotheses[0].reason, "open_request")
        self.assertEqual(hypotheses[1].object_types, ("dresses",))

    def test_explicit_objects_are_preserved_and_hard_bounded(self):
        hypotheses = build_intent_hypotheses(
            plan(objects=("bags", "wallets", "shoes"), positives=("blue",), query="blue"), 2,
        )
        self.assertEqual([item.object_types for item in hypotheses], [("bags",), ("wallets",)])
        with self.assertRaises(ValueError):
            build_intent_hypotheses(plan(query="gift"), 3)


if __name__ == "__main__":
    unittest.main()
