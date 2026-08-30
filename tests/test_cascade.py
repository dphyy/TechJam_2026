import unittest

from mercury.cascade import decide_compute_cascade
from mercury.config import Config
from mercury.types import IntentDecision, RetrievalPlan


def plan(objects=()):
    return RetrievalPlan("mixed", objects, objects, (), (), (), (), (), (), (), "", "Mode: mixed")


class ComputeCascadeTests(unittest.TestCase):
    def test_uncertainty_escalates_once_to_hard_d60_ceiling(self):
        config = Config(compute_cascade=True, cascade_threshold=.5,
                        cascade_candidate_threshold=100, cascade_max_turns=1)
        intent = IntentDecision("mixed", .2, .4, 0, True)
        decision = decide_compute_cascade(intent, plan(), {"a:b": .05}, 120, config, 0, True)
        self.assertTrue(decision.escalate)
        self.assertEqual(decision.rerank_limit, 60)
        exhausted = decide_compute_cascade(intent, plan(), {"a:b": .05}, 120, config, 1, True)
        self.assertFalse(exhausted.escalate)
        self.assertEqual(exhausted.rerank_limit, 30)

    def test_confident_scoped_or_disabled_paths_stay_at_d30(self):
        intent = IntentDecision("buying", .9, .95, 2, False)
        for config, available in ((Config(compute_cascade=True), True),
                                  (Config(compute_cascade=False), True),
                                  (Config(compute_cascade=True), False)):
            with self.subTest(config=config, available=available):
                decision = decide_compute_cascade(intent, plan(("shirts",)), {"a:b": .9}, 40,
                                                   config, 0, available)
                self.assertFalse(decision.escalate)
                self.assertEqual(decision.rerank_limit, 30)

    def test_calibrated_previous_margin_can_trigger_next_turn_escalation(self):
        config = Config(compute_cascade=True, cascade_threshold=.3,
                        cascade_previous_margin_threshold=.5)
        intent = IntentDecision("buying", .9, .95, 2, False)
        decision = decide_compute_cascade(intent, plan(("shirts",)), {}, 40, config, 0, True, .1)
        self.assertTrue(decision.escalate)
        self.assertIn("low_previous_neural_margin", decision.reasons)


if __name__ == "__main__":
    unittest.main()
