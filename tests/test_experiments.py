import unittest

from evaluator.local_evaluator import evaluate
from experiments.compare import paired_comparison, session_score
from experiments.run import ObservedAgent, source_hashes, summarize_traces


class ExperimentTest(unittest.TestCase):
    def test_score_matches_published_first_hit_formula(self):
        self.assertAlmostEqual(session_score({"hit": True, "reciprocal_rank": 1.0, "first_hit_turn": 1}), 1.0)
        self.assertAlmostEqual(session_score({"hit": True, "reciprocal_rank": 0.1, "first_hit_turn": 10}), 0.55)
        self.assertEqual(session_score({"hit": False, "reciprocal_rank": 0, "first_hit_turn": None}), 0)

    def test_paired_bootstrap_is_deterministic_and_identical_runs_are_zero(self):
        sessions = [{"sample_id": str(i), "hit": i % 2 == 0,
                     "reciprocal_rank": 1.0 if i % 2 == 0 else 0.0,
                     "first_hit_turn": 2 if i % 2 == 0 else None} for i in range(10)]
        result = paired_comparison(sessions, list(reversed(sessions)), resamples=100)
        self.assertEqual(result["technical_score_delta"], 0)
        self.assertEqual(result["technical_score_95pct_ci"], [0, 0])
        self.assertEqual(result, paired_comparison(sessions, sessions, resamples=100))

    def test_comparison_rejects_unpaired_runs(self):
        with self.assertRaises(ValueError):
            paired_comparison([{"sample_id": "a"}], [{"sample_id": "b"}])

    def test_uninstrumented_baseline_recall_is_unavailable_not_zero(self):
        summary = summarize_traces([[{"latency_seconds": .01, "response": {"recommendations": []}}]],
                                   [{"sample_id": "baseline", "ground_truth": {"parent_asin": "a"}}],
                                   [{"sample_id": "baseline", "hit": False}])
        self.assertIsNone(summary["ever_ranked_recall"])
        self.assertIsNone(summary["ever_route_recall"])
        self.assertEqual(summary["failure_diagnostics"], {"agent_error_turns": 0})

    def test_intent_override_failures_follow_official_hits(self):
        sample = {
            "sample_id": "synthetic-override", "scenario_type": "intent_override", "user_profile": {},
            "ground_truth": {"parent_asin": "target"},
            "intent_card": {"hard_constraints": ["blue"], "soft_preferences": ["red"]},
            "behavior": {"override": {"turn": 3, "old_value": "red", "new_value": "blue",
                                       "message": "Actually, blue."}},
        }
        for hit_after_override in (False, True):
            with self.subTest(hit_after_override=hit_after_override):
                class SyntheticAgent:
                    def reset(self, session_id, profile):
                        pass

                    def respond(self, session_id, message, turn, top_k):
                        self.last_diagnostics = {"retrieved_ids": ["target", "other"],
                                                 "ranked_ids": ["target", "other"],
                                                 "routes": {"sparse": ["target", "other"]}}
                        recommended = "target" if turn == 1 or hit_after_override and turn >= 3 else "other"
                        return {"message": "Any color preference?", "ask_attribute": "color",
                                "recommendations": [{"parent_asin": recommended}]}

                observed = ObservedAgent(SyntheticAgent())
                result = evaluate(observed, [sample], {"target", "other"}, {}, {})
                self.assertEqual(result["sessions"][0]["hit"], hit_after_override)
                self.assertEqual(result["sessions"][0]["first_hit_turn"], 3 if hit_after_override else None)
                summary = summarize_traces(observed.traces, [sample], result["sessions"])
                self.assertEqual(summary["failure_diagnostics"], {
                    "not_retrieved": 0, "ranking_or_policy": int(not hit_after_override), "agent_error_turns": 0,
                })
                self.assertEqual(summary["ever_ranked_recall"]["10"], 1)
                self.assertEqual(summary["ever_route_recall"], {"sparse": 1})

    def test_failure_diagnostics_partition_official_misses_by_sample_id(self):
        samples = [{"sample_id": identifier, "ground_truth": {"parent_asin": target}}
                   for identifier, target in (("hit", "a"), ("ranking-miss", "b"), ("retrieval-miss", "c"))]
        traces = [
            [{"latency_seconds": .01}],
            [{"latency_seconds": .02, "diagnostics": {"retrieved_ids": ["b"], "ranked_ids": ["b"]}}],
            [{"latency_seconds": .03, "diagnostics": {"retrieved_ids": ["other"], "ranked_ids": ["other"]}}],
        ]
        sessions = [{"sample_id": "retrieval-miss", "hit": False}, {"sample_id": "hit", "hit": True},
                    {"sample_id": "ranking-miss", "hit": False}]
        summary = summarize_traces(traces, samples, sessions)
        failures = summary["failure_diagnostics"]
        self.assertEqual(failures, {"not_retrieved": 1, "ranking_or_policy": 1, "agent_error_turns": 0})
        self.assertEqual(failures["not_retrieved"] + failures["ranking_or_policy"],
                         sum(not session["hit"] for session in sessions))
        self.assertEqual(summary["ever_ranked_recall"]["10"], 1 / 3)

    def test_summary_rejects_mismatched_result_ids(self):
        samples = [{"sample_id": "a", "ground_truth": {"parent_asin": "target"}}]
        for sessions in ([], [{"sample_id": "b", "hit": False}],
                         [{"sample_id": "a", "hit": False}, {"sample_id": "b", "hit": True}]):
            with self.subTest(sessions=sessions), self.assertRaisesRegex(ValueError, "sample_id.*do not match"):
                summarize_traces([[{"latency_seconds": .01}]], samples, sessions)

    def test_summary_rejects_duplicate_ids(self):
        sample = {"sample_id": "a", "ground_truth": {"parent_asin": "target"}}
        session = {"sample_id": "a", "hit": False}
        for samples, sessions in (([sample, sample], [session]), ([sample], [session, session])):
            with self.subTest(samples=samples, sessions=sessions), \
                    self.assertRaisesRegex(ValueError, "duplicate sample_id"):
                summarize_traces([[{"latency_seconds": .01}]] * len(samples), samples, sessions)

    def test_summary_rejects_missing_ids_and_non_boolean_hits(self):
        sample = {"sample_id": "a", "ground_truth": {"parent_asin": "target"}}
        for session in ({"hit": False}, {"sample_id": "a"}, {"sample_id": "a", "hit": 1},
                        {"sample_id": "a", "hit": "false"}):
            with self.subTest(session=session), self.assertRaises(ValueError):
                summarize_traces([[{"latency_seconds": .01}]], [sample], [session])

    def test_summary_rejects_mismatched_trace_count(self):
        with self.assertRaisesRegex(ValueError, "Trace session count"):
            summarize_traces([], [{"sample_id": "a", "ground_truth": {"parent_asin": "target"}}],
                             [{"sample_id": "a", "hit": False}])

    def test_source_hashes_include_runtime_and_development_requirements(self):
        hashes = source_hashes()
        self.assertIn("requirements.txt", hashes)
        self.assertIn("requirements-dev.txt", hashes)
        self.assertIn("experiments/freeze.py", hashes)


if __name__ == "__main__":
    unittest.main()
