import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.cache_benchmark import run_cache_benchmark
from experiments.neural_batch_benchmark import run_batch_benchmark


class CacheBenchmarkTest(unittest.TestCase):
    def test_independent_sessions_report_semantic_digest_and_cache_totals(self):
        class FakeAgent:
            def __init__(self, catalog, config):
                self.startup_fallbacks = {}
                self.last_diagnostics = {}
                self.turns = 0

            def reset(self, session_id, profile):
                pass

            def respond(self, session_id, message, turn, top_k):
                self.turns += 1
                self.last_diagnostics = {
                    "fallbacks": [],
                    "neural_logit_cache": {
                        "enabled": True, "capacity": 8, "size": 2,
                        "hits": max(0, (self.turns - 1) * 2), "misses": 2,
                        "evictions": 0, "evaluated_pairs": 2, "turn": {},
                    },
                }
                return {
                    "message": "Here are options.", "ask_attribute": None,
                    "recommendations": [{"parent_asin": "a"}, {"parent_asin": "b"}],
                    "usage": {"prompt_tokens": 2 if self.turns == 1 else 0,
                              "completion_tokens": 0},
                }

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = root / "catalog.jsonl"
            config = root / "config.json"
            catalog.write_text(json.dumps({"parent_asin": "a", "title": "shirt"}) + "\n")
            config.write_text("{}\n")
            with patch("experiments.cache_benchmark.Agent", FakeAgent), \
                    patch("experiments.cache_benchmark.source_hashes", return_value={"source": "same"}):
                report = run_cache_benchmark(catalog, config, sessions=3)
        self.assertEqual(report["sessions"], 3)
        self.assertEqual(report["prompt_tokens"], 2)
        self.assertEqual(report["cache"]["hits"], 4)
        self.assertEqual(report["cache"]["evaluated_pairs"], 2)
        self.assertEqual(report["fallback_turns"], 0)
        self.assertEqual(len(report["semantic_responses_sha256"]), 64)
        self.assertFalse(report["source_changed_during_run"])

    def test_rejects_invalid_workload_parameters(self):
        with self.assertRaises(ValueError):
            run_cache_benchmark(Path("catalog"), Path("config"), sessions=0)
        with self.assertRaises(ValueError):
            run_cache_benchmark(Path("catalog"), Path("config"), message="")

    def test_batch_benchmark_rejects_unregistered_matrix_values(self):
        for threads, batch_size, repetitions in (
            (1, 16, 20), (4, 15, 20), (4, 16, 1), (4.0, 16, 20),
        ):
            with self.subTest(values=(threads, batch_size, repetitions)), self.assertRaises(ValueError):
                run_batch_benchmark(
                    Path("catalog"), Path("config"), threads, batch_size, repetitions,
                )


if __name__ == "__main__":
    unittest.main()
