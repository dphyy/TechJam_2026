import unittest

from experiments.neural_quantization_benchmark import compare


def report(arm: str, p95: float, size: int, logits: list[float], ranking: str = "rank") -> dict:
    return {
        "schema": "mercury-neural-quantization-benchmark-v1",
        "catalog_sha256": "catalog",
        "config_sha256": "config",
        "candidate_ids_sha256": "ids",
        "candidate_count": 2,
        "repetitions": 20,
        "arm": arm,
        "logits": logits,
        "ranking_sha256": ranking,
        "p95_seconds": p95,
        "serialized_model_bytes": size,
        "max_rss_bytes": 100,
    }


class NeuralQuantizationBenchmarkTest(unittest.TestCase):
    def test_comparison_reports_cost_and_drift(self):
        result = compare(
            report("float32_control", 1.0, 100, [1.0, 0.0]),
            report("dynamic_int8_qnnpack", .7, 40, [.9, .1]),
        )
        self.assertTrue(result["ranking_equal"])
        self.assertAlmostEqual(result["p95_reduction"], .3)
        self.assertAlmostEqual(result["serialized_size_reduction"], .6)
        self.assertAlmostEqual(result["maximum_logit_drift"], .1)

    def test_comparison_rejects_mismatched_inputs(self):
        candidate = report("dynamic_int8_qnnpack", .7, 40, [.9, .1])
        candidate["candidate_ids_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "fixed inputs"):
            compare(report("float32_control", 1.0, 100, [1.0, 0.0]), candidate)


if __name__ == "__main__":
    unittest.main()
