import unittest

from experiments.prepare import partition_samples


class SplitTest(unittest.TestCase):
    def test_deterministic_grouped_split(self):
        rows = [{"sample_id": str(i), "scenario_type": "buying" if i < 10 else "browsing",
                 "ground_truth": {"parent_asin": str(i // 2)}} for i in range(20)]
        development, reserved = partition_samples(rows)
        self.assertEqual(partition_samples(rows), (development, reserved))
        left = {r["ground_truth"]["parent_asin"] for r in development}
        right = {r["ground_truth"]["parent_asin"] for r in reserved}
        self.assertFalse(left & right)
        self.assertEqual(len(development) + len(reserved), 20)
        self.assertTrue(reserved)

    def test_requires_unique_sample_ids(self):
        rows = [{"sample_id": "x", "scenario_type": "buying", "ground_truth": {"parent_asin": "A"}}] * 2
        with self.assertRaises(ValueError):
            partition_samples(rows)
