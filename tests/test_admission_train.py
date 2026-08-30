import unittest

import numpy as np

from experiments.admission_train import _query_variants, _rank_metrics
from mercury.catalog import product_from_dict


class AdmissionTrainingTest(unittest.TestCase):
    def test_query_variants_are_catalog_derived_and_deterministic(self):
        product = product_from_dict({
            "parent_asin": "A",
            "title": "Blue cotton travel shirt",
            "categories": ["Clothing", "Shirts"],
            "features": ["lightweight cotton"],
            "details": {"color": "blue"},
        })
        first = _query_variants(product, 1)
        self.assertEqual(first, _query_variants(product, 1))
        self.assertGreaterEqual(len(first), 2)
        self.assertTrue(all("A" not in query for query in first))

    def test_rank_metrics_group_queries_without_target_ids_as_features(self):
        rows = [
            {"sample_id": "s1", "query_variant": 0, "candidate_id": "target", "target": 1,
             "bm25_rank": 2, "metadata_strata": ["sparse_features"]},
            {"sample_id": "s1", "query_variant": 0, "candidate_id": "other", "target": 0,
             "bm25_rank": 1, "metadata_strata": ["complete"]},
        ]
        metrics = _rank_metrics(rows, np.asarray([2.0, 1.0]), "candidate")
        self.assertEqual(metrics["recall_at_20"], 1.0)
        self.assertEqual(metrics["conditional_mrr"], 1.0)
        self.assertEqual(metrics["slices"]["sparse_features"]["queries"], 1)


if __name__ == "__main__":
    unittest.main()
