import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from mercury.admission import (
    FEATURE_NAMES,
    FEATURE_VERSION_V2,
    MODEL_SCHEMA_V2,
    AdmissionFeatureCache,
    AdmissionModel,
    adaptive_rerank_depth,
    admission_features,
    admission_features_v2,
    score_all_candidates,
    select_rerank_prefix,
)
from mercury.catalog import Catalog
from mercury.types import Candidate, Preference, Product


def candidate(identifier: str, title: str, score: float) -> Candidate:
    return Candidate(Product(identifier, title, {"title": title}), score)


class RerankAdmissionTest(unittest.TestCase):
    def test_prefix_preserves_the_existing_ranking(self):
        candidates = [candidate(f"p{index}", "shirt", 100 - index) for index in range(12)]
        selected = select_rerank_prefix(candidates, [], 5, "prefix")
        self.assertEqual([item.product.parent_asin for item in selected], ["p0", "p1", "p2", "p3", "p4"])
        self.assertTrue(all(left is right for left, right in zip(selected, candidates)))

    def test_stratified_keeps_leaders_and_spreads_the_tail_without_duplicates(self):
        candidates = [candidate(f"p{index:02d}", "shirt", 100 - index) for index in range(30)]
        selected = select_rerank_prefix(candidates, [], 10, "stratified")
        identifiers = [item.product.parent_asin for item in selected]
        self.assertEqual(identifiers, ["p00", "p01", "p02", "p03", "p04", "p05", "p11", "p17", "p23", "p29"])
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_cover_admits_low_ranked_source_supported_rare_preference(self):
        candidates = [candidate(f"p{index:02d}", "cotton shirt", 100 - index) for index in range(39)]
        candidates.append(candidate("blue", "blue cotton shirt", 1.0))
        preferences = [
            Preference("material", "cotton", 1, "cotton", hard=True),
            Preference("color", "blue", 1, "blue"),
            Preference("color", "black", 1, "black", active=False),
        ]
        selected = select_rerank_prefix(candidates, preferences, 20, "cover")
        identifiers = [item.product.parent_asin for item in selected]
        self.assertEqual(identifiers[:10], [f"p{index:02d}" for index in range(10)])
        self.assertIn("blue", identifiers)
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual([item.product.parent_asin for item in candidates][-1], "blue")

    def test_rejects_unknown_modes_and_bad_limits(self):
        candidates = [candidate("one", "shirt", 1.0)]
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            select_rerank_prefix(candidates, [], 1, "oracle")
        with self.assertRaisesRegex(ValueError, "positive"):
            select_rerank_prefix(candidates, [], 0, "prefix")

    def test_fusion_scores_the_full_pool_and_promotes_supported_tail(self):
        candidates = [candidate(f"p{index:02d}", "black polyester formal shirt", 1 - index / 100) for index in range(24)]
        candidates.append(candidate("target", "blue cotton travel shirt", .75))
        candidates.extend(candidate(f"tail{index:02d}", "black polyester formal shirt", .74 - index / 100)
                          for index in range(15))
        preferences = [
            Preference("category", "shirts", 1, "shirts", hard=True),
            Preference("color", "blue", 1, "blue"),
            Preference("material", "cotton", 1, "cotton"),
            Preference("use_case", "travel", 1, "travel"),
        ]
        ordered, diagnostics = score_all_candidates(candidates, preferences, None, "fusion")
        self.assertEqual(len(ordered), 40)
        self.assertEqual({item.product.parent_asin for item in ordered}, {item.product.parent_asin for item in candidates})
        self.assertLess([item.product.parent_asin for item in ordered].index("target"), 20)
        self.assertEqual(diagnostics["pool_size"], 40)

    def test_features_leave_missing_price_neutral(self):
        rows = admission_features(
            [candidate("unknown", "blue shirt", 1.0)],
            [Preference("budget", "<=20", 1, "under 20", hard=True)],
            None,
        )
        self.assertEqual(rows[0]["price_compatibility"], 0.0)

    def test_linear_model_is_hash_bound_to_catalog_and_feature_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            catalog_path.write_text(json.dumps({
                "parent_asin": "A", "title": "Hat", "categories": ["Hats"],
            }) + "\n", encoding="utf-8")
            catalog_hash = Catalog(catalog_path).sha256
            model_path = root / "model.json"
            model_path.write_text(json.dumps({
                "schema": "mercury-admission-linear-v1",
                "feature_version": "admission-features-v1",
                "feature_names": list(FEATURE_NAMES),
                "mean": [0.0] * len(FEATURE_NAMES),
                "scale": [1.0] * len(FEATURE_NAMES),
                "coefficients": [1.0] * len(FEATURE_NAMES),
                "intercept": 0.0,
                "catalog_sha256": catalog_hash,
                "training_sha256": "training",
                "validation_sha256": "validation",
            }), encoding="utf-8")
            model = AdmissionModel.load(model_path, catalog_hash)
            self.assertEqual(model.model_sha256, hashlib.sha256(model_path.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, "catalog hash"):
                AdmissionModel.load(model_path, "different")

    def test_v2_uses_precomputed_tokens_and_preserves_unknown_safe_coverage(self):
        product = Product(
            "one", "Blue cotton shirt", {
                "title": "Blue cotton shirt",
                "categories": "Clothing > Shirts",
                "features": "Breathable",
            },
        )
        candidates = [Candidate(product, 1.0)]
        preferences = [Preference("color", "blue", 1, "blue")]
        cache = AdmissionFeatureCache([product])
        rows = admission_features_v2(candidates, preferences, None, cache)
        self.assertEqual(rows[0]["positive_coverage"], 0.6)
        self.assertEqual(rows[0]["all_field_coverage"], 1.0)

        unknown_query = [Preference("other", "blue unobtainium", 1, "blue unobtainium")]
        rows = admission_features_v2(candidates, unknown_query, None, cache)
        self.assertEqual(rows[0]["positive_coverage"], 0.0)
        self.assertEqual(rows[0]["all_field_coverage"], 0.5)

    def test_v2_model_schema_loads_with_frozen_feature_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            catalog_path.write_text(json.dumps({
                "parent_asin": "A", "title": "Hat", "categories": ["Hats"],
            }) + "\n", encoding="utf-8")
            catalog_hash = Catalog(catalog_path).sha256
            model_path = root / "model.json"
            model_path.write_text(json.dumps({
                "schema": MODEL_SCHEMA_V2,
                "feature_version": FEATURE_VERSION_V2,
                "feature_names": list(FEATURE_NAMES),
                "mean": [0.0] * len(FEATURE_NAMES),
                "scale": [1.0] * len(FEATURE_NAMES),
                "coefficients": [1.0] * len(FEATURE_NAMES),
                "intercept": 0.0,
                "catalog_sha256": catalog_hash,
                "training_sha256": "training",
                "validation_sha256": "validation",
            }), encoding="utf-8")
            model = AdmissionModel.load(model_path, catalog_hash)
            self.assertEqual(model.feature_version, FEATURE_VERSION_V2)

    def test_adaptive_depth_uses_only_the_predeclared_admission_gap(self):
        rows = [candidate(f"p{index}", "shirt", 1.0) for index in range(30)]
        for index, row in enumerate(rows):
            row.route_scores["admission_score"] = 1.0 - index * 0.01
        depth, diagnostics = adaptive_rerank_depth(rows, 20, 30, 0.10)
        self.assertEqual(depth, 20)
        self.assertEqual(diagnostics["reason"], "admission_gap_confident")

        rows[29].route_scores["admission_score"] = rows[19].route_scores["admission_score"]
        depth, diagnostics = adaptive_rerank_depth(rows, 20, 30, 0.10)
        self.assertEqual(depth, 30)
        self.assertEqual(diagnostics["reason"], "admission_gap_uncertain")
