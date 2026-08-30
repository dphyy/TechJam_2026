import json
import tempfile
import unittest
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import RLock
from unittest.mock import patch

import numpy as np

from mercury.catalog import product_from_dict
from mercury.model_assets import MODELS, file_sha256, verify_model
from mercury.neural import (DOCUMENT_VERSION, document_text, fuse_neural_logits,
                            structured_document_text, validate_dense_manifest)


class DeterministicFakeCrossEncoder:
    def __init__(self):
        self.predict_calls = []
        self.predict_options = []

    def tokenizer(self, left, right, **kwargs):
        return {"input_ids": [[1] * max(1, len(document.split())) for document in right]}

    def predict(self, pairs, **kwargs):
        self.predict_calls.append(list(pairs))
        self.predict_options.append(dict(kwargs))
        return np.array([
            sum((index + 1) * byte for index, byte in enumerate(document.encode("utf-8"))) % 10007
            for _, document in pairs
        ], dtype=float)


def cached_ranker(capacity=8, model=None):
    from mercury.neural import NeuralRanker

    ranker = NeuralRanker.__new__(NeuralRanker)
    ranker.model = model or DeterministicFakeCrossEncoder()
    ranker.kind = "reranker"
    ranker.device = "cpu"
    ranker.threads = 4
    ranker.batch_size = 16
    ranker.prompt_tokens = 0
    ranker._cache_capacity = capacity
    ranker._logit_cache = OrderedDict()
    ranker._cache_lock = RLock()
    ranker._score_lock = RLock()
    ranker._cache_hits = 0
    ranker._cache_misses = 0
    ranker._cache_evictions = 0
    ranker._evaluated_pairs = 0
    return ranker


class NeuralAssetsTest(unittest.TestCase):
    def test_exact_pair_cache_restores_scores_in_the_requested_order(self):
        from mercury.types import Candidate

        candidates = [
            Candidate(product_from_dict({"parent_asin": value, "title": title}), score)
            for value, title, score in (("a", "Blue shirt", 2.0), ("b", "Red shirt", 1.0))
        ]
        ranker = cached_ranker()
        first = ranker.score("cotton shirt", candidates)
        second = ranker.score("cotton shirt", list(reversed(candidates)))
        self.assertEqual(list(second), ["b", "a"])
        self.assertEqual(second, {"b": first["b"], "a": first["a"]})
        self.assertEqual(len(ranker.model.predict_calls), 1)
        self.assertEqual(ranker.cache_stats(), {
            "enabled": True, "capacity": 8, "size": 2, "hits": 2,
            "misses": 2, "evictions": 0, "evaluated_pairs": 2,
        })

    def test_pair_cache_invalidates_on_query_id_document_and_mode_changes(self):
        from mercury.types import Candidate

        ranker = cached_ranker()
        original = Candidate(product_from_dict({"parent_asin": "a", "title": "Blue shirt"}), 1.0)
        changed_document = Candidate(
            product_from_dict({"parent_asin": "a", "title": "Blue linen shirt"}), 1.0,
        )
        changed_id = Candidate(product_from_dict({"parent_asin": "b", "title": "Blue shirt"}), 1.0)
        ranker.score("shirt", [original])
        ranker.score("different query", [original])
        ranker.score("shirt", [changed_document])
        ranker.score("shirt", [changed_id])
        ranker.score("shirt", [original], document_mode="lexical")
        stats = ranker.cache_stats()
        self.assertEqual(stats["misses"], 5)
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["evaluated_pairs"], 5)

    def test_pair_cache_key_pins_model_serializer_and_sequence_provenance(self):
        from mercury.types import Candidate

        ranker = cached_ranker()
        candidate = Candidate(product_from_dict({"parent_asin": "a", "title": "Blue shirt"}), 1.0)
        base = ranker._pair_cache_key("shirt", "Title: Blue shirt", candidate, "head", False)
        with patch.dict(MODELS["reranker"], {"revision": "changed-revision"}):
            changed_model = ranker._pair_cache_key(
                "shirt", "Title: Blue shirt", candidate, "head", False,
            )
        with patch("mercury.neural.DOCUMENT_VERSION", "changed-document-version"):
            changed_serializer = ranker._pair_cache_key(
                "shirt", "Title: Blue shirt", candidate, "head", False,
            )
        with patch("mercury.neural.MAX_LENGTH", 128):
            changed_length = ranker._pair_cache_key(
                "shirt", "Title: Blue shirt", candidate, "head", False,
            )
        ranker.batch_size = 30
        changed_batch = ranker._pair_cache_key(
            "shirt", "Title: Blue shirt", candidate, "head", False,
        )
        self.assertEqual(
            len({base, changed_model, changed_serializer, changed_length, changed_batch}), 5,
        )

    def test_pair_cache_uses_deterministic_lru_eviction(self):
        from mercury.types import Candidate

        ranker = cached_ranker(capacity=2)
        candidates = {
            value: Candidate(product_from_dict({"parent_asin": value, "title": title}), 1.0)
            for value, title in (("a", "A shirt"), ("b", "B shirt"), ("c", "C shirt"))
        }
        ranker.score("shirt", [candidates["a"], candidates["b"]])
        ranker.score("shirt", [candidates["a"]])
        ranker.score("shirt", [candidates["c"]])
        self.assertEqual(
            [key[7] for key in ranker._logit_cache], ["a", "c"],
        )
        ranker.score("shirt", [candidates["b"]])
        self.assertEqual([key[7] for key in ranker._logit_cache], ["c", "b"])
        self.assertEqual(ranker.cache_stats()["evictions"], 2)

    def test_pair_cache_serializes_concurrent_lookups_without_duplicate_inference(self):
        from mercury.types import Candidate

        ranker = cached_ranker()
        candidates = [
            Candidate(product_from_dict({"parent_asin": value, "title": f"{value} shirt"}), 1.0)
            for value in ("a", "b")
        ]
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(
                lambda _: ranker.score("shirt", candidates), range(4),
            ))
        self.assertTrue(all(result == results[0] for result in results))
        self.assertEqual(len(ranker.model.predict_calls), 1)
        stats = ranker.cache_stats()
        self.assertEqual(stats["evaluated_pairs"], 2)
        self.assertEqual(stats["hits"], 6)

    def test_cache_enabled_and_disabled_rankings_are_exactly_equal(self):
        from mercury.types import Candidate

        candidates = [
            Candidate(product_from_dict({"parent_asin": str(index), "title": f"Shirt {index}"}),
                      5.0 - index, {"sparse": 1.0 / (index + 1)})
            for index in range(5)
        ]
        control = cached_ranker(capacity=0)
        candidate = cached_ranker(capacity=8)
        control_ranking = control.rank("shirt", candidates, 4, .75)
        candidate_ranking = candidate.rank("shirt", candidates, 4, .75)
        def snapshot(ranking):
            return [(item.product.parent_asin, item.score, item.route_scores) for item in ranking]
        self.assertEqual(snapshot(candidate_ranking), snapshot(control_ranking))

    def test_failed_scores_are_not_inserted_into_the_pair_cache(self):
        from mercury.types import Candidate

        class NonfiniteModel(DeterministicFakeCrossEncoder):
            def predict(self, pairs, **kwargs):
                self.predict_calls.append(list(pairs))
                return np.array([float("nan") for _ in pairs])

        ranker = cached_ranker(model=NonfiniteModel())
        candidate = Candidate(product_from_dict({"parent_asin": "a", "title": "Blue shirt"}), 1.0)
        for _ in range(2):
            with self.assertRaises(ValueError):
                ranker.score("shirt", [candidate])
        self.assertEqual(len(ranker.model.predict_calls), 2)
        stats = ranker.cache_stats()
        self.assertEqual(stats["size"], 0)
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 2)

    def test_ranker_uses_the_configured_inference_batch_size(self):
        from mercury.types import Candidate

        ranker = cached_ranker(capacity=0)
        ranker.batch_size = 30
        candidate = Candidate(product_from_dict({"parent_asin": "a", "title": "Blue shirt"}), 1.0)
        ranker.score("shirt", [candidate])
        self.assertEqual(ranker.model.predict_options[0]["batch_size"], 30)

    def test_progressive_logit_fusion_can_promote_a_scored_tail_candidate(self):
        from mercury.types import Candidate

        candidates = [Candidate(product_from_dict({"parent_asin": str(i), "title": "shirt"}), 5.0 - i)
                      for i in range(5)]
        first = fuse_neural_logits(candidates, {"0": 2.0, "1": 1.0}, .75)
        self.assertEqual([item.product.parent_asin for item in first[:2]], ["0", "1"])
        expanded = fuse_neural_logits(candidates, {"0": 2.0, "1": 1.0, "3": 4.0}, .75)
        self.assertEqual(expanded[0].product.parent_asin, "3")
        self.assertTrue(all(left.score >= right.score for left, right in zip(expanded, expanded[1:])))

    def test_progressive_logit_fusion_rejects_unknown_or_nonfinite_scores(self):
        from mercury.types import Candidate

        candidates = [Candidate(product_from_dict({"parent_asin": "a", "title": "shirt"}), 1.0)]
        for logits in ({"missing": 1.0}, {"a": float("nan")}):
            with self.subTest(logits=logits), self.assertRaises(ValueError):
                fuse_neural_logits(candidates, logits, .75)

    def test_reranked_prefix_and_tail_use_monotone_comparable_scores(self):
        from mercury.neural import NeuralRanker
        from mercury.types import Candidate

        class FakeModel:
            def tokenizer(self, left, right, **kwargs):
                return {"input_ids": [[1, 2] for _ in left]}

            def predict(self, pairs, **kwargs):
                return np.array([2.0, 1.0])

        ranker = NeuralRanker.__new__(NeuralRanker)
        ranker.model, ranker.prompt_tokens = FakeModel(), 0
        candidates = [Candidate(product_from_dict({"parent_asin": str(i), "title": "shirt"}), 5.0 - i)
                      for i in range(3)]
        ranked = ranker.rank("shirt", candidates, 2, .25)
        self.assertEqual([item.product.parent_asin for item in ranked], ["0", "1", "2"])
        self.assertTrue(all(left.score >= right.score for left, right in zip(ranked, ranked[1:])))
        self.assertEqual(candidates[-1].score, 3.0)
        for item in candidates:
            item.route_scores["constraint_penalty"] = 8.0
        ranked = ranker.rank("shirt", candidates, 2, .25)
        self.assertTrue(all("constraint_penalty" not in item.route_scores for item in ranked))

    def test_low_neural_margin_conservatively_reduces_fusion_weight(self):
        from mercury.neural import NeuralRanker
        from mercury.types import Candidate

        class FakeModel:
            def tokenizer(self, left, right, **kwargs):
                return {"input_ids": [[1, 2] for _ in left]}

            def predict(self, pairs, **kwargs):
                return np.array([0.20, 0.25, 0.22])

        ranker = NeuralRanker.__new__(NeuralRanker)
        ranker.model, ranker.prompt_tokens = FakeModel(), 0
        candidates = [Candidate(product_from_dict({"parent_asin": str(i), "title": "shirt"}), 3.0 - i)
                      for i in range(3)]
        ranked = ranker.rank(
            "shirt", candidates, 3, .75, low_margin_weight=.50, margin_threshold=1.0,
        )
        self.assertEqual({item.route_scores["neural_fusion_weight"] for item in ranked}, {.50})
        self.assertTrue(all(abs(item.route_scores["neural_margin"] - .03) < 1e-9 for item in ranked))

    def test_confident_neural_margin_keeps_selected_fusion_weight(self):
        from mercury.neural import NeuralRanker
        from mercury.types import Candidate

        class FakeModel:
            def tokenizer(self, left, right, **kwargs):
                return {"input_ids": [[1, 2] for _ in left]}

            def predict(self, pairs, **kwargs):
                return np.array([3.0, 1.0])

        ranker = NeuralRanker.__new__(NeuralRanker)
        ranker.model, ranker.prompt_tokens = FakeModel(), 0
        candidates = [Candidate(product_from_dict({"parent_asin": str(i), "title": "shirt"}), 2.0 - i)
                      for i in range(2)]
        ranked = ranker.rank(
            "shirt", candidates, 2, .75, low_margin_weight=.50, margin_threshold=1.0,
        )
        self.assertEqual({item.route_scores["neural_fusion_weight"] for item in ranked}, {.75})

    def test_document_view_is_bounded_and_grounded(self):
        product = product_from_dict({"parent_asin": "a", "title": "Cotton shirt",
                                     "description": "word " * 2000})
        text = document_text(product)
        self.assertTrue(text.startswith("Title: Cotton shirt"))
        self.assertLess(len(text), 5000)
        self.assertNotIn("a\n", text)

    def test_structured_document_prioritizes_type_role_and_price(self):
        product = product_from_dict({"parent_asin": "a", "title": "Replacement shoe laces",
                                     "categories": ["Shoe Accessories"], "price": 8.5,
                                     "description": "word " * 2000})
        text = structured_document_text(product)
        self.assertTrue(text.startswith("Product role: component"))
        self.assertIn("Price: 8.5", text)
        self.assertLess(len(text), 5000)

    def test_ranker_uses_structured_document_when_enabled(self):
        from mercury.neural import NeuralRanker
        from mercury.types import Candidate

        class FakeModel:
            def __init__(self):
                self.pairs = None

            def tokenizer(self, left, right, **kwargs):
                return {"input_ids": [[1, 2] for _ in left]}

            def predict(self, pairs, **kwargs):
                self.pairs = pairs
                return np.array([1.0])

        ranker = NeuralRanker.__new__(NeuralRanker)
        ranker.model, ranker.prompt_tokens = FakeModel(), 0
        candidate = Candidate(product_from_dict({"parent_asin": "a", "title": "Cotton shirt",
                                                  "categories": ["Shirts"]}), 1.0)
        ranker.rank("Mode: buying\nObject: shirts", [candidate], 1, .75, structured=True)
        self.assertEqual(ranker.model.pairs[0][0], "Mode: buying\nObject: shirts")
        self.assertIn("Product type: clothing", ranker.model.pairs[0][1])

    def test_protected_document_keeps_title_categories_and_matching_source_spans(self):
        from mercury.types import Preference

        product = product_from_dict({
            "parent_asin": "jacket", "title": "City jacket", "categories": ["Jackets"],
            "features": ["The shell is faux leather."],
            "details": {"Material": "Wool"},
            "description": "word " * 400,
        })
        preferences = [
            Preference("material", "leather", 2, "no leather", hard=True, polarity=-1),
            Preference("material", "wool", 1, "wool", hard=True),
        ]
        text = document_text(product, "wool jacket", preferences, "protected")
        self.assertIn("Title: City jacket", text)
        self.assertIn("Categories: Jackets", text)
        self.assertIn("faux leather", text)
        self.assertIn("Details: Material Wool", text)
        self.assertLessEqual(len(text.split()), 160)

    def test_lexical_document_is_query_dependent_and_rejects_unknown_modes(self):
        product = product_from_dict({"parent_asin": "shoe", "title": "Trail shoe", "categories": ["Shoes"],
                                     "description": "Designed for wet trail running with waterproof lining."})
        self.assertIn("waterproof", document_text(product, "waterproof trail", mode="lexical"))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            document_text(product, mode="oracle")

    def test_manifest_rejects_wrong_catalog_model_or_document_version(self):
        manifest = {"catalog_sha256": "catalog", "model_revision": MODELS["embedding"]["revision"],
                    "document_version": DOCUMENT_VERSION, "count": 4, "dimensions": 384}
        validate_dense_manifest(manifest, "catalog", 4)
        for key in ("catalog_sha256", "model_revision", "document_version", "count", "dimensions"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_dense_manifest({**manifest, key: "wrong"}, "catalog", 4)

    def test_missing_model_is_explicit_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises((ValueError, FileNotFoundError)):
                verify_model(Path(directory), "embedding")

    def test_modified_model_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in MODELS["embedding"]["required"]:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            manifest = {"revision": MODELS["embedding"]["revision"],
                        "sha256": {"model.safetensors": file_sha256(root / "model.safetensors")}}
            (root / "asset_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                verify_model(root, "embedding")

    def test_dense_array_finiteness_is_available_for_validation(self):
        # A corrupt vector file must never introduce NaN scores into a legal response.
        from mercury.neural import validate_vectors
        validate_vectors(np.zeros((4, 384), dtype=np.float32), 4)
        with self.assertRaises(ValueError):
            validate_vectors(np.zeros((3, 384), dtype=np.float32), 4)
        invalid = np.zeros((4, 384), dtype=np.float32)
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_vectors(invalid, 4)

    def test_dense_query_scores_are_finite_and_preserve_identical_ids(self):
        from mercury.neural import DenseIndex

        class Encoder:
            def tokenizer(self, text, **kwargs):
                return {"input_ids": [1, 2]}

            def encode(self, texts, **kwargs):
                return np.array([[1.0, 0.0]], dtype=np.float32)

        index = DenseIndex.__new__(DenseIndex)
        index.ids = ["A", "B", "C"]
        index.vectors = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
        index.encoder, index.prompt_tokens = Encoder(), 0
        self.assertEqual(index.search("shirt", 3), ["A", "B", "C"])
        self.assertEqual(index.prompt_tokens, 2)
        with patch("numpy.einsum", return_value=np.array([np.inf, 0, 0])):
            with self.assertRaises(ValueError):
                index.search("shirt", 3)

    def test_manifest_cannot_omit_required_metadata_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in MODELS["reranker"]["required"]:
                (root / name).write_text("{}")
            manifest = {"revision": MODELS["reranker"]["revision"], "files": {}}
            (root / "asset_manifest.json").write_text(json.dumps(manifest))
            with patch("mercury.model_assets.file_sha256", return_value=MODELS["reranker"]["weights_sha256"]):
                with self.assertRaisesRegex(ValueError, "every required"):
                    verify_model(root, "reranker")


    def test_alternative_reranker_is_pinned_like_the_default(self):
        for kind in ("reranker", "bge_reranker_base"):
            with self.subTest(kind=kind):
                spec = MODELS[kind]
                self.assertRegex(spec["revision"], r"^[0-9a-f]{40}$")
                self.assertRegex(spec["weights_sha256"], r"^[0-9a-f]{64}$")
                self.assertTrue(spec["repo_id"] and spec["license"])
                self.assertIn("model.safetensors", spec["required"])
                self.assertIn("config.json", spec["required"])

    def test_local_domain_reranker_has_a_frozen_revision_and_hash(self):
        spec = MODELS["reranker_domain_v1"]
        self.assertEqual(spec["revision"], "mercury-product-domain-minilm-v1-seed-20260830")
        self.assertRegex(spec["weights_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(spec["license"], "Apache-2.0")

    def test_reranker_kind_selects_its_own_pinned_asset_directory(self):
        import sentence_transformers

        from mercury.neural import NeuralRanker

        seen = {}

        def fake_verify(path, kind):
            seen["verified"] = (Path(path), kind)
            return {}

        class FakeCrossEncoder:
            def __init__(self, path, **kwargs):
                seen["loaded"] = Path(path)

        with patch("mercury.neural.verify_model", fake_verify),                 patch.object(sentence_transformers, "CrossEncoder", FakeCrossEncoder):
            NeuralRanker(Path("artifacts"), kind="bge_reranker_base")
        expected = Path("artifacts") / "models" / "bge_reranker_base"
        self.assertEqual(seen["verified"], (expected, "bge_reranker_base"))
        self.assertEqual(seen["loaded"], expected)


if __name__ == "__main__":
    unittest.main()
