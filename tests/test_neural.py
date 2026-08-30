import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mercury.catalog import product_from_dict
from mercury.model_assets import MODELS, file_sha256, verify_model
from mercury.neural import DOCUMENT_VERSION, document_text, validate_dense_manifest


class NeuralAssetsTest(unittest.TestCase):
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

    def test_document_view_is_bounded_and_grounded(self):
        product = product_from_dict({"parent_asin": "a", "title": "Cotton shirt",
                                     "description": "word " * 2000})
        text = document_text(product)
        self.assertTrue(text.startswith("Title: Cotton shirt"))
        self.assertLess(len(text), 5000)
        self.assertNotIn("a\n", text)

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


if __name__ == "__main__":
    unittest.main()
