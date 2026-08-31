import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mercury.model_assets import MODELS, file_sha256, model_asset_identity, verify_model
from mercury.types import Candidate
from mercury.catalog import product_from_dict
from tests.test_neural import DeterministicFakeCrossEncoder, cached_ranker


class ModelIntegrityTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for name, content in {
            "model.safetensors": "fixture weights",
            "config.json": '{"hidden_size": 8}',
            "tokenizer_config.json": '{"do_lower_case": true}',
            "vocab.txt": "shirt\nblue\n",
        }.items():
            (self.root / name).write_text(content)
        files = {p.name: file_sha256(p) for p in self.root.iterdir()}
        metadata = {name: value for name, value in files.items() if name != "model.safetensors"}
        self.manifest = {"revision": "fixture-revision", "files": files}
        self.spec = {
            "revision": "fixture-revision", "weights_sha256": files["model.safetensors"],
            "required": ["model.safetensors", "config.json", "tokenizer_config.json"],
            "metadata_sha256": hashlib.sha256(json.dumps(
                metadata, sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest(),
        }
        self.write_manifest()
        self.patched = patch.dict(MODELS, {"fixture": self.spec})
        self.patched.start()
        self.addCleanup(self.patched.stop)

    def write_manifest(self):
        (self.root / "asset_manifest.json").write_text(json.dumps(self.manifest))

    def test_intact_bundle_is_verified_and_has_a_stable_identity(self):
        manifest = verify_model(self.root, "fixture")
        identity = model_asset_identity(manifest)
        self.assertRegex(identity, r"^[a-f0-9]{64}$")
        self.manifest["files"] = dict(reversed(list(self.manifest["files"].items())))
        self.write_manifest()
        self.assertEqual(model_asset_identity(verify_model(self.root, "fixture")), identity)

    def test_rewritten_manifest_cannot_authorize_changed_loader_metadata(self):
        for name in ("config.json", "tokenizer_config.json", "vocab.txt"):
            with self.subTest(name=name):
                target = self.root / name
                original = target.read_text()
                target.write_text(original + "\nchanged")
                self.manifest["files"][name] = file_sha256(target)
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, "loader metadata checksum"):
                    verify_model(self.root, "fixture")
                target.write_text(original)
                self.manifest["files"][name] = file_sha256(target)
                self.write_manifest()

    def test_unmanifested_tokenizer_override_is_rejected(self):
        (self.root / "tokenizer.json").write_text('{"added_tokens": []}')
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            verify_model(self.root, "fixture")

    def test_loader_file_cannot_be_removed_from_manifest_and_disk(self):
        # A second vocabulary would otherwise satisfy the loose vocabulary rule.
        (self.root / "tokenizer.json").write_text("{}")
        self.manifest["files"]["tokenizer.json"] = file_sha256(self.root / "tokenizer.json")
        del self.manifest["files"]["vocab.txt"]
        (self.root / "vocab.txt").unlink()
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "loader metadata checksum"):
            verify_model(self.root, "fixture")

    def test_noncanonical_manifest_paths_are_rejected(self):
        self.manifest["files"]["./config.json"] = self.manifest["files"]["config.json"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "manifest path"):
            verify_model(self.root, "fixture")

    def test_documentation_does_not_change_inference_identity(self):
        before = model_asset_identity(verify_model(self.root, "fixture"))
        (self.root / "NOTES.md").write_text("local documentation")
        self.manifest["files"]["NOTES.md"] = file_sha256(self.root / "NOTES.md")
        self.write_manifest()
        self.assertEqual(model_asset_identity(verify_model(self.root, "fixture")), before)

    def test_metadata_identity_changes_without_a_weight_change(self):
        before = model_asset_identity(self.manifest)
        self.manifest["files"]["config.json"] = "a" * 64
        self.assertNotEqual(model_asset_identity(self.manifest), before)


class NeuralBackendIdentityTest(unittest.TestCase):
    def test_replacing_loaded_model_forces_new_inference(self):
        ranker = cached_ranker()
        item = Candidate(product_from_dict({"parent_asin": "a", "title": "Blue shirt"}), 1.0)
        original = ranker.model
        ranker.score("shirt", [item])
        identity = ranker.backend_identity
        replacement = DeterministicFakeCrossEncoder()
        ranker.model = replacement
        ranker.score("shirt", [item])
        self.assertNotEqual(ranker.backend_identity, identity)
        self.assertEqual(len(replacement.predict_calls), 1)
        # Returning to a previously loaded object is still a new backend epoch.
        ranker.model = original
        ranker.score("shirt", [item])
        self.assertEqual(len(original.predict_calls), 2)
        self.assertEqual(ranker.cache_stats()["hits"], 0)

    def test_changed_verified_assets_invalidate_pair_cache(self):
        ranker = cached_ranker()
        item = Candidate(product_from_dict({"parent_asin": "a", "title": "Blue shirt"}), 1.0)
        ranker.asset_identity = "a" * 64
        ranker.score("shirt", [item])
        ranker.asset_identity = "b" * 64
        ranker.score("shirt", [item])
        self.assertEqual(len(ranker.model.predict_calls), 2)


if __name__ == "__main__":
    unittest.main()
