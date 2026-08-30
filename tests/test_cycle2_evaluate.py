import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from experiments.cycle2_evaluate import behavior_parity, create_freeze, model_file_hashes, run_validation, verify_freeze
from mercury.config import Config
from mercury.model_assets import file_sha256


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("experiments.cycle2_evaluate.REPOSITORY", self.root))
        self.stack.enter_context(patch("experiments.cycle2_evaluate.verify_lock", return_value={"verified": True}))
        self.stack.enter_context(patch("experiments.cycle2_evaluate.model_file_hashes", return_value={}))
        self.stack.enter_context(patch("experiments.cycle2_evaluate.source_hashes", side_effect=self.hash_sources))
        previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        for name in ("mercury/agent.py", "tests/test_agent.py", "experiments/cycle2_evaluate.py", "data/catalog.jsonl",
                     "data/public_set.jsonl", "artifacts/cycle2/synthetic-targets/development.jsonl",
                     "artifacts/cycle2/synthetic-targets/validation.jsonl", "artifacts/cycle2/synthetic-targets/manifest.json",
                     "artifacts/cycle2/provenance/cycle2_prepare-original.py", "artifacts/cycle2/capability-development.json",
                     "artifacts/cycle2/capability-validation.json", "docs/CYCLE2_EXPERIMENT_PROTOCOL.md",
                     "docs/CYCLE2_ALTERNATIVES_PROTOCOL.md"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("opaque test bytes; not fixture content\n")
        config_dir = self.root / "configs"
        config_dir.mkdir()
        for mode, alternative in (("frozen", "off"), ("parse", "parse"), ("grouped", "grouped")):
            (config_dir / f"cycle2_{mode}.json").write_text(json.dumps(Config(alternatives_mode=alternative).to_dict()))
        (config_dir / "selected.json").write_text(json.dumps(Config().to_dict()))
        pack = self.root / "artifacts/cycle2"
        (pack / "capability-manifest.json").write_text(json.dumps({
            "schema": "cycle2-capability-lock-v1", "sha256": {
                name: file_sha256(pack / name) for name in ("capability-development.json", "capability-validation.json")}}))

    def hash_sources(self):
        return {path.relative_to(self.root).as_posix(): file_sha256(path)
                for directory in ("mercury", "tests", "experiments")
                for path in sorted((self.root / directory).glob("*.py"))}

    def freeze(self):
        return Path(create_freeze("Development gates passed; test fixture only")["manifest_path"])

    def test_freeze_is_create_only_binds_sources_configs_and_every_input(self):
        manifest_path = self.freeze()
        manifest = verify_freeze(manifest_path)
        self.assertEqual(set(manifest["frozen"]["configs"]), {"frozen", "parse", "grouped"})
        self.assertIn("tests/test_agent.py", manifest["frozen"]["source_hashes"])
        self.assertIn("capability_validation", manifest["frozen"]["inputs"])
        self.assertIn("target_validation", manifest["frozen"]["inputs"])
        self.assertTrue((manifest_path.parent / "source/tests/test_agent.py").is_file())
        with self.assertRaises(FileExistsError):
            self.freeze()

    def test_reason_and_equal_control_budgets_are_required(self):
        with self.assertRaises(ValueError):
            create_freeze(" ")
        path = self.root / "configs/cycle2_grouped.json"
        value = json.loads(path.read_text())
        value["candidate_limit"] = 121
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "budget|configuration"):
            self.freeze()

    def test_changed_data_source_new_source_config_snapshot_and_manifest_fail(self):
        manifest_path = self.freeze()
        paths = [self.root / "data/catalog.jsonl", self.root / "tests/test_agent.py",
                 self.root / "configs/cycle2_parse.json", manifest_path.parent / "source/mercury/agent.py",
                 manifest_path]
        for path in paths:
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.subTest(path=path), self.assertRaises(ValueError):
                verify_freeze(manifest_path)
            path.write_bytes(original)
        (self.root / "mercury/new.py").write_text("new = True\n")
        with self.assertRaises(ValueError):
            verify_freeze(manifest_path)

    def test_consumption_exists_before_callback_and_cannot_change_output_to_repeat(self):
        manifest_path = self.freeze()
        observations = []

        def runner(job):
            marker = manifest_path.parent / "consumption/grouped-targets.json"
            observations.append(json.loads(marker.read_text()))
            self.assertEqual(job["dataset"], str(self.root / "artifacts/cycle2/synthetic-targets/validation.jsonl"))
            return {"test": True}

        run_validation(manifest_path, "grouped", "targets", self.root / "first", runner=runner)
        self.assertEqual(len(observations), 1)
        with self.assertRaises(FileExistsError):
            run_validation(manifest_path, "grouped", "targets", self.root / "another/place", runner=runner)
        self.assertEqual(len(observations), 1)
        receipt = json.loads((manifest_path.parent / "consumption/grouped-targets.result.json").read_text())
        self.assertEqual(receipt["status"], "completed")

    def test_failed_execution_remains_consumed_and_has_failure_receipt(self):
        manifest_path = self.freeze()
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            run_validation(manifest_path, "parse", "capabilities", self.root / "out",
                           runner=lambda job: (_ for _ in ()).throw(RuntimeError("inference failed")))
        receipt = manifest_path.parent / "consumption/parse-capabilities.result.json"
        self.assertEqual(json.loads(receipt.read_text())["status"], "failed")
        with self.assertRaises(FileExistsError):
            run_validation(manifest_path, "parse", "capabilities", self.root / "other", runner=lambda job: None)

    def test_protected_output_paths_fail_before_consumption_or_inference(self):
        manifest_path = self.freeze()
        for output in (self.root, manifest_path.parent, manifest_path.parent.parent,
                       manifest_path.parent / "new-output"):
            with self.subTest(output=output):
                expected = FileExistsError if output.exists() else ValueError
                with self.assertRaises(expected):
                    run_validation(manifest_path, "grouped", "targets", output,
                                   runner=lambda job: self.fail("Protected output reached inference"))
                self.assertFalse((manifest_path.parent / "consumption/grouped-targets.json").exists())
                self.assertFalse((manifest_path.parent / "consumption/grouped-targets.result.json").exists())
                verify_freeze(manifest_path)

    def test_source_changed_during_callback_fails_post_verification(self):
        manifest_path = self.freeze()

        def runner(job):
            (self.root / "mercury/agent.py").write_text("changed\n")

        with self.assertRaises(ValueError):
            run_validation(manifest_path, "frozen", "targets", self.root / "out", runner=runner)
        receipt = manifest_path.parent / "consumption/frozen-targets.result.json"
        self.assertEqual(json.loads(receipt.read_text())["status"], "failed")

    def test_invalid_mode_kind_or_copied_manifest_rejected_without_callback(self):
        manifest_path = self.freeze()
        for mode, kind in (("other", "targets"), ("frozen", "development")):
            with self.subTest(mode=mode, kind=kind), self.assertRaises(ValueError):
                run_validation(manifest_path, mode, kind, self.root / "out", runner=lambda job: self.fail())
        copied = self.root / "copied.json"
        copied.write_bytes(manifest_path.read_bytes())
        with self.assertRaises(ValueError):
            run_validation(copied, "frozen", "targets", self.root / "out", runner=lambda job: self.fail())

    def test_model_inventory_includes_verified_manifest_and_all_declared_files(self):
        directory = self.root / "artifacts/models/reranker"
        directory.mkdir(parents=True)
        for name in ("asset_manifest.json", "model.safetensors", "tokenizer.json"):
            (directory / name).write_text("small fixture")
        with patch("experiments.cycle2_evaluate.verify_model", return_value={
            "files": {"model.safetensors": "unused", "tokenizer.json": "unused"}}) as verify:
            inventory = model_file_hashes(Config(neural_rerank=True))
        verify.assert_called_once_with(directory, "reranker")
        self.assertEqual(set(inventory), {str(directory / name) for name in (
            "asset_manifest.json", "model.safetensors", "tokenizer.json")})

    def test_behavior_parity_compares_full_responses_and_required_diagnostics_by_session_id(self):
        directories = [self.root / name for name in ("original", "candidate")]
        sessions = [{"sample_id": name, "hit": True} for name in ("a", "b")]
        traces = [[{"turn": 1, "message": name, "response": {"message": "ok", "usage": {"prompt_tokens": 3}},
                    "diagnostics": {"query": "bag", "revision": 1, "cache_hit": False, "preferences": [],
                                    "routes": {}, "retrieved_ids": [name], "ranked_ids": [name], "fallbacks": [],
                                    "policy": {}, "latency_seconds": 123}}] for name in ("a", "b")]
        for index, directory in enumerate(directories):
            directory.mkdir()
            (directory / "result.json").write_text(json.dumps({"technical_score": 1.0, "sessions": sessions[::1 - 2 * index]}))
            (directory / "traces.json").write_text(json.dumps(traces[::1 - 2 * index]))
            (directory / "manifest.json").write_text(json.dumps({"source_changed_during_run": False}))
        compared = behavior_parity(*directories)
        self.assertTrue(compared["matched"])
        self.assertEqual(compared["session_count"], 2)
        self.assertEqual(compared["turn_count"], 2)
        value = json.loads((directories[1] / "traces.json").read_text())
        value[0][0]["response"]["usage"]["prompt_tokens"] = 4
        (directories[1] / "traces.json").write_text(json.dumps(value))
        self.assertFalse(behavior_parity(*directories)["matched"])
        del value[0][0]["diagnostics"]["query"]
        (directories[1] / "traces.json").write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            behavior_parity(*directories)


if __name__ == "__main__":
    unittest.main()
