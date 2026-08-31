from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from agent import Agent
from demo.submission import MESSAGES, ROOT, SCHEMA, _sources, build_demo
from experiments.submission_evaluate import EVALUATOR_SHA256, SCHEMA as EVALUATION_SCHEMA, source_receipt
from mercury.lexical.config import DEFAULT_AGENT_CONFIG
from mercury.model_assets import file_sha256


class SubmissionDemoTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog = self.root / "catalog.jsonl"
        rows = [
            {"parent_asin": "A", "title": "raw-title-sentinel", "categories": ["bags"],
             "features": ["black", "leather", "adjustable strap"]},
            {"parent_asin": "B", "title": "raw-title-sentinel", "categories": ["bags"],
             "features": ["blue", "canvas", "adjustable strap"]},
            {"parent_asin": "C", "title": "raw-title-sentinel", "categories": ["bags"],
             "features": ["blue", "canvas", "fixed strap"]},
        ]
        self.catalog.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def evaluation(self):
        metrics = {"sample_count": 1, "hit_rate_at_10": 1.0, "mrr": 1.0, "mttc": 1.0,
                   "efficiency": 1.0, "recommended_technical_score": 1.0}
        report = {"schema": EVALUATION_SCHEMA, "mode": "selected", "valid": True,
                  "source_hashes": source_receipt(), "catalog_sha256": file_sha256(self.catalog),
                  "config": asdict(DEFAULT_AGENT_CONFIG), "evaluator_sha256": EVALUATOR_SHA256,
                  "measurement": {"source_changed": False, "catalog_changed": False, "dataset_changed": False,
                                  "errors": 0, "fallback_turns": 0, "startup_fallbacks": []},
                  "result": {**metrics, "sessions": [{"sample_id": "private-outcome-sentinel", "hit": True,
                      "best_rank": 1, "first_hit_turn": 1, "reciprocal_rank": 1.0}]}}
        path = self.root / "evaluation.json"
        path.write_text(json.dumps(report))
        return path, file_sha256(path), metrics

    def test_actual_public_responses_and_state_are_recorded_without_legacy_attributes(self):
        report = build_demo(self.root / "demo", self.catalog)
        direct = Agent(self.catalog)
        self.addCleanup(direct.close)
        direct.reset("direct", {})
        for number, message in enumerate(MESSAGES, 1):
            response = direct.respond("direct", message, number, 10)
            turn = report["turns"][number - 1]
            self.assertEqual(turn["response"], response)
            raw = direct.last_diagnostics
            self.assertEqual([row["value"] for row in turn["diagnostics"]["evidence"]["active"]],
                             [row["value"] for row in raw["evidence"]["active"]])
            self.assertEqual(turn["diagnostics"]["identity"]["catalog_sha256"], raw["identity"]["catalog_sha256"])
            self.assertGreaterEqual(turn["latency_seconds"], 0)
        self.assertTrue(report["turns"][1]["diagnostics"]["evidence"]["retired"])
        self.assertEqual(report["schema"], SCHEMA)
        self.assertEqual(report["catalog_count"], 3)
        self.assertEqual(set(path.name for path in (self.root / "demo").iterdir()),
                         {"evidence.json", "index.html", "transcript.txt"})

    def test_unrequested_optional_models_are_healthy_and_never_fabricated_as_missing(self):
        report = build_demo(self.root / "demo", self.catalog)
        self.assertEqual(report["status"], "healthy")
        self.assertIsNone(report["evaluation"])
        self.assertEqual(len(report["turns"]), len(MESSAGES))
        for turn in report["turns"]:
            for name in ("vector_rerank", "neural_rerank"):
                component = turn["diagnostics"]["capabilities"][name]
                for flag in ("requested", "loaded", "effective"):
                    self.assertIs(component[flag], False)
            self.assertEqual(turn["diagnostics"]["fallbacks"], [])

    def test_sanitization_omits_raw_catalog_fields_paths_profiles_and_extra_diagnostics(self):
        original = Agent.respond

        def decorated(agent, *args):
            response = original(agent, *args)
            diagnostic = agent.last_diagnostics
            diagnostic["profile"] = {"secret": "private-profile-sentinel"}
            diagnostic["identity"]["local_path"] = str(self.root)
            for row in diagnostic["constraint_checks"]:
                for evidence in row["evidence"]:
                    for witness in evidence["witnesses"]:
                        witness["raw_value"] = "raw-witness-sentinel"
                        witness["unexpected"] = "unexpected-metadata-sentinel"
            agent.last_diagnostics = diagnostic
            return response

        with patch.object(Agent, "respond", decorated):
            report = build_demo(self.root / "demo", self.catalog, messages=MESSAGES[:2])
        payload = json.dumps(report, allow_nan=False)
        for sentinel in ("raw-title-sentinel", "private-profile-sentinel", "raw-witness-sentinel",
                         "unexpected-metadata-sentinel", str(self.root), "recorded-conversation"):
            self.assertNotIn(sentinel, payload)
        self.assertNotIn('"raw_value"', payload)
        self.assertNotIn('"raw_chunk"', payload)
        self.assertIn("agent.py", report["source_hashes"])
        self.assertIn("starter/agent.py", report["source_hashes"])
        self.assertIn("demo/submission.py", report["source_hashes"])

    def test_only_separately_verified_aggregate_metrics_are_exported(self):
        path, digest, metrics = self.evaluation()
        report = build_demo(self.root / "demo", self.catalog, evaluation_report=path, evaluation_sha256=digest)
        self.assertEqual(report["evaluation"]["metrics"], metrics)
        self.assertEqual(report["evaluation"]["report_sha256"], digest)
        self.assertNotIn("private-outcome-sentinel", json.dumps(report))
        self.assertNotIn("sessions", report["evaluation"])
        self.assertNotIn("metrics", report["turns"][0])

    def test_bad_evaluation_checksum_and_partial_metric_arguments_are_rejected(self):
        path, digest, _ = self.evaluation()
        with self.assertRaisesRegex(ValueError, "together"):
            build_demo(self.root / "partial", self.catalog, evaluation_report=path)
        with self.assertRaisesRegex(ValueError, "checksum"):
            build_demo(self.root / "bad", self.catalog, evaluation_report=path, evaluation_sha256="0" * 64)
        self.assertFalse((self.root / "bad").exists())
        edited = json.loads(path.read_text())
        edited["result"]["recommended_technical_score"] = 0.99
        path.write_text(json.dumps(edited))
        with self.assertRaisesRegex(ValueError, "Aggregate"):
            build_demo(self.root / "edited", self.catalog, evaluation_report=path, evaluation_sha256=file_sha256(path))

    def test_missing_diagnostics_and_illegal_response_cannot_create_a_successful_demo(self):
        original = Agent.respond

        def no_diagnostics(agent, *args):
            response = original(agent, *args)
            agent.last_diagnostics = {}
            return response

        with patch.object(Agent, "respond", no_diagnostics), self.assertRaisesRegex(ValueError, "transaction receipt"):
            build_demo(self.root / "missing", self.catalog)
        response = {"message": "Invalid", "ask_attribute": None,
                    "recommendations": [{"parent_asin": "Z", "score": 1}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
        with patch.object(Agent, "respond", return_value=response), self.assertRaisesRegex(ValueError, "illegal"):
            build_demo(self.root / "illegal", self.catalog)
        self.assertFalse((self.root / "missing").exists())
        self.assertFalse((self.root / "illegal").exists())

    def test_false_stage_receipts_and_absent_capability_flags_are_rejected(self):
        original = Agent.respond
        for kind in ("stage", "capability"):
            def altered(agent, *args):
                response = original(agent, *args)
                diagnostic = agent.last_diagnostics
                if kind == "stage":
                    diagnostic["stage_receipts"]["returned"]["sha256"] = "0" * 64
                else:
                    del diagnostic["effective_capabilities"]["components"]["neural_rerank"]["requested"]
                agent.last_diagnostics = diagnostic
                return response
            with self.subTest(kind=kind), patch.object(Agent, "respond", altered), self.assertRaises(ValueError):
                build_demo(self.root / kind, self.catalog)

    def test_runtime_changes_are_rejected_before_any_output_is_published(self):
        original = _sources()
        changed = {**original, "agent.py": "changed"}
        with patch("demo.submission._sources", side_effect=(original, changed)):
            with self.assertRaisesRegex(ValueError, "changed"):
                build_demo(self.root / "changed", self.catalog)
        self.assertFalse((self.root / "changed").exists())

    def test_error_closes_actual_agent_and_does_not_publish_partial_artifact(self):
        closed = []
        original = Agent.close

        def close(agent):
            closed.append(True)
            return original(agent)

        with patch.object(Agent, "close", close), patch.object(Agent, "respond", side_effect=RuntimeError("transient")):
            with self.assertRaisesRegex(RuntimeError, "transient"):
                build_demo(self.root / "failed", self.catalog)
        self.assertEqual(closed, [True])
        self.assertFalse((self.root / "failed").exists())

    def test_portable_html_escapes_text_and_records_no_invented_pacing(self):
        report = build_demo(self.root / "html", self.catalog,
                            messages=("I'm looking for bags. <script>alert(1)</script>",))
        page = (self.root / "html/index.html").read_text()
        self.assertNotIn("<script>", page)
        self.assertIn("&lt;script&gt;", page)
        self.assertNotIn("http", page)
        self.assertNotIn("presentation_seconds", report)
        self.assertFalse((self.root / "html/replay.cast").exists())
        self.assertGreaterEqual(report["cold_start_seconds"], 0)

    def test_output_is_create_only_and_messages_are_bounded(self):
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "keep"
        sentinel.write_text("unchanged")
        with self.assertRaises(FileExistsError):
            build_demo(output, self.catalog)
        self.assertEqual(sentinel.read_text(), "unchanged")
        for messages in ((), ("",), ("x" * 8001,), ("bags",) * 11):
            with self.subTest(count=len(messages)), self.assertRaises(ValueError):
                build_demo(self.root / "invalid", self.catalog, messages=messages)

    def test_real_cli_records_tiny_catalog_without_optional_evaluation(self):
        output = self.root / "cli"
        command = [sys.executable, "-m", "demo.submission", "--catalog", str(self.catalog), "--output", str(output)]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((output / "evidence.json").read_text())
        self.assertEqual(report["schema"], SCHEMA)
        self.assertEqual(len(report["turns"]), 3)
        self.assertEqual(json.loads(result.stdout)["verified_evaluation_attached"], False)
        before = (output / "evidence.json").read_bytes()
        repeat = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertNotEqual(repeat.returncode, 0)
        self.assertEqual((output / "evidence.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
