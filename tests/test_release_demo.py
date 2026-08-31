import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from demo.release import MESSAGES, build_release, field_evidence, source_receipt, validated_metrics
from mercury.agent import Agent
from mercury.catalog import product_from_dict
from mercury.config import Config
from mercury.model_assets import file_sha256
from mercury.types import Preference


class ReleaseDemoTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.catalog = self.root / "catalog.jsonl"
        rows = [
            {"parent_asin": "A", "title": "Black leather shoulder bag with adjustable strap",
             "features": ["leather", "adjustable"], "categories": ["Bags"]},
            {"parent_asin": "<script>alert(1)</script>", "title": "Blue canvas shoulder bag with adjustable strap",
             "features": ["canvas", "adjustable"], "categories": ["Bags"]},
            {"parent_asin": "C", "title": "Catalog identity must not be exported", "categories": ["Bags"]},
        ]
        self.catalog.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.config_path = self.root / "selected.json"
        self.config_path.write_text(json.dumps({"neural_rerank": False, "question_policy": "none",
                                                "slate_size": 3, "artifact_dir": str(self.root / "assets")}))
        self.config = Config.load(self.config_path)

    def evaluation(self):
        return {"schema": "mercury-evaluation-suite-v1", "source_changed_during_run": False,
                "catalog_sha256": file_sha256(self.catalog), "source_hashes": source_receipt(),
                "runs": [{"kind": "config", "config": self.config.to_dict(),
                          "sessions": [{"sample_id": "private-session-must-not-be-exported", "hit": True,
                                        "best_rank": 2, "first_hit_turn": 2, "reciprocal_rank": .5}],
                          "metrics": {"sample_count": 1, "hit_rate_at_10": 1.0, "mrr": .5, "mttc": 2.0,
                                      "efficiency": .9, "technical_score": .83,
                                      "startup_fallbacks": {}, "fallback_turns": 0}}]}

    def write_evaluation(self, value=None):
        path = self.root / "evaluation.json"
        path.write_text(json.dumps(self.evaluation() if value is None else value))
        return path, file_sha256(path)

    def test_actual_api_responses_and_separate_missing_model_run(self):
        with patch("demo.release.Agent", wraps=Agent) as factory:
            report = build_release(self.root / "out", self.catalog, self.config_path)
        selected, fallback = report["runs"]
        self.assertEqual(selected["status"], "healthy")
        self.assertEqual(fallback["status"], "intentional_fallback")
        missing_path = Path(factory.call_args_list[1].args[1].artifact_dir)
        self.assertNotEqual(missing_path, Path(self.config.artifact_dir))
        self.assertFalse(missing_path.exists())
        agent = Agent(self.catalog, self.config)
        self.addCleanup(agent.close)
        agent.reset("direct", {})
        for turn, message in enumerate(MESSAGES, 1):
            actual = agent.respond("direct", message, turn, 10)
            self.assertEqual(selected["turns"][turn - 1]["response"]["recommendations"], actual["recommendations"])
            self.assertEqual(selected["turns"][turn - 1]["response"]["message"], actual["message"])
        self.assertTrue(selected["turns"][1]["retired_preferences"])
        for turn in fallback["turns"]:
            neural = turn["runtime"]["components"]["neural_rerank"]
            self.assertIs(neural["requested"], True)
            self.assertIs(neural["loaded"], False)
            self.assertIs(neural["effective"], False)
            ids = [row["parent_asin"] for row in turn["response"]["recommendations"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertLessEqual(len(ids), 10)
        self.assertIsNone(report["evaluation"])
        self.assertNotIn(str(self.root), json.dumps(report))
        self.assertNotIn("Catalog identity must not be exported", json.dumps(report))

    def test_healthy_label_cannot_hide_selected_model_failure(self):
        self.config_path.write_text(json.dumps({**self.config.to_dict(), "neural_rerank": True}))
        report = build_release(self.root / "degraded", self.catalog, self.config_path)
        self.assertEqual(report["runs"][0]["status"], "degraded")
        self.assertEqual(report["runs"][1]["status"], "intentional_fallback")

    def test_matching_evaluation_exports_only_verified_aggregates(self):
        path, digest = self.write_evaluation()
        report = build_release(self.root / "measured", self.catalog, self.config_path,
                               evaluation_report=path, evaluation_sha256=digest)
        self.assertEqual(report["evaluation"]["metrics"]["technical_score"], .83)
        self.assertNotIn("private-session-must-not-be-exported", json.dumps(report))
        self.assertNotIn("sessions", report["evaluation"])

    def test_edited_file_is_rejected_by_expected_checksum(self):
        path, digest = self.write_evaluation()
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "checksum"):
            validated_metrics(path, digest, file_sha256(self.catalog), self.config)

    def test_receipt_includes_nested_runtime_modules(self):
        runtime = self.root / "mercury" / "search"
        runtime.mkdir(parents=True)
        module = runtime / "ranking.py"
        module.write_text("value = 1\n")
        evaluator = self.root / "evaluator" / "local_evaluator.py"
        evaluator.parent.mkdir()
        evaluator.write_text("value = 2\n")
        with patch("demo.release.ROOT", self.root):
            receipt = source_receipt()
        self.assertEqual(receipt["mercury/search/ranking.py"], file_sha256(module))

    def test_identity_and_metric_edits_fail_even_with_fresh_checksum(self):
        base = self.evaluation()
        cases = []
        for key, value in (("catalog_sha256", "bad"), ("source_changed_during_run", True),
                           ("source_changed_during_run", "false")):
            edited = deepcopy(base)
            edited[key] = value
            cases.append(edited)
        for key in ("mercury/state.py", "evaluator/local_evaluator.py", "mercury/new_runtime.py"):
            edited = deepcopy(base)
            edited["source_hashes"][key] = "bad"
            cases.append(edited)
        edited = deepcopy(base)
        edited["runs"][0]["config"]["slate_size"] = 2
        cases.append(edited)
        for key, value in (("technical_score", .99), ("sample_count", True),
                           ("startup_fallbacks", {"neural_rerank": "missing"}), ("fallback_turns", 1)):
            edited = deepcopy(base)
            edited["runs"][0]["metrics"][key] = value
            cases.append(edited)
        for index, edited in enumerate(cases):
            with self.subTest(index=index):
                path, digest = self.write_evaluation(edited)
                with self.assertRaises(ValueError):
                    validated_metrics(path, digest, file_sha256(self.catalog), self.config)

    def test_html_is_escaped_and_recording_is_three_minutes(self):
        output = self.root / "html"
        build_release(output, self.catalog, self.config_path, messages=("A bag. <script>alert(1)</script>",))
        page = (output / "index.html").read_text()
        self.assertNotIn("<script>", page)
        self.assertIn("&lt;script&gt;", page)
        cast = [json.loads(row) for row in (output / "replay.cast").read_text().splitlines()]
        self.assertEqual(cast[0]["duration"], 180)
        self.assertEqual(cast[-1][0], 179)
        self.assertFalse(any("http" in row for row in page.splitlines()))

    def test_create_only_and_illegal_response_rejection(self):
        output = self.root / "existing"
        output.mkdir()
        (output / "sentinel").write_text("keep")
        with self.assertRaises(FileExistsError):
            build_release(output, self.catalog, self.config_path)
        self.assertEqual((output / "sentinel").read_text(), "keep")
        with patch.object(Agent, "respond", return_value={"message": "x", "recommendations": [{"parent_asin": "invalid"}]}):
            with self.assertRaisesRegex(ValueError, "illegal"):
                build_release(self.root / "invalid", self.catalog, self.config_path)
        self.assertFalse((self.root / "invalid").exists())

    def test_field_evidence_requires_observation_and_keeps_conflicts_unknown(self):
        product = product_from_dict({"parent_asin": "A", "title": "Canvas bag", "description": "No canvas."})
        evidence = field_evidence(product, Preference("material", "canvas", 1, "canvas"))
        self.assertEqual(evidence["status"], "unknown")
        self.assertEqual({row["field"] for row in evidence["fields"]}, {"title", "description"})
        self.assertEqual(field_evidence(product, Preference("color", "blue", 1, "blue"))["status"], "unknown")
        faux = product_from_dict({"parent_asin": "B", "title": "Faux leather bag"})
        self.assertEqual(field_evidence(faux, Preference("material", "leather", 1, "leather"))["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
