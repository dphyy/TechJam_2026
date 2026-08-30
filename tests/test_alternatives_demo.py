import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from demo.alternatives import REAL_PROBES, SYNTHETIC_ROWS, render_replay, run_replay, select_witness
from mercury.catalog import Catalog, product_from_dict
from mercury.model_assets import file_sha256
from mercury.ranking import preference_evidence
from mercury.types import Preference


def pair(probe_id="bag", *, kind="real"):
    options = [
        {"attribute": "material", "value": value, "source_turn": 1, "polarity": 1,
         "hard": True, "alternative_group": "material:1"}
        for value in ("cotton", "linen")
    ]
    rows = []
    for mode in ("parse", "grouped"):
        preferences = copy.deepcopy(options)
        if mode == "parse":
            for preference in preferences:
                preference.pop("alternative_group")
        rows.append({"catalog_kind": kind, "mode": mode, "probe_id": probe_id, "turn": 1,
                     "user_message": "A shirt, cotton or linen.",
                     "response": {"recommendations": [{"parent_asin": "A"}]},
                     "response_contract": {"status": "passed"}, "error": None,
                     "diagnostics": {"query": "shirts cotton linen", "retrieved_ids": ["A", "B"],
                                     "ranked_ids": ["B", "A"] if mode == "parse" else ["A", "B"],
                                     "constraint_penalties": {"A": 2.0 if mode == "parse" else 0.0, "B": 0.0},
                                     "preferences": preferences, "fallbacks": []},
                     "evidence": {"A": {"source": {"parent_asin": "A", "title": "Cotton shirt",
                                                     "features": "Linen-free", "details": "Fabric cotton"},
                                         "options": [{**option, "signal": signal}
                                                     for option, signal in zip(options, (0.95, -0.4))]}}})
    return rows


class WitnessTests(unittest.TestCase):
    def test_requires_real_common_candidate_explicit_penalties_and_source_evidence(self):
        witness = select_witness(pair())
        self.assertEqual(witness["parent_asin"], "A")
        self.assertEqual((witness["parse_rank"], witness["grouped_rank"]), (2, 1))
        self.assertEqual(witness["source"]["features"], "Linen-free")
        self.assertEqual([item["signal"] for item in witness["option_evidence"]], [0.95, -0.4])

    def test_missing_nonfinite_boolean_or_negative_penalties_are_not_zero(self):
        for mode_index, value in ((1, None), (1, {}), (1, {"A": float("nan")}),
                                  (1, {"A": float("inf")}), (1, {"A": False}),
                                  (1, {"A": -1}), (0, {"A": 0}), (0, {"A": float("inf")})):
            rows = pair()
            rows[mode_index]["diagnostics"]["constraint_penalties"] = value
            with self.subTest(value=value):
                self.assertIsNone(select_witness(rows))

    def test_rejects_changed_query_retrieval_missing_candidate_or_unhealthy_guard(self):
        mutations = [lambda d: d.update(query="another query"),
                     lambda d: d.update(retrieved_ids=["B", "A"]),
                     lambda d: d.update(ranked_ids=["B"]),
                     lambda d: d.update(fallbacks=["constraints"]),
                     lambda d: d.pop("fallbacks"),
                     lambda d: d["preferences"][0].update(hard=False)]
        for mutate in mutations:
            rows = pair()
            mutate(rows[0]["diagnostics"])
            with self.subTest(mutate=mutate):
                self.assertIsNone(select_witness(rows))

    def test_rejects_missing_source_all_contradicted_or_inactive_options(self):
        for change in ("source", "contradicted", "inactive", "missing"):
            rows = pair()
            entry = rows[1]["evidence"]["A"]
            if change == "source":
                entry.pop("source")
            elif change == "contradicted":
                entry["options"][0]["signal"] = -0.5
            elif change == "inactive":
                rows[1]["diagnostics"]["preferences"][0]["active"] = False
            else:
                entry["options"].pop()
            with self.subTest(change=change):
                self.assertIsNone(select_witness(rows))

    def test_unknown_is_accepted_and_choice_is_stable_by_registered_probe_order(self):
        rows = pair("shirt") + pair("bag")
        rows[-1]["evidence"]["A"]["options"][0]["signal"] = 0.0
        self.assertEqual(select_witness(rows)["probe_id"], "bag")
        self.assertEqual(select_witness(list(reversed(rows)))["probe_id"], "bag")


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.catalog = self.directory / "catalog.jsonl"
        self.catalog.write_text(json.dumps({"parent_asin": "REAL", "title": "Everyday shirt"}) + "\n")
        self.output = self.directory / "output"
        self.instances = []
        self.fail_response = False
        self.unhashable_id = False
        self.fail_close = False
        self.fail_start = False
        self.bad_diagnostics = False
        self.interrupt_response = False
        self.mutate_catalog = False
        owner = self

        class FakeAgent:
            def __init__(self, catalog, config):
                if owner.fail_start:
                    raise RuntimeError("startup failed")
                self.catalog = Catalog(catalog)
                self.config = config
                self.calls = []
                self.closed = False
                self.startup_fallbacks = {}
                self.last_diagnostics = {}
                owner.instances.append(self)

            def reset(self, session, profile):
                self.calls.append(("reset", session, copy.deepcopy(profile)))

            def respond(self, session, message, turn, top_k):
                self.calls.append(("respond", session, message, turn, top_k))
                if owner.interrupt_response:
                    raise KeyboardInterrupt("interrupted response")
                identifiers = list(self.catalog.by_id)
                options = [{"attribute": "material", "value": value, "source_turn": turn,
                            "hard": True, "polarity": 1,
                            **({"alternative_group": "material:1"} if self.config.alternatives_mode == "grouped" else {})}
                           for value in (("cotton", "linen") if turn == 1 else ("cotton",))]
                self.last_diagnostics = {"query": "shirts cotton linen" if turn == 1 else "shirts cotton",
                                         "retrieved_ids": identifiers, "ranked_ids": identifiers,
                                         "constraint_penalties": dict.fromkeys(identifiers, 0.0),
                                         "preferences": options, "fallbacks": []}
                if owner.bad_diagnostics:
                    self.last_diagnostics = ["malformed diagnostics"]
                if owner.mutate_catalog:
                    owner.catalog.write_text(json.dumps({"parent_asin": "CHANGED", "title": "Changed shirt"}) + "\n")
                return {"message": "Recorded actual response", "ask_attribute": "other",
                        "recommendations": [{"parent_asin": [] if owner.unhashable_id else
                                             "INVALID" if owner.fail_response else identifiers[0]}],
                        "usage": {"prompt_tokens": 7, "completion_tokens": 0}}

            def close(self):
                self.closed = True
                if owner.fail_close:
                    raise RuntimeError("close failed")

        self.factory = FakeAgent
        self.source_patch = patch("demo.alternatives.source_hashes", return_value={"demo/alternatives.py": "source-digest"})
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)
        self.model_patch = patch("demo.alternatives.model_file_hashes", return_value={"model.safetensors": "model-digest"})
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)

    def run_demo(self, **kwargs):
        return run_replay(self.catalog, self.output, agent_factory=self.factory, **kwargs)

    def test_calls_all_fixed_probes_and_controls_retains_actual_outputs_and_closes(self):
        report = self.run_demo(selected_mode="parse")
        self.assertEqual(len(self.instances), 6)
        self.assertTrue(all(agent.closed for agent in self.instances))
        self.assertEqual(len(report["records"]), 24)
        self.assertEqual({record["mode"] for record in report["records"]}, {"frozen", "parse", "grouped"})
        for agent in self.instances:
            calls = [call for call in agent.calls if call[0] == "respond"]
            self.assertTrue(all(call[-1] == 10 for call in calls))
            self.assertTrue(all(call[-1] == {} for call in agent.calls if call[0] == "reset"))
            if "REAL" in agent.catalog.by_id:
                self.assertEqual([call[2] for call in calls], [message for _, messages in REAL_PROBES for message in messages])
        self.assertEqual(report["records"][0]["response"]["message"], "Recorded actual response")
        self.assertEqual(report["records"][0]["response"]["usage"]["prompt_tokens"], 7)
        self.assertIsNone(report["real_witness"])
        self.assertEqual(json.loads((self.output / "responses.json").read_text()), report)

    def test_provenance_covers_source_configs_catalog_models_and_nonvideo_pacing(self):
        self.run_demo()
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["before"], manifest["after"])
        self.assertEqual(manifest["before"]["catalog_sha256"], file_sha256(self.catalog))
        self.assertEqual(set(manifest["configs"]), {"frozen", "parse", "grouped"})
        self.assertTrue(all(len(item["sha256"]) == 64 for item in manifest["configs"].values()))
        self.assertEqual(manifest["before"]["models"], {"model.safetensors": "model-digest"})
        self.assertEqual(manifest["paid_calls"], 0)
        self.assertFalse(manifest["output_is_video"])
        self.assertEqual(len(manifest["agents"]), 6)
        cast = [json.loads(line) for line in (self.output / "replay.cast").read_text().splitlines()]
        self.assertEqual(cast[0]["duration"], 180.0)
        self.assertEqual(cast[-1][0], 179.0)
        transcript = (self.output / "transcript.txt").read_text()
        self.assertIn("invented", transcript.lower())
        self.assertIn("No real-catalog intervention observed", transcript)
        self.assertIn("not a latency measurement", transcript)

    def test_synthetic_truth_states_use_actual_evidence_and_correction_is_retained(self):
        report = self.run_demo()
        first = next(record for record in report["records"] if record["catalog_kind"] == "invented"
                     and record["mode"] == "grouped" and record["turn"] == 1)
        products = {row["parent_asin"]: product_from_dict(row) for row in SYNTHETIC_ROWS}
        states = set()
        for identifier, entry in first["evidence"].items():
            signals = [item["signal"] for item in entry["options"]]
            for option in entry["options"]:
                self.assertEqual(option["signal"], preference_evidence(products[identifier],
                                 Preference(option["attribute"], option["value"], 1, "")))
            states.add("supported" if max(signals) > 0 else "unknown" if max(signals) == 0 else "contradicted")
        self.assertEqual(states, {"supported", "unknown", "contradicted"})
        corrections = [record for record in report["records"] if record["catalog_kind"] == "invented" and record["turn"] == 2]
        self.assertEqual(len(corrections), 3)
        self.assertTrue(all(record["user_message"] == "Actually, no linen." for record in corrections))

    def test_evidence_capture_calls_the_adapter_instead_of_guessing_from_text(self):
        with patch("demo.alternatives.preference_evidence", return_value=0.25) as adapter:
            report = self.run_demo()
        self.assertGreater(adapter.call_count, 0)
        self.assertTrue(all(option["signal"] == 0.25 for record in report["records"]
                            for entry in record["evidence"].values() for option in entry["options"]))

    def test_create_only_output_never_overwrites_prior_artifacts(self):
        self.run_demo()
        before = (self.output / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_demo()
        self.assertEqual((self.output / "manifest.json").read_bytes(), before)
        self.assertEqual(len(self.instances), 6)

    def test_invalid_id_stops_and_preserves_partial_records_and_closes(self):
        self.fail_response = True
        with self.assertRaisesRegex(RuntimeError, "contract"):
            self.run_demo()
        self.assertTrue(self.instances[0].closed)
        report = json.loads((self.output / "responses.json").read_text())
        self.assertEqual(len(report["records"]), 1)
        self.assertEqual(report["records"][0]["response"]["recommendations"][0]["parent_asin"], "INVALID")
        self.assertEqual(json.loads((self.output / "manifest.json").read_text())["status"], "failed")

    def test_startup_and_close_failures_leave_receipts(self):
        for failure in ("fail_start", "fail_close"):
            with self.subTest(failure=failure):
                self.output = self.directory / failure
                setattr(self, failure, True)
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    self.run_demo()
                manifest = json.loads((self.output / "manifest.json").read_text())
                self.assertEqual(manifest["status"], "failed")
                self.assertTrue(manifest["error"])
                setattr(self, failure, False)

    def test_unhashable_identifier_keeps_original_failure_and_narrated_receipt(self):
        self.unhashable_id = True
        with self.assertRaisesRegex(RuntimeError, "Response contract failed"):
            self.run_demo(selected_mode="frozen")
        manifest = json.loads((self.output / "manifest.json").read_text())
        report = json.loads((self.output / "responses.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertIn("Response contract failed", manifest["error"])
        self.assertEqual(report["records"][0]["response"]["recommendations"][0]["parent_asin"], [])
        self.assertIn("source unavailable", (self.output / "transcript.txt").read_text())
        self.assertTrue(self.instances[0].closed)

    def test_render_failure_keeps_completed_responses_and_failed_manifest(self):
        with patch("demo.alternatives.render_replay", side_effect=ValueError("render failed")), \
                self.assertRaisesRegex(ValueError, "render failed"):
            self.run_demo()
        manifest = json.loads((self.output / "manifest.json").read_text())
        report = json.loads((self.output / "responses.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(report["status"], "failed")
        self.assertIn("render failed", manifest["presentation_error"])
        self.assertEqual(len(report["records"]), 24)
        self.assertIsNone(report["real_witness"])
        self.assertIsNone(report["invented_witness"])
        self.assertTrue(all(agent.closed for agent in self.instances))
        self.assertEqual(manifest["output_sha256"]["responses.json"], file_sha256(self.output / "responses.json"))

    def test_render_failure_does_not_replace_the_original_contract_error(self):
        self.fail_response = True
        with patch("demo.alternatives.render_replay", side_effect=ValueError("render failed")), \
                self.assertRaisesRegex(RuntimeError, "Response contract failed"):
            self.run_demo(selected_mode="frozen")
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertIn("Response contract failed", manifest["error"])
        self.assertIn("render failed", manifest["presentation_error"])
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(len(json.loads((self.output / "responses.json").read_text())["records"]), 1)

    def test_source_or_catalog_changes_fail_without_discarding_completed_turns(self):
        with patch("demo.alternatives.source_hashes", side_effect=[{"code": "before"}, {"code": "after"}]):
            with self.assertRaisesRegex(RuntimeError, "changed"):
                self.run_demo()
        self.assertEqual(len(json.loads((self.output / "responses.json").read_text())["records"]), 24)
        self.assertTrue(all(agent.closed for agent in self.instances))

    def test_changed_model_or_config_inventory_fails_the_run(self):
        for inventory in ("models", "configs"):
            self.output = self.directory / inventory
            config_reads = 0

            def digest(path):
                nonlocal config_reads
                if path.parent.name == "configs" and path.name.startswith("cycle2_"):
                    config_reads += 1
                    if config_reads > 3:
                        return "0" * 64
                return file_sha256(path)

            probe = (patch("demo.alternatives.model_file_hashes", side_effect=[{"model": "before"}, {"model": "after"}])
                     if inventory == "models" else patch("demo.alternatives.file_sha256", side_effect=digest))
            with self.subTest(inventory=inventory), probe, self.assertRaisesRegex(RuntimeError, "changed"):
                self.run_demo()
            manifest = json.loads((self.output / "manifest.json").read_text())
            self.assertNotEqual(manifest["before"][inventory], manifest["after"][inventory])
            report = json.loads((self.output / "responses.json").read_text())
            self.assertEqual(len(report["records"]), 24, f"{inventory}: {manifest['error']}")
            self.assertIsNone(report["real_witness"])
            self.assertIsNone(report["invented_witness"])

    def test_changed_catalog_stops_next_agent_and_preserves_recorded_responses(self):
        self.mutate_catalog = True
        with self.assertRaisesRegex(RuntimeError, "changed"):
            self.run_demo()
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertNotEqual(manifest["before"]["catalog_sha256"], manifest["after"]["catalog_sha256"])
        self.assertEqual(manifest["status"], "failed")
        self.assertTrue(json.loads((self.output / "responses.json").read_text())["records"])
        self.assertTrue(all(agent.closed for agent in self.instances))

    def test_malformed_diagnostics_and_interrupt_preserve_failure_receipts(self):
        for failure, exception in (("bad_diagnostics", AttributeError), ("interrupt_response", KeyboardInterrupt)):
            with self.subTest(failure=failure):
                self.output = self.directory / failure
                setattr(self, failure, True)
                with self.assertRaises(exception):
                    self.run_demo(selected_mode="frozen")
                manifest = json.loads((self.output / "manifest.json").read_text())
                self.assertEqual(manifest["status"], "failed")
                record = json.loads((self.output / "responses.json").read_text())["records"][0]
                self.assertIsNotNone(record["error"])
                self.assertIn("latency_seconds", record)
                self.assertTrue(self.instances[-1].closed)
                setattr(self, failure, False)

    def test_renderer_is_pure_and_never_invents_an_improvement(self):
        report = {"records": [], "real_witness": None, "invented_witness": None}
        transcript, events = render_replay(report, "frozen")
        self.assertIn("No real-catalog intervention observed", transcript)
        self.assertIn("No score, conversion or hidden-set gain is claimed", transcript)
        self.assertEqual(events[-1][0], 179.0)

    def test_narration_fits_terminal_width_without_dumping_raw_records(self):
        report = self.run_demo()
        transcript, events = render_replay(report, "grouped")
        self.assertLessEqual(max(map(len, transcript.splitlines())), 120)
        self.assertIn("Top recommendations", transcript)
        self.assertIn("material=cotton", transcript)
        self.assertNotIn('"source_turn":', transcript)
        self.assertEqual(events[-1][0], 179.0)
        self.assertEqual(len(report["records"]), 24)


if __name__ == "__main__":
    unittest.main()
