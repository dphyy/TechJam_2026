from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from demo.role_evidence import PROBE, run_replay, verify_report
from mercury.catalog import Catalog


EXPECTED_WITNESS = {
    "WHOLE": [{"preference": "leather outer shell", "material": "leather", "role": "outer shell",
               "source": "description", "span": "leather outer shell", "start": 16, "end": 35}],
}


class RoleEvidenceReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "proof"
        self.fail_response = False
        self.fallback = False
        self.nondeterministic = False
        self.instances = []
        owner = self

        class FakeAgent:
            def __init__(self, catalog, config):
                self.catalog = Catalog(catalog)
                self.config = config
                self.startup_fallbacks = {}
                self.last_diagnostics = {}
                self.closed = False
                owner.instances.append(self)

            def reset(self, session, profile):
                self.session = session

            def respond(self, session, message, turn, top_k):
                identifiers = list(self.catalog.by_id)
                preferences = [
                    {"attribute": "other", "value": "outer shell", "polarity": 1, "source_turn": 1, "hard": False},
                    {"attribute": "other", "value": "leather outer shell", "polarity": 1,
                     "source_turn": 1, "hard": False},
                    {"attribute": "material", "value": "leather", "polarity": 1, "source_turn": 1, "hard": True},
                ]
                if turn == 2:
                    preferences = [preferences[1],
                                   {"attribute": "material", "value": "canvas", "polarity": 1,
                                    "source_turn": 2, "hard": False},
                                   {"attribute": "other", "value": "outer shell", "polarity": 1,
                                    "source_turn": 2, "hard": False}]
                elif turn == 3:
                    preferences = [preferences[1], preferences[0],
                                   {"attribute": "material", "value": "any", "polarity": 0,
                                    "source_turn": 3, "hard": False}]
                witnesses = EXPECTED_WITNESS if self.config.role_evidence and turn == 1 else {}
                ranked = list(reversed(identifiers)) if owner.nondeterministic and session.endswith("enabled-second") \
                    else identifiers
                self.last_diagnostics = {"preferences": preferences, "role_evidence": witnesses,
                                         "ranked_ids": ranked, "retrieved_ids": identifiers,
                                         "fallbacks": ["ranking"] if owner.fallback else []}
                recommendation = "INVALID" if owner.fail_response else identifiers[0]
                return {"message": "Recorded actual response", "ask_attribute": "other",
                        "recommendations": [{"parent_asin": recommendation}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

            def close(self):
                self.closed = True

        self.factory = FakeAgent
        self.source_patch = patch("demo.role_evidence.source_hashes", return_value={"demo/role_evidence.py": "source"})
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)
        self.models_patch = patch("demo.role_evidence.model_file_hashes", return_value={"model": "digest"})
        self.models_patch.start()
        self.addCleanup(self.models_patch.stop)

    def run_demo(self):
        return run_replay(self.output, agent_factory=self.factory)

    def test_records_expected_witness_correction_retraction_and_provenance(self):
        report = self.run_demo()

        self.assertEqual(report["status"], "completed")
        self.assertEqual(len(report["records"]), 9)
        first = report["records"][0]
        self.assertEqual(first["user_message"], PROBE[0])
        self.assertEqual(first["diagnostics"]["role_evidence"], EXPECTED_WITNESS)
        retractions = [record for record in report["records"]
                       if record["label"].startswith("enabled") and record["turn"] in {2, 3}]
        self.assertTrue(all(record["diagnostics"]["role_evidence"] == {} for record in retractions))
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["before"], manifest["after"])
        self.assertEqual(manifest["configurations"]["enabled"]["values"], {
            **manifest["configurations"]["control"]["values"], "role_evidence": True,
        })
        self.assertTrue(all(agent.closed for agent in self.instances))
        transcript = (self.output / "transcript.txt").read_text()
        self.assertIn("not technical-score", transcript)
        self.assertIn("leather outer shell", transcript)
        self.assertTrue((self.output / "invented-catalog.jsonl").is_file())

    def test_refuses_output_overwrite(self):
        self.run_demo()
        before = (self.output / "manifest.json").read_bytes()

        with self.assertRaises(FileExistsError):
            self.run_demo()

        self.assertEqual((self.output / "manifest.json").read_bytes(), before)

    def test_invalid_response_and_fallback_leave_failed_receipts(self):
        for condition in ("fail_response", "fallback"):
            with self.subTest(condition=condition):
                self.output = Path(self.temporary.name) / condition
                setattr(self, condition, True)
                with self.assertRaisesRegex(RuntimeError, "failed|health"):
                    self.run_demo()
                manifest = json.loads((self.output / "manifest.json").read_text())
                report = json.loads((self.output / "responses.json").read_text())
                self.assertEqual(manifest["status"], "failed")
                self.assertEqual(report["status"], "failed")
                self.assertTrue(report["records"])
                self.assertTrue(self.instances[-1].closed)
                setattr(self, condition, False)

    def test_verifier_rejects_component_cross_field_and_nondeterministic_witnesses(self):
        report = self.run_demo()
        mutations = [
            lambda value: value["records"][0]["diagnostics"].update(role_evidence={"COMPONENT": EXPECTED_WITNESS["WHOLE"]}),
            lambda value: value["records"][0]["diagnostics"]["role_evidence"]["WHOLE"][0].update(source="title"),
            lambda value: value["records"][3]["diagnostics"]["ranked_ids"].reverse(),
            lambda value: value["records"][1]["diagnostics"].update(role_evidence=EXPECTED_WITNESS),
        ]

        for mutate in mutations:
            candidate = copy.deepcopy(report)
            mutate(candidate)
            with self.subTest(mutate=mutate), self.assertRaises(RuntimeError):
                verify_report(candidate)

    def test_runner_rejects_nondeterministic_enabled_diagnostics(self):
        self.nondeterministic = True

        with self.assertRaisesRegex(RuntimeError, "not deterministic"):
            self.run_demo()

        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")


if __name__ == "__main__":
    unittest.main()
