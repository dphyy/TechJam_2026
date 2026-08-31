from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments.lexical_validate import (
    ROOT, SCHEMA, STAGES, _digest, authored_pack, check_turn, evaluate_pack, write_report,
)
from mercury.lexical.agent import Agent
from mercury.lexical.config import FULL_WIDTH_CONFIG


def one_case(identifier="paraphrase"):
    pack = authored_pack()
    pack["cases"] = [case for case in pack["cases"] if case["id"] == identifier]
    return pack


class Adapter:
    def __init__(self, path, **kwargs):
        self.inner = Agent(path, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def respond(self, *args):
        return self.inner.respond(*args)


class NoDiagnostics(Adapter):
    @property
    def last_diagnostics(self):
        return None


class BrokenSearch(Adapter):
    def respond(self, *args):
        raise RuntimeError("injected search failure")


class InvalidScore(Adapter):
    def respond(self, *args):
        response = self.inner.respond(*args)
        response["recommendations"][0]["score"] = float("nan")
        return response


class LexicalValidationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.case = one_case()["cases"][0]
        self.catalog = Path(self.temp.name) / "catalog.jsonl"
        self.catalog.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in self.case["catalog"]))
        self.agent = Agent(self.catalog, config=FULL_WIDTH_CONFIG)
        self.addCleanup(self.agent.close)
        self.agent.reset("S", {})
        self.message = "I'm looking for shirts. A key requirement is: blue; cotton."
        self.response = self.agent.respond("S", self.message, 1, 10)
        self.diagnostics = self.agent.last_diagnostics

    def checks(self, *, response=None, diagnostics=None, expect=None, mode="fullwidth"):
        result = check_turn(self.response if response is None else response,
                            self.diagnostics if diagnostics is None else diagnostics,
                            catalog=self.case["catalog"], catalog_digest=hashlib.sha256(self.catalog.read_bytes()).hexdigest(),
                            mode=mode, turn=1, sources={1: self.message}, expect=expect)
        return {row["check"]: row for row in result}

    def artifact_directory(self):
        (ROOT / "artifacts").mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(dir=ROOT / "artifacts", prefix="authored-test-")

    def test_pack_is_fresh_and_has_twelve_distinct_families(self):
        pack = authored_pack()
        self.assertEqual(pack, authored_pack())
        self.assertEqual(len(pack["cases"]), 12)
        self.assertEqual(len({case["id"] for case in pack["cases"]}), 12)
        pack["cases"][0]["catalog"][0]["title"] = "changed"
        self.assertNotEqual(pack, authored_pack())

    def test_observed_fullwidth_receipts_pass_and_allow_both_equal_products(self):
        for chosen in ("A", "C"):
            response, diagnostics = deepcopy(self.response), deepcopy(self.diagnostics)
            response["recommendations"].sort(key=lambda row: row["parent_asin"] != chosen)
            ids = [row["parent_asin"] for row in response["recommendations"]]
            for stage in ("ranked_prefix", "returned"):
                diagnostics["stage_ids"][stage] = ids
                diagnostics["stage_receipts"][stage].update(ids=ids, sha256=_digest(ids))
            checks = self.checks(response=response, diagnostics=diagnostics,
                                 expect={"leaders": ["A", "C"], "active": ["blue", "cotton"]})
            self.assertTrue(all(check["passed"] for check in checks.values()), checks)

    def test_missing_diagnostics_never_pass_state_membership_or_privacy(self):
        checks = self.checks(diagnostics={}, expect={"active": ["blue"]})
        for name in ("runtime_binding", "stage:retrieval_union", "active_evidence_available",
                     "profile_fields_private", "catalog_witnesses"):
            self.assertEqual(checks[name]["status"], "unavailable")
        self.assertTrue(checks["legal_response"]["passed"])
        report = evaluate_pack(pack=one_case(), agent_factory=NoDiagnostics)
        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(report["case_status_counts"], {"unavailable": 1})
        self.assertTrue(all(row["available"] == 0 for row in report["stage_availability"].values()))

    def test_missing_and_partial_stages_do_not_use_other_id_lists_as_fallback(self):
        for changed in (None, {"available": True, "complete": False, "ids": ["A"]}):
            diagnostics = deepcopy(self.diagnostics)
            diagnostics["stage_receipts"]["retrieval_union"] = changed
            checks = self.checks(diagnostics=diagnostics)
            self.assertEqual(checks["stage:retrieval_union"]["status"], "unavailable")
            self.assertEqual(checks["stage_membership"]["status"], "unavailable")

    def test_forged_stage_digest_and_foreign_membership_are_failures(self):
        diagnostics = deepcopy(self.diagnostics)
        diagnostics["stage_receipts"]["retrieval_union"]["sha256"] = "0" * 64
        self.assertEqual(self.checks(diagnostics=diagnostics)["stage:retrieval_union"]["status"], "fail")
        diagnostics["stage_receipts"]["retrieval_union"].update(ids=["Z"], count=1, sha256=_digest(["Z"]))
        diagnostics["stage_ids"]["retrieval_union"] = ["Z"]
        diagnostics["stage_counts"]["retrieval_union"] = 1
        checks = self.checks(diagnostics=diagnostics)
        self.assertEqual(checks["catalog_membership_preserved"]["status"], "fail")
        self.assertEqual(checks["stage_membership"]["status"], "fail")

    def test_false_evidence_and_fabricated_raw_witnesses_fail(self):
        diagnostics = deepcopy(self.diagnostics)
        witness = diagnostics["constraint_checks"][0]["evidence"][1]["witnesses"][0]
        witness["raw_value"] = "blue silk"
        self.assertEqual(self.checks(diagnostics=diagnostics)["catalog_witnesses"]["status"], "fail")
        checks = self.checks(expect={"absent": ["cotton"], "retired": ["blue"]})
        self.assertTrue(all(row["status"] == "fail" for name, row in checks.items()
                            if name.startswith(("absent:", "retired:"))))

    def test_unshown_evidence_is_unavailable_even_for_a_legitimate_empty_slate(self):
        response, diagnostics = deepcopy(self.response), deepcopy(self.diagnostics)
        response["recommendations"] = []
        diagnostics["constraint_checks"] = []
        checks = self.checks(response=response, diagnostics=diagnostics,
                             expect={"statuses": [{"id": "A", "phrase": "cotton", "status": "supported"}]})
        constraint = checks["constraint:A:cotton"]
        self.assertEqual(constraint["status"], "unavailable")
        self.assertEqual(constraint["observed"]["coverage"], "not_displayed")
        self.assertEqual(checks["catalog_witnesses"]["status"], "unavailable")

    def test_binding_requires_correct_catalog_config_and_live_code(self):
        for field in ("catalog_sha256", "config_sha256", "runtime_source_sha256"):
            diagnostics = deepcopy(self.diagnostics)
            diagnostics["identity"][field] = "0" * 64
            self.assertEqual(self.checks(diagnostics=diagnostics)["runtime_binding"]["status"], "fail")
        diagnostics = deepcopy(self.diagnostics)
        diagnostics["identity"]["runtime_hashes"]["agent.py"] = "0" * 64
        self.assertEqual(self.checks(diagnostics=diagnostics)["runtime_source_files"]["status"], "fail")

    def test_profile_secret_and_missing_source_receipt_are_not_accepted(self):
        diagnostics = deepcopy(self.diagnostics)
        diagnostics["private_debug"] = "private-sentinel"
        self.assertEqual(self.checks(diagnostics=diagnostics)["profile_fields_private"]["status"], "fail")
        diagnostics["evidence_sources"] = []
        checks = self.checks(diagnostics=diagnostics)
        self.assertEqual(checks["message_receipts"]["status"], "fail")
        self.assertEqual(checks["active_evidence_source_coverage"]["status"], "unavailable")

    def test_real_errors_are_recorded_with_type_and_message(self):
        report = evaluate_pack(pack=one_case(), agent_factory=BrokenSearch)
        self.assertEqual(report["status"], "error")
        event = report["cases"][0]["variants"][0]["operations"][0]
        self.assertEqual(event["error"], {"type": "RuntimeError", "message": "injected search failure", "expected": False})
        self.assertGreater(report["check_status_counts"]["error"], 0)

    def test_invalid_scores_remain_reportable_json_and_fail_legality(self):
        report = evaluate_pack(pack=one_case(), agent_factory=InvalidScore)
        payload = json.dumps(report, allow_nan=False)
        self.assertIn('"invalid_number": "nan"', payload)
        self.assertNotEqual(report["status"], "pass")
        event = report["cases"][0]["variants"][0]["operations"][0]
        self.assertTrue(any(row["check"] == "legal_response" and row["status"] == "fail" for row in event["checks"]))

    def test_retry_conflict_reset_and_detached_response_protocol(self):
        report = evaluate_pack(pack=one_case("reset_retry"))
        self.assertEqual(report["status"], "pass")
        errors = [event["error"] for variant in report["cases"][0]["variants"]
                  for event in variant["operations"] if "error" in event]
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0]["expected"])

    def test_runtime_receives_catalog_and_visible_requests_without_expectations(self):
        calls = []

        def factory(path, **kwargs):
            calls.append((json.loads(path.read_text().splitlines()[0]), kwargs))
            return Agent(path, **kwargs)

        pack = one_case()
        before = deepcopy(pack)
        report = evaluate_pack(pack=pack, agent_factory=factory)
        self.assertEqual(pack, before)
        self.assertTrue(calls)
        for row, kwargs in calls:
            self.assertEqual(set(kwargs), {"config", "share_profile_memory"})
            self.assertNotIn("expect", row)
            self.assertNotIn("leaders", row)
        self.assertEqual(report["fixture_sha256"], _digest(pack))

    def test_empty_variants_and_unknown_expectations_are_rejected(self):
        pack = one_case()
        pack["cases"][0]["variants"] = []
        with self.assertRaisesRegex(ValueError, "variants"):
            evaluate_pack(pack=pack)
        pack = one_case()
        pack["cases"][0]["variants"][0]["steps"][0]["expect"] = {"hidden_answer": "A"}
        with self.assertRaisesRegex(ValueError, "expectation"):
            evaluate_pack(pack=pack)

    def test_both_modes_report_all_families_and_stage_availability(self):
        for mode in ("fullwidth", "current-default"):
            report = evaluate_pack(mode)
            self.assertEqual(report["case_count"], 12)
            self.assertEqual(set(report["stage_availability"]), set(STAGES))
            self.assertNotIn("error", report["check_status_counts"])
            for case in report["cases"]:
                for variant in case["variants"]:
                    for check in variant["checks"]:
                        if check["check"] in {"legal_response", "runtime_binding", "runtime_source_files", "stage_membership"}:
                            self.assertTrue(check["passed"], (mode, case["id"], check))

    def test_report_is_create_only_and_cannot_escape_ignored_artifacts(self):
        with self.assertRaisesRegex(ValueError, "ignored artifacts"):
            write_report({"schema": SCHEMA}, Path(self.temp.name) / "report.json")
        with self.artifact_directory() as directory:
            path = Path(directory) / "report.json"
            write_report({"schema": SCHEMA}, path)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_report({"schema": "changed"}, path)
            self.assertEqual(path.read_bytes(), original)

    def test_real_cli_writes_bound_report_and_refuses_overwrite(self):
        with self.artifact_directory() as directory:
            path = Path(directory) / "report.json"
            command = [sys.executable, "-m", "experiments.lexical_validate", "--mode", "fullwidth", "--output", str(path)]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertIn(result.returncode, (0, 1), result.stderr)
            report = json.loads(path.read_text())
            self.assertEqual(report["schema"], SCHEMA)
            self.assertEqual(report["case_count"], 12)
            self.assertEqual(result.returncode, 0 if report["status"] == "pass" else 1)
            self.assertEqual(len(report["fixture_sha256"]), 64)
            original = path.read_bytes()
            repeat = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertNotEqual(repeat.returncode, 0)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
