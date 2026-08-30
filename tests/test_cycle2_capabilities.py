import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.cycle2_capabilities import (
    check_assertion,
    evaluate_fixture,
    paired_capability_comparison,
    run_development,
    validate_fixture,
    validate_response,
)
from mercury.catalog import product_from_dict
from mercury.config import Config


def fixture():
    return {"schema": "cycle2-capability-fixtures-v1", "split": "development", "cases": [
        {"id": "case-a", "group": "unknown", "catalog": [
            {"parent_asin": "A", "title": "Backpack", "categories": ["Bags"]},
            {"parent_asin": "B", "title": "Other backpack", "categories": ["Bags"]}],
         "turns": [{"message": "A waterproof backpack please.", "assertions": [
             {"kind": "rank_before", "preferred": "A", "other": "B"},
             {"kind": "not_hard_excluded", "product_id": "A"},
             {"kind": "unknown_fact", "product_id": "A", "attribute": "feature", "value": "waterproof"}]}]}]}


def response():
    return {"message": "Any other preference?", "ask_attribute": "other",
            "recommendations": [{"parent_asin": "A"}], "usage": {"prompt_tokens": 7, "completion_tokens": 0}}


class CapabilityTests(unittest.TestCase):
    def setUp(self):
        self.pack = fixture()
        self.products = {row["parent_asin"]: product_from_dict(row) for row in self.pack["cases"][0]["catalog"]}
        self.diagnostics = {"ranked_ids": ["A", "B"], "preferences": [],
                            "constraint_penalties": {"A": 0.0, "B": 0.0}, "fallbacks": []}

    def test_validate_rejects_unknown_schema_duplicate_ids_and_bad_refs(self):
        mutations = [lambda p: p.update(schema="other"),
                     lambda p: p["cases"].append(copy.deepcopy(p["cases"][0])),
                     lambda p: p["cases"][0]["catalog"].append(p["cases"][0]["catalog"][0]),
                     lambda p: p["cases"][0]["turns"][0]["assertions"][0].update(other="missing"),
                     lambda p: p["cases"][0]["turns"][0]["assertions"][0].update(kind="invented")]
        validate_fixture(self.pack, "development")
        for mutate in mutations:
            value = copy.deepcopy(self.pack)
            mutate(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_fixture(value, "development")
        with self.assertRaises(ValueError):
            validate_fixture(self.pack, "validation")

    def test_missing_comparator_is_failure_not_pass(self):
        assertion = self.pack["cases"][0]["turns"][0]["assertions"][0]
        self.diagnostics["ranked_ids"] = ["A"]
        self.assertEqual(check_assertion(assertion, self.diagnostics, self.products)["status"], "failed")
        self.assertEqual(check_assertion(assertion, {}, self.products)["status"], "unverified")

    def test_absent_preference_requires_real_complete_state_adapter(self):
        assertion = {"kind": "absent_positive_preference", "attribute": "color", "value": "blue"}
        for diagnostics in ({}, {"preferences": None}, {"preferences": [{}]}):
            self.assertEqual(check_assertion(assertion, diagnostics, self.products)["status"], "unverified")
        self.assertEqual(check_assertion(assertion, self.diagnostics, self.products)["status"], "passed")
        self.diagnostics["preferences"] = [{"attribute": "color", "value": "blue", "polarity": 1}]
        self.assertEqual(check_assertion(assertion, self.diagnostics, self.products)["status"], "failed")

    def test_active_preference_requires_matching_polarity_and_active_record(self):
        assertion = {"kind": "active_preference", "attribute": "material", "value": "linen", "polarity": -1}
        for polarity, active, expected in ((1, True, "failed"), (-1, False, "failed"), (-1, True, "passed")):
            self.diagnostics["preferences"] = [{"attribute": "material", "value": "linen",
                                                "polarity": polarity, "active": active}]
            self.assertEqual(check_assertion(assertion, self.diagnostics, self.products)["status"], expected)

    def test_neutral_preference_is_valid_and_does_not_hide_other_state_checks(self):
        self.diagnostics["preferences"] = [
            {"attribute": "color", "value": "any", "polarity": 0},
            {"attribute": "material", "value": "cotton", "polarity": 1},
        ]
        assertions = [
            {"kind": "active_preference", "attribute": "color", "value": "any", "polarity": 0},
            {"kind": "active_preference", "attribute": "material", "value": "cotton", "polarity": 1},
            {"kind": "absent_positive_preference", "attribute": "color", "value": "navy"},
        ]
        self.pack["cases"][0]["turns"][0]["assertions"] = assertions
        validate_fixture(self.pack, "development")
        for assertion in assertions:
            with self.subTest(assertion=assertion):
                self.assertEqual(check_assertion(assertion, self.diagnostics, self.products)["status"], "passed")

    def test_every_case_needs_an_assertion_but_intermediate_turns_may_have_none(self):
        self.pack["cases"][0]["turns"].insert(0, {"message": "I need a backpack.", "assertions": []})
        validate_fixture(self.pack, "development")
        for turn in self.pack["cases"][0]["turns"]:
            turn["assertions"] = []
        with self.assertRaisesRegex(ValueError, "assertion"):
            validate_fixture(self.pack, "development")

    def test_no_exclusion_requires_present_candidate_finite_explicit_zero_and_healthy_guard(self):
        assertion = {"kind": "not_hard_excluded", "product_id": "A"}
        for penalties in (None, {}, {"A": float("nan")}, {"A": float("inf")}, {"A": False}, {"A": -1.0}):
            self.diagnostics["constraint_penalties"] = penalties
            self.assertEqual(check_assertion(assertion, self.diagnostics, self.products)["status"], "unverified")
        self.diagnostics["constraint_penalties"] = {"A": 0.0}
        self.assertEqual(check_assertion(assertion, self.diagnostics, self.products)["status"], "passed")
        self.diagnostics["fallbacks"] = ["constraints"]
        self.assertEqual(check_assertion(assertion, self.diagnostics, self.products)["status"], "unverified")
        self.diagnostics["fallbacks"] = []
        self.diagnostics["constraint_penalties"]["A"] = 1.0
        self.assertEqual(check_assertion(assertion, self.diagnostics, self.products)["status"], "failed")
        self.diagnostics["ranked_ids"] = ["B"]
        self.assertEqual(check_assertion(assertion, self.diagnostics, self.products)["status"], "failed")

    def test_unknown_uses_actual_evidence_function_and_missing_adapter_is_unverified(self):
        assertion = self.pack["cases"][0]["turns"][0]["assertions"][2]
        self.assertEqual(check_assertion(assertion, {}, self.products)["status"], "passed")
        with patch("experiments.cycle2_capabilities.preference_evidence", return_value=0.65) as adapter:
            self.assertEqual(check_assertion(assertion, {}, self.products)["status"], "failed")
            self.assertEqual(adapter.call_args.args[1].polarity, 1)
        with patch("experiments.cycle2_capabilities.preference_evidence", side_effect=ValueError("missing")):
            self.assertEqual(check_assertion(assertion, {}, self.products)["status"], "unverified")

    def test_response_contract_rejects_invalid_ids_duplicates_ask_usage_and_size(self):
        self.assertEqual(validate_response(response(), {"A", "B"})["status"], "passed")
        changes = [{"recommendations": [{"parent_asin": "X"}]},
                   {"recommendations": [{"parent_asin": "A"}] * 2},
                   {"recommendations": [{"parent_asin": str(i)} for i in range(11)]},
                   {"ask_attribute": "secret"}, {"message": None},
                   {"usage": {"prompt_tokens": True, "completion_tokens": 0}}, {"usage": None}]
        for change in changes:
            with self.subTest(change=change):
                self.assertEqual(validate_response({**response(), **change}, {"A", "B"})["status"], "failed")

    def test_evaluation_sends_only_catalog_messages_and_empty_profile_and_keeps_all_checks(self):
        calls = []
        diagnostics = self.diagnostics

        class FakeAgent:
            startup_fallbacks = {}
            reranker = object()

            def __init__(self, catalog_path, config):
                calls.append(("catalog", [json.loads(line) for line in catalog_path.read_text().splitlines()], config))

            def reset(self, session_id, profile):
                calls.append(("reset", profile))

            def respond(self, session_id, message, turn, top_k):
                calls.append(("respond", message, turn, top_k))
                self.last_diagnostics = diagnostics
                return response()

            def close(self):
                calls.append(("close",))

        config = Config(neural_rerank=True)
        result = evaluate_fixture(self.pack, config, agent_factory=FakeAgent)
        self.assertEqual(calls[0], ("catalog", self.pack["cases"][0]["catalog"], config))
        self.assertIn(("reset", {}), calls)
        self.assertIn(("respond", "A waterproof backpack please.", 1, 10), calls)
        self.assertEqual(result["assertion_counts"], {"passed": 3, "failed": 0, "unverified": 0})
        self.assertTrue(result["all_passed"])
        self.assertEqual(result["usage"]["prompt_tokens"], 7)

    def test_fallback_or_invalid_response_cannot_be_an_all_pass_run(self):
        for fallback, invalid in ((True, False), (False, True)):
            diagnostics = {**self.diagnostics, "fallbacks": ["neural_rerank"] if fallback else []}

            class FakeAgent:
                startup_fallbacks = {}

                def __init__(self, *_):
                    self.last_diagnostics = diagnostics

                def reset(self, *_):
                    pass

                def respond(self, *_):
                    return {**response(), "ask_attribute": "invalid"} if invalid else response()

            result = evaluate_fixture(self.pack, Config(), agent_factory=FakeAgent)
            self.assertFalse(result["all_passed"])
            if invalid:
                self.assertEqual(result["assertion_counts"]["passed"], 0)

    def test_silent_missing_declared_model_and_inference_errors_are_not_success(self):
        for error, missing_model in ((True, False), (False, True)):
            diagnostics = self.diagnostics

            class FakeAgent:
                startup_fallbacks = {}
                reranker = None

                def __init__(self, *_):
                    self.last_diagnostics = diagnostics

                def reset(self, *_):
                    pass

                def respond(self, *_):
                    if error:
                        raise RuntimeError("request failed")
                    return response()

            result = evaluate_fixture(self.pack, Config(neural_rerank=missing_model), agent_factory=FakeAgent)
            self.assertFalse(result["all_passed"])
            if error:
                self.assertEqual(result["assertion_counts"], {"passed": 0, "failed": 0, "unverified": 3})
                self.assertEqual(result["api_error_turns"], 1)
            else:
                self.assertEqual(result["fallback_turns"], 1)

    def test_development_output_is_create_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            from experiments.cycle2_capabilities import run_capabilities
            with self.assertRaises(FileExistsError):
                run_capabilities(output / "not-opened.json", output / "not-opened-config.json", output,
                                 "development", provenance={})

    def test_startup_failure_preserves_every_turn_and_assertion(self):
        self.pack["cases"][0]["turns"].append(copy.deepcopy(self.pack["cases"][0]["turns"][0]))

        def broken_factory(*args):
            raise RuntimeError("startup unavailable")

        result = evaluate_fixture(self.pack, Config(), agent_factory=broken_factory)
        self.assertEqual(result["turn_count"], 2)
        self.assertEqual(result["api_error_turns"], 2)
        self.assertEqual(result["assertion_counts"], {"passed": 0, "failed": 0, "unverified": 6})
        self.assertFalse(result["all_passed"])

    def test_close_failure_cannot_turn_successful_assertions_into_a_passing_run(self):
        diagnostics = self.diagnostics

        class FakeAgent:
            startup_fallbacks = {}

            def __init__(self, *_):
                self.last_diagnostics = diagnostics

            def reset(self, *_):
                pass

            def respond(self, *_):
                return response()

            def close(self):
                raise RuntimeError("close failed")

        result = evaluate_fixture(self.pack, Config(), agent_factory=FakeAgent)
        self.assertEqual(result["assertion_counts"], {"passed": 3, "failed": 0, "unverified": 0})
        self.assertIn("close failed", result["cases"][0]["close_error"])
        self.assertFalse(result["all_passed"])

    def test_pairing_uses_case_ids_and_exact_assertion_identity(self):
        def report(identifier, status):
            return {"id": identifier, "group": "g", "turns": [{"turn": 1, "assertions": [
                {"assertion": {"kind": "active_preference", "attribute": "color", "value": "blue", "polarity": 1},
                 "status": status}]}]}

        left = {"dataset_sha256": "same", "cases": [report("a", "passed"), report("b", "failed")]}
        right = {"dataset_sha256": "same", "cases": [report("b", "passed"), report("a", "unverified")]}
        compared = paired_capability_comparison(left, right)
        self.assertEqual(len(compared["improved"]), 1)
        self.assertEqual(len(compared["regressed"]), 1)
        right["cases"][0]["turns"][0]["assertions"][0]["assertion"]["value"] = "red"
        with self.assertRaises(ValueError):
            paired_capability_comparison(left, right)
        with self.assertRaises(ValueError):
            paired_capability_comparison({"cases": left["cases"]}, {"cases": left["cases"]})

    def test_development_does_not_accept_validation_disguised_by_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "artifacts/cycle2"
            pack.mkdir(parents=True)
            content = {**fixture(), "split": "validation"}
            path = pack / "capability-development.json"
            path.write_text(json.dumps(content))
            from mercury.model_assets import file_sha256
            (pack / "capability-manifest.json").write_text(json.dumps({
                "schema": "cycle2-capability-lock-v1", "sha256": {path.name: file_sha256(path)}}))
            config_path = root / "config.json"
            config_path.write_text("{}")
            with patch("experiments.cycle2_capabilities.REPOSITORY", root), \
                    self.assertRaisesRegex(ValueError, "split"):
                run_development(config_path, root / "output")


if __name__ == "__main__":
    unittest.main()
