from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import patch

from mercury.lexical.agent import Agent
from mercury.lexical.config import FULL_WIDTH_CONFIG
from mercury.lexical.diagnostics import MAX_STAGE_IDS, constraint_receipts, evidence_receipt, stage_receipt
from mercury.lexical.dialogue import Evidence
from mercury.lexical.product_features import ProductFeatureStore
from mercury.lexical.ranking import DEFAULT_RANKING_POLICIES
from mercury.lexical.retrieval import CatalogSearch
from mercury.lexical.vector_index import VectorSearchResult


class EmptyVector:
    enabled = True

    def __init__(self, status=None):
        self.calls = 0
        self.last_call_status = {} if status is None else {"status": status, "inference_attempted": True}

    def search(self, query, limit):
        self.calls += 1
        return VectorSearchResult([])

    def close(self):
        pass


class ReverseSearch(CatalogSearch):
    def __init__(self, *args, reverse=True, **kwargs):
        self.reverse = reverse
        super().__init__(*args, **kwargs)

    def search_with_context(self, *args, **kwargs):
        result = super().search_with_context(*args, **kwargs)
        if self.reverse:
            result.recommendations.reverse()
        return result


class LexicalDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.catalog = Path(self.temp.name) / "catalog.jsonl"
        self.catalog.write_text("\n".join(json.dumps({
            "parent_asin": f"p{index:03}", "title": "cotton shirt", "categories": ["shirts"],
            "features": ["cotton", "blue"], "details": {"lining": "cotton"}, "price": 20,
        }) for index in range(130)) + "\n")

    def agent(self, **kwargs):
        agent = Agent(self.catalog, config=FULL_WIDTH_CONFIG, **kwargs)
        self.addCleanup(agent.close)
        agent.reset("private-session-sentinel", {"profile_id": "private-profile-sentinel",
                                                 "private_value": "private-value-sentinel"})
        return agent

    def respond(self, agent, text="I'm looking for shirts. A key requirement is: cotton.", turn=1):
        return agent.respond("private-session-sentinel", text, turn, 10)

    def test_union_membership_is_not_the_hundred_candidate_context(self):
        agent = self.agent()
        response = self.respond(agent)
        diagnostics = agent.last_diagnostics
        self.assertEqual(diagnostics["stage_counts"], {
            "retrieval_union": 130, "question_context": 100, "ranked_prefix": 10, "returned": 10,
        })
        self.assertEqual(len(diagnostics["stage_ids"]["retrieval_union"]), 130)
        self.assertTrue(set(diagnostics["stage_ids"]["question_context"]) < set(diagnostics["stage_ids"]["retrieval_union"]))
        self.assertEqual(diagnostics["stage_ids"]["returned"], [row["parent_asin"] for row in response["recommendations"]])
        self.assertTrue(diagnostics["output_width"]["full_width"])
        self.assertFalse(diagnostics["output_width"]["ambiguity_deferred"])

    def test_retirement_and_original_raw_chunk_follow_observed_state(self):
        agent = self.agent()
        self.respond(agent, "For that, what matters is: blue cotton.")
        self.respond(agent, "Actually, what I need is: black.", 2)
        diagnostics = agent.last_diagnostics
        active = {row["value"]: row for row in diagnostics["preferences"]}
        self.assertEqual(active["cotton"]["source_turn"], 1)
        self.assertEqual(active["cotton"]["raw_chunk"], "blue cotton")
        self.assertEqual(active["black"]["source_turn"], 2)
        self.assertEqual([row["value"] for row in diagnostics["retired_preferences"]], ["blue cotton"])
        self.assertEqual(diagnostics["retired_preferences"][0]["retired_turn"], 2)
        self.assertEqual(diagnostics["evidence_sources"][0]["text"], "For that, what matters is: blue cotton.")

    def test_append_does_not_fabricate_retirement(self):
        agent = self.agent()
        self.respond(agent, "For that, what matters is: cotton.")
        self.respond(agent, "For that, what matters is: blue.", 2)
        self.assertEqual(agent.last_diagnostics["retired_preferences"], [])

    def test_exact_catalog_witnesses_and_unknowns_are_separate(self):
        agent = self.agent()
        self.respond(agent, "I'm looking for shirts. A key requirement is: cotton; luminous trim.")
        checks = agent.last_diagnostics["constraint_checks"][0]["evidence"]
        by_value = {row["value"]: row for row in checks}
        self.assertEqual(by_value["cotton"]["status"], "supported")
        self.assertTrue(by_value["cotton"]["witnesses"])
        self.assertTrue(all(row["match_kind"] == "normalized_phrase" for row in by_value["cotton"]["witnesses"]))
        self.assertEqual(by_value["luminous trim"]["status"], "unknown")
        self.assertEqual(by_value["luminous trim"]["witnesses"], [])

    def test_scoped_witness_uses_owner_and_numeric_witness_uses_catalog_price(self):
        agent = self.agent()
        self.respond(agent, "I'm looking for shirts. A key requirement is: cotton lining; under $25.")
        checks = {row["value"]: row for row in agent.last_diagnostics["constraint_checks"][0]["evidence"]}
        self.assertEqual(checks["cotton lining"]["status"], "supported")
        self.assertEqual(checks["cotton lining"]["witnesses"][0]["scope"], "lining")
        self.assertEqual(checks["under $25"]["status"], "supported")
        self.assertEqual(checks["under $25"]["witnesses"][0]["catalog_value"], 20)

    def test_detached_diagnostics_and_cached_replay_do_not_execute_search(self):
        agent = self.agent()
        response = self.respond(agent)
        original = agent.last_diagnostics
        changed = agent.last_diagnostics
        changed["stage_ids"]["retrieval_union"].clear()
        changed["preferences"][0]["value"] = "changed"
        self.assertEqual(agent.last_diagnostics, original)
        with patch.object(agent.search, "search_with_context", side_effect=AssertionError("repeat")):
            self.assertEqual(self.respond(agent), response)
        replay = agent.last_diagnostics
        self.assertTrue(replay["cache_hit"])
        self.assertFalse(replay["current_call"]["search_executed"])
        self.assertFalse(replay["current_call"]["inference_executed"])
        self.assertFalse(replay["vector_stage"]["attempted"])
        self.assertEqual(replay["retired_preferences"], [])

    def test_wrapper_receipt_assignment_is_also_detached(self):
        agent = self.agent()
        receipt = {"stage_ids": {"returned": ["p000"]}}
        agent.last_diagnostics = receipt
        receipt["stage_ids"]["returned"].clear()
        self.assertEqual(agent.last_diagnostics["stage_ids"]["returned"], ["p000"])

    def test_identity_has_hashes_and_never_exports_session_or_profile_data(self):
        agent = self.agent()
        self.respond(agent)
        diagnostics = agent.last_diagnostics
        encoded = json.dumps(diagnostics, allow_nan=False)
        for secret in ("private-session-sentinel", "private-profile-sentinel", "private-value-sentinel"):
            self.assertNotIn(secret, encoded)
        for key in ("catalog_sha256", "config_sha256", "runtime_source_sha256"):
            self.assertEqual(len(diagnostics["identity"][key]), 64)
        self.assertEqual(diagnostics["identity"]["catalog_count"], 130)
        self.assertFalse(diagnostics["effective_capabilities"]["components"]["vector_rerank"]["requested"])

    def test_identity_distinguishes_injected_implementation_and_bound_arguments(self):
        agents = [self.agent(), self.agent(search_factory=partial(ReverseSearch, reverse=True)),
                  self.agent(search_factory=partial(ReverseSearch, reverse=False))]
        responses = [self.respond(agent) for agent in agents]
        identities = [agent.last_diagnostics["identity"] for agent in agents]
        self.assertNotEqual(responses[0], responses[1])
        self.assertEqual(responses[0], responses[2])
        self.assertEqual(len({item["runtime_source_sha256"] for item in identities}), 2)
        self.assertEqual(len({item["config_sha256"] for item in identities}), 3)
        self.assertTrue(identities[0]["binding"]["complete"])
        self.assertFalse(identities[1]["binding"]["complete"])
        self.assertIn("custom_component_configuration", identities[1]["binding"]["limitations"])

    def test_injected_search_cannot_claim_a_different_requested_catalog(self):
        other = Path(self.temp.name) / "other.jsonl"
        other.write_text(json.dumps({"parent_asin": "other", "title": "cotton shirt"}) + "\n")
        created = []

        def wrong_catalog(_requested, **kwargs):
            search = CatalogSearch(other, **kwargs)
            created.append(search)
            return search

        with self.assertRaisesRegex(ValueError, "catalog"):
            self.agent(search_factory=wrong_catalog)
        self.assertTrue(created)

    def test_opaque_vector_identity_does_not_claim_known_model_assets(self):
        agent = self.agent(vector_index=EmptyVector())
        self.respond(agent)
        identity = agent.last_diagnostics["identity"]
        self.assertFalse(identity["binding"]["complete"])
        self.assertIn("external_vector_identity", identity["binding"]["limitations"])
        self.assertEqual(identity["vector_assets"], {})

    def test_failed_turn_receipt_does_not_claim_a_fallback_or_state_commit(self):
        agent = self.agent()
        with patch.object(agent.search, "search_with_context", side_effect=RuntimeError("private-error-sentinel")):
            with self.assertRaises(RuntimeError):
                self.respond(agent)
        diagnostics = agent.last_diagnostics
        self.assertFalse(diagnostics["request_succeeded"])
        self.assertFalse(diagnostics["state_committed"])
        self.assertEqual(diagnostics["fallbacks"], [])
        self.assertEqual(diagnostics["error_type"], "RuntimeError")
        self.assertNotIn("private-error-sentinel", json.dumps(diagnostics))
        self.respond(agent)
        self.assertTrue(agent.last_diagnostics["state_committed"])

    def test_empty_vector_result_is_not_reported_as_known_inference_failure(self):
        policies = replace(DEFAULT_RANKING_POLICIES,
                           buying=replace(DEFAULT_RANKING_POLICIES.buying, vector_scale=1.0))
        vector = EmptyVector()
        agent = self.agent(vector_index=vector, ranking_policies=policies)
        self.respond(agent)
        diagnostics = agent.last_diagnostics
        self.assertEqual(vector.calls, 1)
        self.assertEqual(diagnostics["vector_stage"]["status"], "empty_result_unknown")
        self.assertEqual(diagnostics["fallbacks"], [])
        self.assertEqual(diagnostics["effective_capabilities"]["ranking_faults"], [])

    def test_vector_rows_without_a_score_contribution_are_not_effective_reranking(self):
        policies = replace(DEFAULT_RANKING_POLICIES,
                           buying=replace(DEFAULT_RANKING_POLICIES.buying, vector_scale=1.0))
        vector = EmptyVector()
        with patch.object(vector, "search", return_value=VectorSearchResult([(9999, 0.99)])):
            agent = self.agent(vector_index=vector, ranking_policies=policies)
            self.respond(agent)
        diagnostics = agent.last_diagnostics
        self.assertEqual(diagnostics["vector_stage"]["returned_count"], 1)
        self.assertEqual(diagnostics["vector_stage"]["contribution_count"], 0)
        capability = diagnostics["effective_capabilities"]["components"]["vector_rerank"]
        self.assertTrue(capability["loaded"])
        self.assertFalse(capability["effective"])

    def test_known_vector_failure_reports_the_actual_lexical_fallback(self):
        policies = replace(DEFAULT_RANKING_POLICIES,
                           buying=replace(DEFAULT_RANKING_POLICIES.buying, vector_scale=1.0))
        agent = self.agent(vector_index=EmptyVector("inference_failed"), ranking_policies=policies)
        response = self.respond(agent)
        self.assertEqual(len(response["recommendations"]), 10)
        diagnostics = agent.last_diagnostics
        self.assertEqual(diagnostics["fallbacks"], ["vector_rerank"])
        capability = diagnostics["effective_capabilities"]["components"]["vector_rerank"]
        self.assertTrue(capability["requested"])
        self.assertTrue(capability["loaded"])
        self.assertFalse(capability["effective"])

    def test_backend_receipt_cannot_override_execution_counters_or_export_extra_data(self):
        policies = replace(DEFAULT_RANKING_POLICIES,
                           buying=replace(DEFAULT_RANKING_POLICIES.buying, vector_scale=1.0))
        vector = EmptyVector()
        vector.last_call_status = {"status": ["invalid"], "inference_attempted": "yes",
                                  "contribution_count": None, "returned_count": 999,
                                  "attempted": False, "confidence_gate": True,
                                  "error_type": "private error sentinel", "private_value": "do not export"}
        agent = self.agent(vector_index=vector, ranking_policies=policies)
        response = self.respond(agent)
        receipt = agent.last_diagnostics["vector_stage"]
        self.assertEqual(len(response["recommendations"]), 10)
        self.assertTrue(receipt["attempted"])
        self.assertEqual(receipt["returned_count"], 0)
        self.assertEqual(receipt["contribution_count"], 0)
        self.assertFalse(receipt["confidence_gate"])
        self.assertIsNone(receipt["inference_attempted"])
        self.assertEqual(receipt["status"], "empty_result_unknown")
        self.assertNotIn("error_type", receipt)
        self.assertNotIn("private_value", receipt)

    def test_diagnostics_retention_follows_session_eviction_and_close(self):
        agent = self.agent(max_sessions=1)
        self.respond(agent)
        agent.reset("next", {})
        self.assertFalse(agent._diagnostics)
        self.assertFalse(agent._sources)
        self.assertEqual(agent.last_diagnostics, {})
        agent.respond("next", "cotton shirt", 1, 10)
        agent.close()
        self.assertFalse(agent._diagnostics)
        self.assertFalse(agent._sources)
        self.assertEqual(agent.last_diagnostics, {})


class BoundedReceiptTest(unittest.TestCase):
    def test_requested_literal_absence_can_have_a_raw_catalog_witness(self):
        product = {"parent_asin": "absent", "features": "fragrance-free"}
        product["_features"] = ProductFeatureStore().add("absent", {"features": "fragrance-free"})
        checks = constraint_receipts([product], [Evidence("fragrance free", 3.8, "hard_constraint", 1)])
        check = checks[0]["evidence"][0]
        self.assertEqual(check["status"], "supported")
        self.assertEqual(check["witnesses"][0]["raw_value"], "fragrance-free")

    def test_negated_raw_metadata_does_not_witness_a_positive_requirement(self):
        product = {"parent_asin": "absent", "features": "no cotton"}
        product["_features"] = ProductFeatureStore().add("absent", {"features": "no cotton"})
        checks = constraint_receipts([product], [Evidence("cotton", 3.8, "hard_constraint", 1)])
        check = checks[0]["evidence"][0]
        self.assertNotEqual(check["status"], "supported")
        self.assertEqual(check["witnesses"], [])

    def test_missing_stage_hook_reports_unknown_not_an_empty_union(self):
        receipt = stage_receipt(None)
        self.assertFalse(receipt["available"])
        self.assertIsNone(receipt["count"])
        self.assertFalse(receipt["complete"])

    def test_explicit_absence_is_distinct_from_missing_catalog_data(self):
        store = ProductFeatureStore()
        absent = {"parent_asin": "absent", "features": "no leather"}
        unknown = {"parent_asin": "unknown", "features": "machine washable"}
        for product in (absent, unknown):
            product["_features"] = store.add(product["parent_asin"], {"features": product["features"]})
        evidence = [Evidence("leather", 3.8, "exclusion", 1)]
        checks = constraint_receipts([absent, unknown], evidence)
        self.assertEqual(checks[0]["evidence"][0]["status"], "supported")
        self.assertEqual(checks[0]["evidence"][0]["witnesses"][0]["match_kind"], "explicit_absence")
        self.assertEqual(checks[1]["evidence"][0]["status"], "unknown")

    def test_large_stage_keeps_true_count_and_digest_with_explicit_truncation(self):
        values = [f"p{index}" for index in range(MAX_STAGE_IDS + 1)]
        receipt = stage_receipt(values)
        self.assertEqual(receipt["count"], len(values))
        self.assertEqual(len(receipt["ids"]), MAX_STAGE_IDS)
        self.assertFalse(receipt["complete"])
        self.assertNotEqual(receipt["sha256"], stage_receipt(values[:-1])["sha256"])

    def test_ambiguous_lineage_does_not_invent_a_raw_chunk(self):
        first = Evidence("blue cotton", 2.0, "clarification", 1)
        second = Evidence("red cotton", 2.0, "clarification", 1)
        current = Evidence("cotton", 2.0, "clarification", 1)
        receipt = evidence_receipt([first, second], [current], {}, 2)
        self.assertEqual(receipt["active"][0]["raw_chunk"], "cotton")
        self.assertEqual(receipt["retired_count"], 2)


if __name__ == "__main__":
    unittest.main()
