from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.presentation_evaluate import (
    POLICIES,
    PRESETS,
    ContextObserver,
    PresentationAgent,
    PresentationConfig,
    explicit_slate_rejection,
    make_agent,
)
from mercury.lexical.agent import Agent
from mercury.lexical.config import AgentConfig, FULL_WIDTH_CONFIG, RecommendationPolicy
from mercury.lexical.diagnostics import signature
from mercury.lexical.question_planner import QuestionPlan
from mercury.lexical.retrieval import SearchResult


def candidate(identifier: str, rank: int, *, ambiguous: bool = False, violation: bool = False) -> dict:
    return {"parent_asin": identifier, "title": "Cotton shirt", "categories": ["Shirts"],
            "features": ["cotton"], "_rank_score": float(20 - rank), "_semantic_violation": violation,
            "_hard_constraint_count": 0, "_hard_constraint_exact_count": 0,
            "_category_leaf_match": True, "_catalog_tiebreak": (0.0 if ambiguous else 3.0, 1.0, 1)}


class FixtureSearch:
    def __init__(self, rows: list[dict], catalog_search) -> None:
        self.rows = rows
        self.catalog_search = catalog_search
        self.calls = 0
        self.closed = False

    def __getattr__(self, name: str):
        return getattr(self.catalog_search, name)

    def search_with_context(self, state, limit: int = 10) -> SearchResult:
        self.calls += 1
        return SearchResult([(row["parent_asin"], row["_rank_score"]) for row in self.rows[:limit]], self.rows)

    def close(self) -> None:
        self.closed = True


class FixturePlanner:
    def choose(self, state, candidates, turn) -> QuestionPlan:
        if not candidates:
            return QuestionPlan(None, "No matching records.", 0.0, 0.0, 0.0)
        state.record_question("other")
        return QuestionPlan("other", "What other feature matters most?", 1.0, 1.0, 1.0)


class PresentationPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.catalog = Path(self.directory.name) / "catalog.jsonl"
        self.catalog.write_text("\n".join(json.dumps({
            "parent_asin": str(index), "title": "Cotton shirt", "categories": ["Shirts"],
            "features": ["cotton"], "details": {"Color": "blue"}, "price": 10,
            "average_rating": 4, "rating_number": 10,
        }) for index in range(12)))

    def fixture(self, policy: str = "explicit_rejection", *, ambiguous: bool = False,
                rows: list[dict] | None = None, max_sessions: int = 256) -> PresentationAgent:
        rows = [candidate(str(index), index, ambiguous=ambiguous) for index in range(6)] if rows is None else rows
        config = FULL_WIDTH_CONFIG if policy == "raw10" else AgentConfig(
            recommendation_policy=RecommendationPolicy(adaptive=False))
        inner = Agent(self.catalog, config=config, max_sessions=max_sessions)
        inner.search.close()
        inner.search = FixtureSearch(rows, inner.search)
        inner.question_planner = FixturePlanner()
        result = PresentationAgent(self.catalog, PRESETS[policy], inner=inner)
        self.addCleanup(result.close)
        result.reset("session", {})
        return result

    @staticmethod
    def ids(agent: PresentationAgent, message: str, turn: int, width: int = 2, session: str = "session") -> list[str]:
        return [item["parent_asin"] for item in agent.respond(session, message, turn, width)["recommendations"]]

    def test_only_named_isolated_presets_are_available(self) -> None:
        self.assertEqual(POLICIES, ("existing", "tentative_top1", "explicit_rejection", "raw10"))
        self.assertEqual(set(PRESETS), set(POLICIES))
        with self.assertRaises(ValueError):
            PresentationConfig("combined")

    def test_explicit_rejection_requires_a_negative_statement_about_the_slate(self) -> None:
        for message in (
            "Those options aren't right.", "None of these work for me.", "None of these.",
            "None of these are what I want.", "I do not want any of those recommendations.",
            "Skip those items.", "These options are unsuitable.", "The options are not quite right.",
            "These don't match what I need.",
        ):
            with self.subTest(message=message):
                self.assertTrue(explicit_slate_rejection(message))
        for message in (
            "I'm still browsing.", "Please show more options.", "I need cotton.",
            "I have no preference for color.", "Could I see some alternatives?",
            'You asked "Those options are not right" earlier.',
            "I'm not saying those options aren't right.",
            "If those options aren't right, what comes next?", "None of these are bad.",
            "None of these are wrong.", "I don't want any of these options hidden.",
        ):
            with self.subTest(message=message):
                self.assertFalse(explicit_slate_rejection(message))

    def test_explicit_rejection_moves_only_the_displayed_ids_behind_alternatives(self) -> None:
        agent = self.fixture()
        self.assertEqual(self.ids(agent, "I prefer cotton", 1), ["0", "1"])
        question = agent.last_diagnostics["question_unchanged"]
        self.assertEqual(self.ids(agent, "Those options aren't right.", 2), ["2", "3"])
        diagnostic = agent.last_diagnostics
        self.assertEqual(diagnostic["newly_rejected_ids"], ["0", "1"])
        self.assertEqual(diagnostic["presentation_context_ids"], ["2", "3", "4", "5", "0", "1"])
        self.assertEqual(set(diagnostic["presentation_context_ids"]), set(diagnostic["candidate_context_ids"]))
        self.assertEqual(diagnostic["output_width"]["delta"], 0)
        self.assertTrue(question and diagnostic["question_unchanged"])

    def test_browsing_and_added_constraints_do_not_create_rejections(self) -> None:
        agent = self.fixture()
        self.ids(agent, "I'm looking for Shirts.", 1)
        self.assertEqual(self.ids(agent, "I'm still exploring.", 2), ["0", "1"])
        self.assertEqual(self.ids(agent, "A key requirement is: cotton.", 3), ["0", "1"])
        self.assertEqual(agent.last_diagnostics["rejected_ids"], [])

    def test_rejection_persists_without_rejecting_newly_shown_items_on_browsing(self) -> None:
        agent = self.fixture()
        self.ids(agent, "I prefer red", 1)
        self.ids(agent, "Those options aren't right.", 2)
        self.assertEqual(self.ids(agent, "I'm still browsing.", 3), ["2", "3"])
        self.assertEqual(agent.last_diagnostics["rejected_ids"], ["0", "1"])
        self.assertEqual(agent.last_diagnostics["newly_rejected_ids"], [])
        self.assertEqual(self.ids(agent, "A key requirement is: cotton.", 4), ["2", "3"])
        self.assertIsNone(agent.last_diagnostics["rejection_memory_reset"])

    def test_only_subsequent_explicit_rejection_adds_the_next_displayed_slate(self) -> None:
        agent = self.fixture()
        self.ids(agent, "I prefer red", 1)
        self.ids(agent, "Those options aren't right.", 2)
        self.assertEqual(self.ids(agent, "None of these work for me.", 3), ["4", "5"])
        self.assertEqual(agent.last_diagnostics["newly_rejected_ids"], ["2", "3"])
        self.assertEqual(agent.last_diagnostics["rejected_ids"], ["0", "1", "2", "3"])

    def test_real_color_replacement_clears_old_rejection_context(self) -> None:
        agent = self.fixture()
        self.ids(agent, "I prefer red", 1)
        self.ids(agent, "Those options aren't right.", 2)
        self.assertEqual(self.ids(agent, "Actually, what I need is: blue.", 3), ["0", "1"])
        self.assertEqual(agent.last_diagnostics["rejection_memory_reset"], "active_preference_replaced")
        self.assertEqual(agent.last_diagnostics["forgotten_rejected_ids"], ["0", "1"])
        self.assertEqual(agent.last_diagnostics["rejected_ids"], [])

    def test_noop_correction_does_not_clear_rejection_memory(self) -> None:
        agent = self.fixture()
        self.ids(agent, "I prefer red", 1)
        self.ids(agent, "Those options aren't right.", 2)
        self.assertEqual(self.ids(agent, "Actually, what I need is: red.", 3), ["2", "3"])
        self.assertIsNone(agent.last_diagnostics["rejection_memory_reset"])

    def test_actual_word_without_replaced_constraint_does_not_clear_memory(self) -> None:
        agent = self.fixture()
        self.ids(agent, "I prefer red", 1)
        self.ids(agent, "Those options aren't right.", 2)
        self.assertEqual(self.ids(agent, "Actually, what I need is: cotton.", 3), ["2", "3"])
        self.assertIsNone(agent.last_diagnostics["rejection_memory_reset"])

    def test_category_replacement_clears_memory(self) -> None:
        agent = self.fixture()
        self.ids(agent, "I'm looking for Shirts.", 1)
        self.ids(agent, "Those options aren't right.", 2)
        self.assertEqual(self.ids(agent, "Actually, I'm looking for Shoes instead.", 3), ["0", "1"])
        self.assertEqual(agent.last_diagnostics["rejection_memory_reset"], "category_replaced")

    def test_rejection_with_replacement_does_not_reapply_the_obsolete_display(self) -> None:
        agent = self.fixture()
        self.ids(agent, "I prefer red", 1)
        self.ids(agent, "Those options aren't right.", 2)
        self.ids(agent, "Actually, what I need is: blue. Those options aren't right.", 3)
        self.assertTrue(agent.last_diagnostics["explicit_slate_rejection"])
        self.assertEqual(agent.last_diagnostics["rejected_ids"], [])

    def test_rejection_does_not_promote_a_known_semantic_violation(self) -> None:
        rows = [candidate("0", 0), candidate("1", 1), candidate("2", 2, violation=True)]
        agent = self.fixture(rows=rows)
        self.ids(agent, "I prefer cotton", 1)
        self.assertEqual(self.ids(agent, "Those options aren't right.", 2), ["0", "1"])
        self.assertEqual(agent.last_diagnostics["presentation_context_ids"], ["0", "1", "2"])

    def test_all_rejected_scarce_catalog_keeps_membership_and_order(self) -> None:
        agent = self.fixture(rows=[candidate("0", 0), candidate("1", 1)])
        self.ids(agent, "I prefer cotton", 1)
        self.assertEqual(self.ids(agent, "Those options aren't right.", 2), ["0", "1"])

    def test_tentative_policy_replaces_only_an_actual_empty_ambiguity_slate(self) -> None:
        agent = self.fixture("tentative_top1", ambiguous=True)
        response = agent.respond("session", "I'm looking for Shirts.", 1, 2)
        self.assertEqual([item["parent_asin"] for item in response["recommendations"]], ["0"])
        self.assertEqual(response["message"], "What other feature matters most?")
        self.assertEqual(response["ask_attribute"], "other")
        diagnostic = agent.last_diagnostics
        self.assertEqual(diagnostic["base_returned_ids"], [])
        self.assertEqual({key: diagnostic["output_width"][key] for key in ("before", "after", "delta")},
                         {"before": 0, "after": 1, "delta": 1})
        self.assertTrue(diagnostic["output_width"]["base_ambiguity_deferred"])
        self.assertFalse(diagnostic["output_width"]["ambiguity_deferred"])
        self.assertEqual(diagnostic["output_width"]["returned"], 1)
        self.assertEqual(diagnostic["output_width"]["policy_limit"], 1)
        self.assertEqual(diagnostic["reasons"], ["tentative_top1_after_ambiguity_deferral"])
        self.assertEqual(self.ids(agent, "A key requirement is: cotton.", 2), ["0", "1"])

    def test_tentative_policy_does_not_invent_results_for_empty_catalog_context(self) -> None:
        agent = self.fixture("tentative_top1", rows=[])
        self.assertEqual(self.ids(agent, "I prefer cotton", 1), [])
        self.assertFalse(agent.last_diagnostics["output_width"]["ambiguity_deferred"])

    def test_tentative_policy_keeps_empty_slate_when_leader_is_known_violation(self) -> None:
        rows = [candidate(str(index), index, ambiguous=True, violation=True) for index in range(2)]
        agent = self.fixture("tentative_top1", rows=rows)
        response = agent.respond("session", "I'm looking for shirts. No leather.", 1, 2)
        self.assertEqual(response["recommendations"], [])
        receipt = agent.last_diagnostics
        self.assertEqual(receipt["reasons"], ["tentative_top1_blocked_known_violation"])
        self.assertEqual(receipt["stage_receipts"]["returned"]["count"], 0)
        self.assertTrue(receipt["output_width"]["ambiguity_deferred"])
        self.assertTrue(receipt["question_unchanged"])

    def test_real_tentative_response_retains_base_receipts_and_adds_actual_witnesses(self) -> None:
        path = Path(self.directory.name) / "identical.jsonl"
        path.write_text("".join(json.dumps({"parent_asin": key, "title": "Shirt", "categories": ["shirts"],
                                           "features": ["cotton"]}) + "\n" for key in ("A", "B")))
        agent = make_agent(path, "tentative_top1")
        self.addCleanup(agent.close)
        agent.reset("session", {})
        response = agent.respond("session", "I'm looking for shirts.", 1, 10)
        base, receipt = agent.inner.last_diagnostics, agent.last_diagnostics
        self.assertEqual(len(response["recommendations"]), 1)
        self.assertEqual(base["stage_ids"]["returned"], [])
        for key in ("identity", "evidence", "evidence_sources", "preferences", "retired_preferences", "effective_capabilities"):
            self.assertEqual(receipt[key], base[key])
        for name in ("retrieval_union", "question_context", "ranked_prefix"):
            self.assertEqual(receipt["stage_receipts"][name], base["stage_receipts"][name])
        shown = [item["parent_asin"] for item in response["recommendations"]]
        self.assertEqual(receipt["stage_receipts"]["returned"]["sha256"], signature(shown))
        self.assertEqual(receipt["constraint_checks"][0]["parent_asin"], shown[0])
        self.assertEqual(receipt["constraint_checks"][0]["evidence"][0]["status"], "supported")
        self.assertTrue(receipt["constraint_checks"][0]["evidence"][0]["witnesses"])
        self.assertEqual(len(receipt["presentation_identity"]["implementation_sha256"]), 64)
        self.assertEqual(receipt["presentation_identity"]["config_sha256"], signature({"policy": "tentative_top1"}))
        detached = agent.last_diagnostics
        detached["identity"]["catalog_sha256"] = "changed"
        detached["evidence"]["active"].clear()
        self.assertEqual(agent.last_diagnostics, receipt)
        self.assertEqual(agent.inner.last_diagnostics, base)

    def test_rejection_records_a_presentation_branch_beyond_the_unchanged_raw_prefix(self) -> None:
        agent = self.fixture()
        self.ids(agent, "I prefer cotton", 1)
        self.assertEqual(self.ids(agent, "Those options aren't right.", 2), ["2", "3"])
        receipt = agent.last_diagnostics
        self.assertEqual(receipt["stage_ids"]["ranked_prefix"], ["0", "1"])
        self.assertEqual(receipt["stage_ids"]["presentation_prefix"], ["2", "3"])
        self.assertEqual(receipt["stage_ids"]["returned"], ["2", "3"])
        self.assertEqual(receipt["stage_relationships"]["returned_parent"], "presentation_prefix")
        self.assertFalse(receipt["stage_receipts"]["retrieval_union"]["available"])
        self.assertEqual([row["parent_asin"] for row in receipt["constraint_checks"]], ["2", "3"])

    def test_cached_presentation_retains_its_own_base_receipt_and_reports_no_new_work(self) -> None:
        agent = self.fixture("tentative_top1", ambiguous=True)
        first = agent.respond("session", "I prefer cotton", 1, 2)
        original = agent.last_diagnostics
        agent.reset("other", {})
        agent.respond("other", "I prefer blue", 1, 2)
        with patch.object(agent.inner, "respond", side_effect=AssertionError("cached request executed")):
            self.assertEqual(agent.respond("session", "I prefer cotton", 1, 2), first)
        receipt = agent.last_diagnostics
        for key in ("identity", "evidence", "stage_receipts", "constraint_checks", "presentation_identity"):
            self.assertEqual(receipt[key], original[key])
        self.assertEqual(receipt["current_call"], {"search_executed": False, "inference_executed": False,
                                                  "presentation_executed": False})
        self.assertFalse(receipt["vector_stage"]["attempted"])
        self.assertEqual(receipt["vector_stage"]["origin_receipt"], original["vector_stage"])

    def test_failed_inner_turn_publishes_failure_receipt_without_replacing_successful_cache(self) -> None:
        agent = self.fixture()
        first = agent.respond("session", "I prefer cotton", 1, 2)
        with patch.object(agent.observer.inner, "search_with_context", side_effect=RuntimeError("transient")):
            with self.assertRaises(RuntimeError):
                agent.respond("session", "Those options aren't right.", 2, 2)
        self.assertFalse(agent.last_diagnostics["request_succeeded"])
        self.assertFalse(agent.last_diagnostics["state_committed"])
        self.assertEqual(agent.respond("session", "I prefer cotton", 1, 2), first)
        self.assertTrue(agent.last_diagnostics["request_succeeded"])

    def test_explicit_rejection_does_not_record_a_slate_that_was_never_shown(self) -> None:
        agent = self.fixture(ambiguous=True)
        self.assertEqual(self.ids(agent, "I'm looking for Shirts.", 1), [])
        self.assertEqual(self.ids(agent, "Those options aren't right.", 2), ["0", "1"])
        self.assertEqual(agent.last_diagnostics["rejected_ids"], [])

    def test_existing_policy_is_unchanged_by_rejection_or_ambiguity(self) -> None:
        for ambiguous in (False, True):
            with self.subTest(ambiguous=ambiguous):
                agent = self.fixture("existing", ambiguous=ambiguous)
                self.ids(agent, "I prefer red", 1)
                self.assertEqual(self.ids(agent, "Those options aren't right.", 2), ["0", "1"])
                self.assertEqual(agent.last_diagnostics["rejected_ids"], [])
                self.assertEqual(agent.last_diagnostics["base_returned_ids"], agent.last_diagnostics["returned_ids"])

    def test_raw10_control_is_literal_raw_prefix_and_has_no_rejection_partition(self) -> None:
        rows = [candidate(str(index), index, ambiguous=True) for index in range(12)]
        agent = self.fixture("raw10", rows=rows)
        self.assertEqual(len(self.ids(agent, "I'm looking for Shirts.", 1, 10)), 10)
        result = self.ids(agent, "Those options aren't right.", 2, 10)
        self.assertEqual(result, agent.last_diagnostics["ranked_context_ids"][:10])
        self.assertEqual(agent.last_diagnostics["rejected_ids"], [])

    def test_repeated_request_cannot_reject_the_newly_displayed_slate(self) -> None:
        agent = self.fixture()
        self.ids(agent, "I prefer cotton", 1)
        response = agent.respond("session", "Those options aren't right.", 2, 2)
        original = deepcopy(response)
        calls = agent.observer.calls
        response["recommendations"].clear()
        agent.last_diagnostics["rejected_ids"].append("foreign")
        repeated = agent.respond("session", "Those options aren't right.", 2, 2)
        self.assertEqual(repeated, original)
        self.assertEqual(agent.observer.calls, calls)
        self.assertEqual(agent.last_diagnostics["rejected_ids"], ["0", "1"])
        self.assertTrue(agent.last_diagnostics["cache_hit"])

    def test_retry_restores_its_own_session_diagnostics(self) -> None:
        agent = self.fixture("tentative_top1", ambiguous=True)
        first = agent.respond("session", "I prefer red", 1, 2)
        agent.reset("second", {})
        agent.respond("second", "I prefer blue", 1, 2)
        agent.respond("second", "I need cotton", 2, 2)
        self.assertEqual(agent.respond("session", "I prefer red", 1, 2), first)
        self.assertEqual(agent.last_diagnostics["output_width"]["delta"], 1)
        self.assertEqual(agent.last_diagnostics["reasons"], ["tentative_top1_after_ambiguity_deferral"])

    def test_conflicting_and_stale_requests_do_not_change_rejection_memory(self) -> None:
        agent = self.fixture()
        self.ids(agent, "I prefer cotton", 1)
        self.ids(agent, "Those options aren't right.", 2)
        receipt = agent._receipts["session"]
        for message, turn, width in (("None of these.", 2, 2), ("I prefer cotton", 1, 2),
                                     ("Those options aren't right.", 2, 1)):
            with self.subTest(turn=turn, width=width), self.assertRaises(ValueError):
                agent.respond("session", message, turn, width)
        self.assertIs(agent._receipts["session"], receipt)

    def test_failed_response_does_not_mark_the_previous_slate_rejected(self) -> None:
        agent = self.fixture()
        self.ids(agent, "I prefer cotton", 1)
        receipt = agent._receipts["session"]
        with patch.object(agent.inner, "respond", side_effect=RuntimeError("temporary failure")):
            with self.assertRaises(RuntimeError):
                agent.respond("session", "Those options aren't right.", 2, 2)
        self.assertIs(agent._receipts["session"], receipt)
        self.assertEqual(receipt.rejected, frozenset())
        self.assertEqual(self.ids(agent, "Those options aren't right.", 2), ["2", "3"])
        self.assertEqual(agent.last_diagnostics["newly_rejected_ids"], ["0", "1"])

    def test_old_rejected_ids_are_never_reinserted_into_a_new_candidate_context(self) -> None:
        agent = self.fixture()
        self.ids(agent, "I prefer cotton", 1)
        agent.observer.inner.rows = [candidate(str(index), index) for index in range(2, 6)]
        self.assertEqual(self.ids(agent, "Those options aren't right.", 2), ["2", "3"])
        self.assertEqual(agent.last_diagnostics["presentation_context_ids"], ["2", "3", "4", "5"])
        self.assertEqual(agent.last_diagnostics["rejected_ids"], ["0", "1"])

    def test_full_message_hash_distinguishes_retries_after_parser_limit(self) -> None:
        agent = self.fixture()
        prefix = "x" * 8000
        agent.respond("session", prefix + "one", 1, 2)
        with self.assertRaises(ValueError):
            agent.respond("session", prefix + "two", 1, 2)

    def test_reset_eviction_and_close_clean_experiment_state(self) -> None:
        agent = self.fixture(max_sessions=2)
        self.ids(agent, "I prefer cotton", 1)
        self.ids(agent, "Those options aren't right.", 2)
        agent.reset("session", {})
        self.assertNotIn("session", agent._receipts)
        self.ids(agent, "I prefer cotton", 1)
        agent.reset("second", {})
        self.ids(agent, "I prefer cotton", 1, session="second")
        agent.reset("third", {})
        self.assertNotIn("session", agent._receipts)
        self.assertEqual(set(agent._receipts), {"second"})
        agent.close()
        self.assertFalse(agent._receipts)
        self.assertEqual(agent.observer.context, ())
        with self.assertRaises(RuntimeError):
            agent.reset("new", {})

    def test_observer_rejects_foreign_duplicate_or_nonfinite_context(self) -> None:
        cases = [
            SearchResult([("foreign", 1.0)], [candidate("0", 0)]),
            SearchResult([], [candidate("0", 0), candidate("0", 1)]),
            SearchResult([], [{**candidate("0", 0), "_rank_score": float("nan")}]),
        ]
        for result in cases:
            with self.subTest(result=result):
                observer = ContextObserver(SimpleNamespace(search_with_context=lambda *args, **kwargs: result))
                with self.assertRaises(ValueError):
                    observer.search_with_context(SimpleNamespace())
                self.assertEqual(observer.calls, 0)
                self.assertEqual(observer.context, ())

    def test_all_factories_run_against_a_real_tiny_catalog(self) -> None:
        for policy in POLICIES:
            with self.subTest(policy=policy):
                agent = make_agent(self.catalog, policy)
                try:
                    agent.reset("session", {})
                    response = agent.respond("session", "I'm looking for Shirts. A key requirement is: cotton.", 1, 10)
                    ids = [item["parent_asin"] for item in response["recommendations"]]
                    self.assertLessEqual(set(ids), {str(index) for index in range(12)})
                    self.assertTrue(agent.last_diagnostics["question_unchanged"])
                    if policy == "raw10":
                        self.assertEqual(ids, agent.last_diagnostics["ranked_context_ids"][:10])
                finally:
                    agent.close()

    def test_existing_and_raw_controls_match_independent_base_agents(self) -> None:
        messages = ["I'm looking for Shirts.", "For that, what matters is: blue.",
                    "Actually, what I need is: cotton."]
        for policy in ("existing", "raw10"):
            with self.subTest(policy=policy):
                control = Agent(self.catalog, **({"config": FULL_WIDTH_CONFIG} if policy == "raw10" else {}))
                wrapped = make_agent(self.catalog, policy)
                try:
                    control.reset("session", {})
                    wrapped.reset("session", {})
                    for turn, message in enumerate(messages, 1):
                        self.assertEqual(wrapped.respond("session", message, turn, 10),
                                         control.respond("session", message, turn, 10))
                finally:
                    control.close()
                    wrapped.close()


if __name__ == "__main__":
    unittest.main()
