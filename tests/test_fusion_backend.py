from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mercury.fusion import Agent, Config, make_agent
from mercury.fusion.retrieval import FusionCatalogSearch
from mercury.lexical.agent import Agent as LexicalAgent
from mercury.lexical.dialogue import Evidence, SessionState
from mercury.lexical.product_features import FIELD_ORDER
from mercury.lexical.retrieval import CatalogSearch


def row(identifier: str, **fields) -> dict:
    return {"parent_asin": identifier, "title": "Plain garment", "categories": [], **fields}


def state(category: str = "") -> SessionState:
    evidence = [Evidence("hidden clasp", 3.3, "clarification", 2, "other")]
    if category:
        evidence.insert(0, Evidence(category, 1.4, "category", 1, "category"))
    return SessionState(user_profile={}, evidence=evidence, category_text=category, last_turn=2)


class FusionBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.counter = 0

    def catalog(self, rows: list[dict]) -> Path:
        self.counter += 1
        path = Path(self.directory.name) / f"catalog-{self.counter}.jsonl"
        path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
        return path

    def search(self, path: Path, enabled: bool) -> FusionCatalogSearch:
        search = FusionCatalogSearch(path, additional_admission=enabled, use_prebuilt_index=False)
        self.addCleanup(search.close)
        return search

    @staticmethod
    def ids(response: dict) -> list[str]:
        return [item["parent_asin"] for item in response["recommendations"]]

    def test_off_arm_has_exact_search_score_context_and_agent_parity(self) -> None:
        path = self.catalog([row(f"P{index:02}", title="Cotton shirt", categories=["Shirts"],
                                 features=["hidden clasp"], details={"Color": "blue"})
                             for index in range(16)])
        original = CatalogSearch(path, use_prebuilt_index=False)
        self.addCleanup(original.close)
        off = self.search(path, False)
        original_result = original.search_with_context(state("Shirts"), 10)
        off_result = off.search_with_context(state("Shirts"), 10)
        self.assertEqual(original_result, off_result)
        self.assertEqual(off.diagnostics["additional_routes"], {})
        self.assertEqual(off.diagnostics["stage_ids"]["additional_admitted"], [])
        for fullwidth in (False, True):
            contender = Agent(path, Config(additional_admission=False, fullwidth=fullwidth))
            baseline = LexicalAgent(path, config=contender.config)
            self.addCleanup(contender.close)
            self.addCleanup(baseline.close)
            for agent in (contender, baseline):
                agent.reset("s", {})
            for turn, message in enumerate(("I'm looking for Shirts.",
                                            "For that, what matters is: blue; hidden clasp.",
                                            "Actually, what I need is: red.",
                                            "I don't have a color preference."), 1):
                self.assertEqual(contender.respond("s", message, turn, 10),
                                 baseline.respond("s", message, turn, 10))

    def test_each_weighted_view_rescues_a_record_absent_from_baseline_union(self) -> None:
        cases = {
            "identity": ({"title": "Plain", "categories": ["hidden clasp"]},
                         {"title": "hidden clasp", "categories": ["Plain"]}),
            "structured": ({"title": "hidden clasp", "description": "finish"},
                           {"title": "", "details": {"Finish": "hidden clasp"}}),
            "descriptive": ({"title": "hidden clasp", "description": "plain"},
                            {"title": "plain", "description": "hidden clasp"}),
        }
        for view, (distractor, wanted) in cases.items():
            with self.subTest(view=view):
                path = self.catalog([row(f"P{index:03}", **distractor) for index in range(351)]
                                    + [row("Z", **wanted)])
                off, on = self.search(path, False), self.search(path, True)
                off.search_with_context(state())
                on.search_with_context(state())
                self.assertNotIn("Z", off.diagnostics["stage_ids"]["raw_union"])
                self.assertIn("Z", on.diagnostics["additional_routes"]["view:" + view])
                self.assertIn("Z", on.diagnostics["stage_ids"]["additional_admitted"])

    def test_each_independent_field_rescues_a_record_absent_from_baseline_union(self) -> None:
        for field in FIELD_ORDER:
            with self.subTest(field=field):
                distractor = {"title": "hidden clasp"}
                if field == "title":
                    distractor = {"title": "plain", "categories": ["hidden clasp"]}
                value = "care " * 500 + "hidden clasp"
                wanted = {"title": "plain", field: value}
                if field == "details":
                    wanted[field] = {"Finish": value}
                elif field in {"categories", "features"}:
                    wanted[field] = [value]
                path = self.catalog([row(f"P{index:03}", **distractor) for index in range(351)]
                                    + [row("Z", **wanted)])
                off, on = self.search(path, False), self.search(path, True)
                off.search_with_context(state())
                on.search_with_context(state())
                self.assertNotIn("Z", off.diagnostics["stage_ids"]["raw_union"])
                self.assertIn("Z", on.diagnostics["additional_routes"]["field:" + field])
                self.assertIn("Z", on.diagnostics["stage_ids"]["additional_admitted"])

    def test_category_bucket_rescues_beyond_global_routes_and_stages_remain_distinct(self) -> None:
        path = self.catalog([
            row(f"P{index:03}", title="Plain shirt", categories=["Shirts", "Main"],
                description="hidden clasp") for index in range(351)
        ] + [row("Z", title="Plain shirt", categories=["Shirts", "Specialty"],
                 description="hidden clasp")])
        off, on = self.search(path, False), self.search(path, True)
        off.search_with_context(state("Shirts"))
        on.search_with_context(state("Shirts"))
        self.assertNotIn("Z", off.diagnostics["stage_ids"]["raw_union"])
        nonbucket = [ids for name, ids in on.diagnostics["additional_routes"].items()
                     if not name.startswith("bucket:")]
        self.assertNotIn("Z", set().union(*(set(ids) for ids in nonbucket)))
        self.assertTrue(any("Z" in ids for name, ids in on.diagnostics["additional_routes"].items()
                            if name.startswith("bucket:")))
        stages = on.diagnostics["stage_ids"]
        self.assertIn("Z", stages["raw_union"])
        self.assertNotIn("Z", stages["ranked_top100"])
        self.assertEqual(set(stages["raw_union"]), set(stages["raw_ranked"]))
        self.assertEqual(len(stages["ranked_top100"]), 100)
        self.assertEqual(stages["neural_prefix"], [])
        self.assertTrue(on.diagnostics["candidate_frequency_recomputed_for_union"])

    def test_budget_bounds_only_new_admissions_without_dropping_baseline_members(self) -> None:
        path = self.catalog([row(f"P{index:03}", title="hidden clasp") for index in range(351)]
                            + [row(f"Z{index}", title="plain", description="hidden clasp")
                               for index in range(12)])
        search = self.search(path, True)
        with patch("mercury.fusion.retrieval.MAX_ADDITIONAL_CANDIDATES", 2):
            search.search_with_context(state())
        stages = search.diagnostics["stage_ids"]
        self.assertEqual(len(stages["additional_admitted"]), 2)
        self.assertLessEqual(set(stages["base_union"]), set(stages["raw_union"]))
        with patch("mercury.fusion.retrieval.UNION_BUDGET", 3):
            search.search_with_context(state())
        self.assertEqual(search.diagnostics["stage_ids"]["additional_admitted"], [])
        self.assertTrue(search.diagnostics["budgets"]["inherited_union_overflow"])
        self.assertEqual(search.diagnostics["stage_ids"]["raw_union"],
                         search.diagnostics["stage_ids"]["base_union"])

    def test_one_connection_no_model_loading_and_fullwidth_is_the_raw_top_ten(self) -> None:
        path = self.catalog([row(f"P{index:02}", title="Cotton shirt", categories=["Shirts"])
                             for index in range(16)])
        with patch("mercury.lexical.retrieval.sqlite3.connect", wraps=sqlite3.connect) as connect, \
                patch("mercury.lexical.retrieval.CatalogVectorIndex", side_effect=AssertionError("model load")):
            agent = make_agent(path, additional_admission=True, fullwidth=True)
            self.addCleanup(agent.close)
            self.assertEqual(connect.call_count, 1)
        agent.reset("s", {})
        response = agent.respond("s", "I'm looking for Shirts.", 1, 10)
        stages = agent.diagnostics["stage_ids"]
        self.assertEqual(self.ids(response), stages["raw_ranked"][:10])
        self.assertEqual(self.ids(response), stages["presented"])
        self.assertEqual(len(self.ids(response)), 10)
        self.assertEqual(len(set(self.ids(response))), 10)
        self.assertFalse(agent.diagnostics["runtime"]["neural_requested"])
        self.assertFalse(agent.diagnostics["runtime"]["neural_loaded"])

    def test_exclusions_corrections_and_or_use_shared_state_and_ranking(self) -> None:
        path = self.catalog([
            row("A", title="Blue cotton shirt", categories=["Shirts"], details={"Trim": "leather"}),
            row("B", title="Red cotton shirt", categories=["Shirts"], details={"Trim": "canvas"}),
            row("C", title="Green cotton shirt", categories=["Shirts"], details={"Trim": "canvas"}),
        ])
        agent = make_agent(path, fullwidth=True)
        self.addCleanup(agent.close)
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Shirts.", 1, 10)
        agent.respond("s", "For that, what matters is: red or blue; cotton.", 2, 10)
        response = agent.respond("s", "No leather.", 3, 10)
        self.assertNotEqual(self.ids(response)[0], "A")
        self.assertNotIn('"leather"', " ".join(agent.diagnostics["additional_queries"].values()))
        agent.respond("s", "Actually, what I need is: green.", 4, 10)
        queries = " ".join(agent.diagnostics["additional_queries"].values())
        self.assertNotIn('"red"', queries)
        self.assertNotIn('"blue"', queries)
        self.assertIn('"cotton"', queries)

    def test_retry_restores_matching_stage_trace_and_reset_close_clear_it(self) -> None:
        path = self.catalog([row("A", title="Blue shirt", categories=["Shirts"]),
                             row("B", title="Red shirt", categories=["Shirts"])])
        agent = make_agent(path, fullwidth=True)
        self.addCleanup(agent.close)
        agent.reset("one", {})
        first = agent.respond("one", "I'm looking for Shirts. I prefer blue.", 1, 10)
        first_trace = json.loads(json.dumps(agent.diagnostics))
        agent.diagnostics["stage_ids"]["raw_union"].clear()
        agent.reset("two", {})
        agent.respond("two", "I'm looking for Shirts. I prefer red.", 1, 10)
        self.assertEqual(first, agent.respond("one", "I'm looking for Shirts. I prefer blue.", 1, 10))
        self.assertEqual(first_trace, agent.diagnostics)
        agent.reset("one", {})
        self.assertNotIn("one", agent._traces)
        agent.close()
        self.assertFalse(agent._traces)
        self.assertFalse(agent.diagnostics)
        self.assertFalse(agent.search.diagnostics)

    def test_configuration_has_only_two_boolean_controls(self) -> None:
        self.assertEqual(set(Config.__dataclass_fields__), {"additional_admission", "fullwidth"})
        for arguments in ({"additional_admission": 1}, {"fullwidth": "yes"}):
            with self.assertRaises(ValueError):
                Config(**arguments)


if __name__ == "__main__":
    unittest.main()
