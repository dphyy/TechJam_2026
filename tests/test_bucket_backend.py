from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mercury.bucket import Agent, AgentConfig
from mercury.bucket.index import CatalogIndex


def product(identifier: str, *, title: str = "Shirt", material: str = "cotton",
            color: str = "blue", price: object = 20, count: object = 1,
            categories: list[str] | None = None, **extra: object) -> dict:
    return {
        "parent_asin": identifier, "title": title,
        "categories": ["Clothing", "Tops", "Shirts"] if categories is None else categories,
        "details": {"Material": material, "Color": color}, "features": [],
        "price": price, "rating_number": count, "average_rating": 4,
        **extra,
    }


class BucketAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.sequence = 0

    def agent(self, rows: list[dict], **settings: object) -> Agent:
        self.sequence += 1
        path = Path(self.directory.name) / f"catalog-{self.sequence}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        agent = Agent(path, config=AgentConfig(**settings))
        self.addCleanup(agent.close)
        agent.reset("session", {})
        return agent

    def ids(self, agent: Agent, message: str, turn: int = 1, top_k: int = 10) -> list[str]:
        return [item["parent_asin"] for item in agent.respond("session", message, turn, top_k)["recommendations"]]

    def test_union_preserves_catalog_membership_and_complete_taxonomy(self) -> None:
        rows = [product("wrong", title="Blue shirt", count=1000, categories=["Electronics", "Cases"]),
                product("right", title="Plain item", categories=["Clothing", "Tops", "Shirts"])]
        agent = self.agent(rows)
        result = self.ids(agent, "I'm looking for Clothing Tops Shirts. I prefer blue.")
        self.assertEqual(result[0], "right")
        self.assertEqual(set(result), {row["parent_asin"] for row in rows})
        self.assertEqual(agent.diagnostics("session")["category_mode"], "exact")

    def test_atomic_full_raw_metadata_beats_partial_title(self) -> None:
        detail = "precise " + "padding " * 40 + "concealed vent"
        agent = self.agent([
            product("partial", title="Precise padding shirt with vent", count=100000),
            product("complete", uncommon_field={"construction": [detail]}),
        ])
        self.assertEqual(self.ids(agent, "A key requirement is: " + detail)[0], "complete")
        self.assertEqual(agent.product("complete")["uncommon_field"]["construction"], [detail])
        self.assertEqual(agent.diagnostics("session")["top_evidence"][0]["exact"], 1)

    def test_bounded_candidate_union_still_rescues_atomic_match(self) -> None:
        rows = [product(f"decoy-{index:02d}", count=1000, features=["soft fabric"]) for index in range(30)]
        rows.append(product("wanted", features=["independently adjustable cuffs"]))
        agent = self.agent(rows, candidate_limit=10, lexical_limit=10)
        ids = self.ids(agent, "I'm looking for Shirts. A key requirement is: independently adjustable cuffs.")
        self.assertEqual(ids[0], "wanted")
        self.assertEqual(len(ids), 10)
        self.assertIn("wanted", agent.diagnostics("session")["candidate_ids"])

    def test_unknown_category_does_not_invent_membership(self) -> None:
        agent = self.agent([product("a"), product("b", features=["sealed zipper"])])
        self.assertEqual(self.ids(agent, "I'm looking for Uncatalogued objects. A key requirement is: sealed zipper.")[0], "b")
        self.assertEqual(agent.diagnostics("session")["category_mode"], "unresolved")

    def test_exclusion_overrides_many_positive_exact_matches(self) -> None:
        agent = self.agent([
            product("excluded", material="leather", features=["sealed zipper", "wide cuffs"], count=100000),
            product("safe", material="cotton", features=["sealed zipper"]),
        ])
        self.ids(agent, "I'm looking for Shirts. A key requirement is: sealed zipper; wide cuffs.")
        self.assertEqual(self.ids(agent, "I want no leather.", 2)[0], "safe")
        self.assertEqual(agent.diagnostics("session")["top_evidence"][0]["violations"], 0)

    def test_exclusion_only_query_searches_beyond_presentation_width(self) -> None:
        rows = [product(f"excluded-{index}", material="leather", count=100) for index in range(15)]
        rows.append(product("safe", material="cotton"))
        agent = self.agent(rows)
        self.assertEqual(self.ids(agent, "I want no leather.")[0], "safe")
        self.assertEqual(agent.diagnostics("session")["candidate_count"], 16)

    def test_number_and_material_cannot_be_combined_across_unrelated_fields(self) -> None:
        agent = self.agent([
            product("exact", material="100% cotton", price=20),
            product("split", material="60% cotton", price=100, count=1000),
        ])
        self.assertEqual(self.ids(agent, "A key requirement is: 100% cotton.")[0], "exact")
        evidence = agent.diagnostics("session")["top_evidence"]
        self.assertEqual(evidence[0]["exact"], 1)
        self.assertEqual(evidence[1]["complete"], 0)

    def test_negated_catalog_property_is_not_positive_support(self) -> None:
        agent = self.agent([
            product("denied", material="no cotton", count=10000),
            product("affirmed", material="cotton"),
        ])
        self.assertEqual(self.ids(agent, "A key requirement is: cotton.")[0], "affirmed")
        denied = next(item for item in agent.diagnostics("session")["top_evidence"] if item["parent_asin"] == "denied")
        self.assertEqual(denied["complete"], 0)

    def test_free_shipping_does_not_negate_material(self) -> None:
        agent = self.agent([
            product("wanted", material="cotton with free shipping"),
            product("other", material="polyester", count=100),
        ])
        self.assertEqual(self.ids(agent, "I need cotton.")[0], "wanted")

    def test_or_is_one_requirement_with_alternative_branches(self) -> None:
        agent = self.agent([
            product("cotton", material="cotton", count=3),
            product("linen", material="linen", count=2),
            product("neither", material="polyester", count=1000),
        ])
        self.assertEqual(set(self.ids(agent, "A key requirement is: cotton or linen.")[:2]), {"cotton", "linen"})
        diagnostics = agent.diagnostics("session")
        self.assertEqual(diagnostics["requirement_count"], 1)
        self.assertTrue(all(item["complete"] == 1 for item in diagnostics["top_evidence"][:2]))

    def test_correction_removes_earlier_color_but_keeps_other_constraint(self) -> None:
        agent = self.agent([
            product("red", color="red", count=1000), product("blue", color="blue"),
            product("wrong-material", color="blue", material="polyester", count=2000),
        ])
        self.ids(agent, "I'm looking for Shirts. A key requirement is: red; cotton.")
        self.assertEqual(self.ids(agent, "Actually, what I need is: blue.", 2)[0], "blue")
        active = [item.text.casefold() for item in agent._sessions["session"].evidence]
        self.assertNotIn("red", active)
        self.assertIn("cotton", active)

    def test_component_constraint_cannot_borrow_other_component_material(self) -> None:
        agent = self.agent([
            product("wrong", material="", details={"Shell": "polyester", "Lining": "cotton"}, count=1000),
            product("right", material="", details={"Shell": "cotton", "Lining": "polyester"}),
        ])
        self.assertEqual(self.ids(agent, "A key requirement is: cotton shell.")[0], "right")

    def test_zero_budget_is_not_missing_and_unknown_price_is_not_free(self) -> None:
        agent = self.agent([
            product("free", price=0), product("paid", price=1, count=1000),
            product("unknown", price=None, count=2000),
        ])
        self.assertEqual(self.ids(agent, "A key requirement is: budget 0.")[0], "free")
        self.assertEqual(agent.diagnostics("session")["top_evidence"][-1]["parent_asin"], "paid")

    def test_budget_bounds_and_ranges_are_numeric_not_lexical(self) -> None:
        rows = [product("low", price=10, count=2000), product("middle", price=20),
                product("high", price=30, count=1000)]
        agent = self.agent(rows)
        self.assertEqual(self.ids(agent, "A key requirement is: between $15 and $25.")[0], "middle")
        agent.reset("session", {})
        self.assertEqual(self.ids(agent, "A key requirement is: over $20.")[0], "high")

    def test_nonfinite_and_overflow_values_do_not_escape_scores(self) -> None:
        rows = [product("normal"), product("invalid", price=float("nan"), count="9" * 600,
                                           average_rating=float("inf"))]
        agent = self.agent(rows)
        response = agent.respond("session", "A key requirement is: under $25.", 1, 10)
        self.assertEqual(response["recommendations"][0]["parent_asin"], "normal")
        self.assertTrue(all(math.isfinite(item["score"]) for item in response["recommendations"]))

    def test_scarce_and_empty_catalogs_return_only_available_records(self) -> None:
        for size in (0, 1, 4):
            with self.subTest(size=size):
                rows = [product(str(index)) for index in range(size)]
                agent = self.agent(rows)
                self.assertEqual(set(self.ids(agent, "Any suitable shirt")), {str(index) for index in range(size)})

    def test_default_and_reduced_presentation_are_true_raw_prefixes(self) -> None:
        rows = [product(f"item-{index:02d}", count=index) for index in range(20)]
        for width in (1, 4, 10):
            agent = self.agent(rows, slate_size=width)
            result = self.ids(agent, "I'm looking for Shirts.")
            self.assertEqual(result, agent.diagnostics("session")["raw_ranked_ids"][:width])
            self.assertEqual(len(result), width)

    def test_ranking_is_deterministic_across_catalog_order_and_question_policy(self) -> None:
        rows = [product(f"item-{index}", count=3, features=["reinforced seams"]) for index in range(12)]
        first = self.agent(rows)
        second = self.agent(list(reversed(rows)), question_policy="none")
        message = "I'm looking for Shirts. A key requirement is: reinforced seams."
        self.assertEqual(self.ids(first, message), self.ids(second, message))

    def test_retry_is_detached_and_conflict_rejected_before_mutation(self) -> None:
        agent = self.agent([product("a")])
        first = agent.respond("session", "I need cotton", 1, 10)
        first["recommendations"].clear()
        repeated = agent.respond("session", "I need cotton", 1, 10)
        self.assertEqual(len(repeated["recommendations"]), 1)
        before = list(agent._sessions["session"].evidence)
        with self.assertRaises(ValueError):
            agent.respond("session", "I need linen", 1, 10)
        self.assertEqual(agent._sessions["session"].evidence, before)
        diagnostics = agent.diagnostics("session")
        diagnostics["raw_ranked_ids"].clear()
        self.assertEqual(agent.diagnostics("session")["raw_ranked_ids"], ["a"])

    def test_different_long_messages_are_not_identical_retries(self) -> None:
        agent = self.agent([product("a")])
        prefix = "x" * 8000
        agent.respond("session", prefix + "first", 1, 10)
        with self.assertRaises(ValueError):
            agent.respond("session", prefix + "second", 1, 10)

    def test_ranking_failure_does_not_commit_partial_turn(self) -> None:
        agent = self.agent([product("a")])
        with patch("mercury.bucket.agent.rank_candidates", side_effect=RuntimeError("temporary")):
            with self.assertRaises(RuntimeError):
                agent.respond("session", "I need cotton", 1, 10)
        self.assertEqual(agent._sessions["session"].last_turn, 0)
        self.assertEqual(len(self.ids(agent, "I need cotton")), 1)

    def test_lru_eviction_and_close_remove_all_session_state(self) -> None:
        agent = self.agent([product("a")], max_sessions=2)
        agent.respond("session", "I need cotton", 1, 10)
        agent.reset("second", {})
        agent.respond("second", "I need linen", 1, 10)
        agent.respond("session", "I need cotton", 1, 10)
        agent.reset("third", {})
        self.assertNotIn("second", agent._sessions)
        self.assertNotIn("second", agent._responses)
        self.assertNotIn("second", agent._diagnostics)
        agent.close()
        self.assertFalse(agent._sessions)
        self.assertFalse(agent._responses)
        self.assertFalse(agent._diagnostics)
        self.assertFalse(agent.index.products)
        with self.assertRaises(RuntimeError):
            agent.reset("new", {})

    def test_reset_and_product_access_do_not_share_mutable_metadata(self) -> None:
        agent = self.agent([product("a", nested={"values": ["one", "two"]})])
        profile = {"preference_tags": ["cotton"]}
        agent.reset("session", profile)
        profile["preference_tags"].append("linen")
        self.assertEqual(agent._sessions["session"].user_profile["preference_tags"], ["cotton"])
        record = agent.product("a")
        record["nested"]["values"].clear()
        self.assertEqual(agent.product("a")["nested"]["values"], ["one", "two"])

    def test_duplicate_catalog_identifiers_are_rejected(self) -> None:
        path = Path(self.directory.name) / "duplicates.jsonl"
        path.write_text(json.dumps(product("same")) + "\n" + json.dumps(product("same")) + "\n")
        with self.assertRaises(ValueError):
            CatalogIndex(path)

    def test_invalid_request_does_not_change_state(self) -> None:
        agent = self.agent([product("a")])
        for turn, width in ((True, 10), (1, False), (0, 10), (11, 10), (1, 11)):
            with self.subTest(turn=turn, width=width), self.assertRaises(ValueError):
                agent.respond("session", "cotton", turn, width)
        self.assertEqual(agent._sessions["session"].last_turn, 0)


if __name__ == "__main__":
    unittest.main()
