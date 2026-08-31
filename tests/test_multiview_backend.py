from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from mercury.catalog import product_from_dict
from mercury.multiview import Agent, Config, make_agent
from mercury.multiview.retrieval import evidence_score, explain
from mercury.state import SessionState
from mercury.types import Preference


def product(identifier: str, title: str, **fields) -> dict:
    return {"parent_asin": identifier, "title": title, "categories": ["Clothing", "Shirts"],
            **fields}


class MultiViewBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.counter = 0

    def build(self, rows: list[dict], config: Config | None = None) -> Agent:
        self.counter += 1
        path = Path(self.directory.name) / f"catalog-{self.counter}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        agent = Agent(path, config)
        self.addCleanup(agent.close)
        return agent

    @staticmethod
    def identifiers(response: dict) -> list[str]:
        return [item["parent_asin"] for item in response["recommendations"]]

    def test_real_protocol_fullwidth_control_and_legal_membership(self) -> None:
        rows = [product(f"P{index:02}", "Cotton shirt", details={"Color": "blue"})
                for index in range(16)]
        control = self.build(rows, Config(fullwidth=False))
        full = make_agent(control.catalog.path, fullwidth=True)
        self.addCleanup(full.close)
        for agent in (control, full):
            agent.reset("s", {})
        messages = ("I need a cotton shirt.", "Blue is ideal.",
                    "No extra preferences to add.", "Actually, red instead.")
        for turn, message in enumerate(messages, 1):
            left = control.respond("s", message, turn, 10)
            right = full.respond("s", message, turn, 10)
            self.assertEqual(left, right)
            identifiers = self.identifiers(right)
            self.assertEqual(len(identifiers), 10)
            self.assertEqual(len(set(identifiers)), 10)
            self.assertLessEqual(set(identifiers), set(full.catalog.by_id))
            self.assertEqual(right["ask_attribute"], "other")
            self.assertEqual(right["usage"], {"prompt_tokens": 0, "completion_tokens": 0})
        self.assertFalse(full.diagnostics["runtime"]["requested_neural"])
        self.assertFalse(full.diagnostics["runtime"]["loaded_neural"])

    def test_every_field_route_member_survives_fusion(self) -> None:
        rows = [product(f"P{index}", "Cotton shirt") for index in range(12)]
        rows.append(product("Z", "Plain garment", details={"Finish": "hidden clasp"}))
        agent = self.build(rows, Config(route_limit=1, constraint_limit=1, category_limit=1))
        agent.reset("s", {})
        agent.respond("s", "A cotton shirt with a hidden clasp.", 1, 10)
        routes = agent.diagnostics["routes"]
        self.assertIn("Z", routes["field:details"])
        admitted = set().union(*(set(identifiers) for identifiers in routes.values()))
        self.assertLessEqual(admitted, set(agent.diagnostics["candidate_ids"]))

    def test_independent_constraint_recovers_below_all_global_view_limits(self) -> None:
        rows = [product("A", "Blue cotton shirt with hidden trim"),
                product("B", "Blue cotton shirt with clasp"),
                product("C", "Blue cotton shirt"),
                product("Z", "Plain shirt", description="care " * 5000 + "hidden clasp")]
        config = Config(route_limit=1, category_limit=1, constraint_limit=1,
                        field_routes=False)
        agent = self.build(rows, config)
        agent.reset("s", {})
        agent.respond("s", "A blue cotton shirt with a hidden clasp.", 1, 10)
        routes = agent.diagnostics["routes"]
        view_members = set().union(*(set(ids) for key, ids in routes.items()
                                    if key.startswith("view:")))
        self.assertNotIn("Z", view_members)
        matching_routes = [identifiers for name, identifiers in routes.items()
                           if name.startswith("constraint:")
                           and '"hidden" AND "clasp"' in agent.diagnostics["queries"][name]]
        self.assertTrue(matching_routes)
        self.assertTrue(any("Z" in identifiers for identifiers in matching_routes))
        self.assertIn("Z", agent.diagnostics["candidate_ids"])

    def test_category_rescue_keeps_category_members_despite_descriptive_noise(self) -> None:
        rows = [product("A", "Boots with blue cotton canvas pockets", categories=["Footwear", "Boots"]),
                product("Z", "Plain garment", categories=["Clothing", "Shirts"])]
        agent = self.build(rows, Config(route_limit=1, category_limit=1,
                                       constraint_routes=False, field_routes=False))
        agent.reset("s", {})
        agent.respond("s", "A shirt in blue cotton canvas with pockets.", 1, 10)
        self.assertIn("Z", agent.diagnostics["routes"]["category"])
        self.assertIn("Z", agent.diagnostics["candidate_ids"])

    def test_long_raw_metadata_is_searchable_and_evidence_keeps_its_field(self) -> None:
        rows = [product("A", "Plain bag", categories=["Bags"]),
                product("Z", "Plain bag", categories=["Bags"],
                        features=["care " * 5000 + "concealed key compartment"])]
        agent = self.build(rows)
        agent.reset("s", {})
        response = agent.respond("s", "A bag with a concealed key compartment.", 1, 10)
        self.assertEqual(self.identifiers(response)[0], "Z")
        fields = agent.diagnostics["evidence"]["Z"]
        witness = next(item for item in fields if item["value"] == "concealed key compartment")
        self.assertEqual(witness["status"], "supported")
        self.assertEqual(witness["fields"], ["features"])
        self.assertGreater(len(agent.catalog.by_id["Z"].fields["features"]), 16000)

    def test_correction_retires_only_changed_color_and_no_preference_is_scoped(self) -> None:
        rows = [product("A", "Blue cotton shirt with pockets"),
                product("B", "Red cotton shirt with pockets"),
                product("C", "Red polyester shirt")]
        agent = self.build(rows)
        agent.reset("s", {})
        agent.respond("s", "I need a blue cotton shirt with pockets.", 1, 10)
        response = agent.respond("s", "Correction: red instead.", 2, 10)
        self.assertEqual(self.identifiers(response)[0], "B")
        state = agent.sessions["s"]
        values = {(p.attribute, p.value, p.polarity) for p in state.active_preferences()}
        self.assertIn(("material", "cotton", 1), values)
        self.assertIn(("feature", "pockets", 1), values)
        self.assertIn(("color", "red", 1), values)
        self.assertNotIn(("color", "blue", 1), values)
        self.assertNotIn('"blue"', " ".join(agent.diagnostics["queries"].values()))
        agent.respond("s", "I no longer have a color preference.", 3, 10)
        values = {(p.attribute, p.value, p.polarity) for p in state.active_preferences()}
        self.assertNotIn(("color", "red", 1), values)
        self.assertIn(("material", "cotton", 1), values)
        self.assertIn(("feature", "pockets", 1), values)

    def test_explicit_alternatives_are_one_evidence_vote_and_admit_each_choice(self) -> None:
        red = product_from_dict(product("R", "Red cotton shirt"))
        blue = product_from_dict(product("B", "Blue cotton shirt"))
        both = product_from_dict(product("RB", "Red blue cotton shirt"))
        state = SessionState({}, alternatives_mode="grouped")
        state.update("A cotton shirt in either red or blue.", 1)
        colors = [item for item in state.active_preferences() if item.attribute == "color"]
        self.assertEqual(len(colors), 2)
        self.assertEqual(len({item.alternative_group for item in colors}), 1)
        self.assertEqual(evidence_score(red, colors), evidence_score(blue, colors))
        self.assertEqual(evidence_score(red, colors), evidence_score(both, colors))
        agent = self.build([product("R", "Red cotton shirt"),
                            product("B", "Blue cotton shirt"),
                            product("G", "Green cotton shirt")])
        agent.reset("s", {})
        response = agent.respond("s", "A cotton shirt in either red or blue.", 1, 10)
        self.assertEqual(set(self.identifiers(response)[:2]), {"R", "B"})
        expressions = list(agent.diagnostics["queries"].values())
        self.assertTrue(any('(\"red\") OR (\"blue\")' in item for item in expressions))

    def test_negative_support_unknown_and_absence_are_distinct(self) -> None:
        excluded = Preference("material", "leather", 1, "No leather.", hard=True, polarity=-1)
        rows = [product("A", "Leather shirt"), product("B", "Cotton shirt without leather"),
                product("C", "Plain shirt")]
        products = [product_from_dict(row) for row in rows]
        reports = [explain(item, [excluded])[0] for item in products]
        self.assertEqual([item["status"] for item in reports],
                         ["contradicted", "supported", "unknown"])
        self.assertEqual(reports[-1]["fields"], [])
        agent = self.build(rows)
        agent.reset("s", {})
        response = agent.respond("s", "A shirt, no leather.", 1, 10)
        self.assertNotEqual(self.identifiers(response)[0], "A")
        self.assertNotIn('"leather"', " ".join(agent.diagnostics["queries"].values()))

    def test_budget_witness_is_price_not_arbitrary_text_fields(self) -> None:
        preference = Preference("budget", "<=30", 1, "Under $30", hard=True)
        known = product_from_dict(product("A", "Shirt", price=20))
        unknown = product_from_dict(product("B", "Shirt", price="from $20"))
        self.assertEqual(explain(known, [preference])[0]["fields"], ["price"])
        self.assertEqual(explain(unknown, [preference])[0]["status"], "unknown")

    def test_observed_hard_exclusion_cannot_be_overridden_by_positive_matches(self) -> None:
        rows = [product("A", "Blue cotton shirt with pockets and leather trim",
                        features=["breathable lightweight adjustable waterproof"]),
                product("Z", "Plain shirt")]
        agent = self.build(rows)
        agent.reset("s", {})
        response = agent.respond(
            "s", "I need a blue cotton shirt with pockets. Breathable and lightweight would be nice. "
            "It should be adjustable and waterproof. No leather.", 1, 10,
        )
        self.assertEqual(self.identifiers(response), ["Z", "A"])
        self.assertIn("A", agent.diagnostics["candidate_ids"])
        self.assertGreater(response["recommendations"][0]["score"],
                           response["recommendations"][1]["score"])
        reports = agent.diagnostics["evidence"]["Z"]
        self.assertEqual(next(item for item in reports if item["value"] == "leather")["status"],
                         "unknown")

    def test_component_binding_and_or_unknown_do_not_create_false_hard_violations(self) -> None:
        rows = [product("A", "Jacket", categories=["Jackets"],
                        details={"Construction": "cotton outer and polyester lining"}),
                product("Z", "Jacket", categories=["Jackets"],
                        details={"Construction": "polyester outer and cotton lining"})]
        agent = self.build(rows)
        agent.reset("s", {})
        response = agent.respond("s", "I need a jacket with a cotton lining.", 1, 10)
        self.assertEqual(self.identifiers(response)[0], "Z")
        self.assertTrue(any(item.scope == "lining" for item in agent.sessions["s"].active_preferences()))
        witness = next(item for item in agent.diagnostics["evidence"]["Z"] if item["value"] == "cotton")
        self.assertEqual(witness["fields"], ["details"])
        state = SessionState({}, alternatives_mode="grouped")
        state.update("I need either red or blue.", 1)
        unknown_choice = product_from_dict(product("B", "Not red shirt"))
        # Red is contradicted but blue is unknown, so the OR remains unknown.
        self.assertEqual(evidence_score(unknown_choice, state.active_preferences()), 0.0)

    def test_retry_is_idempotent_and_reset_clears_history(self) -> None:
        agent = self.build([product("A", "Blue shirt"), product("B", "Red shirt")])
        agent.reset("s", {})
        response = agent.respond("s", "A blue shirt.", 1, 10)
        before = agent.sessions["s"].semantic_signature()
        response["recommendations"].clear()
        retry = agent.respond("s", "A blue shirt.", 1, 10)
        self.assertEqual(len(retry["recommendations"]), 2)
        self.assertEqual(agent.sessions["s"].semantic_signature(), before)
        with self.assertRaises(ValueError):
            agent.respond("s", "A red shirt.", 1, 10)
        agent.reset("s", {})
        self.assertEqual(agent.sessions["s"].active_preferences(), [])
        self.assertEqual(self.identifiers(agent.respond("s", "A red shirt.", 1, 10))[0], "B")

    def test_session_budget_profile_isolation_and_no_lookup_from_session_identity(self) -> None:
        rows = [product("A", "Blue shirt"), product("B", "Red shirt")]
        agent = self.build(rows, Config(max_sessions=1))
        profile = {"target_id": "B", "target_features": ["red"], "preference_tags": ["red"]}
        agent.reset("first", profile)
        profile["target_id"] = "A"
        self.assertEqual(agent.sessions["first"].profile["target_id"], "B")
        left = agent.respond("first", "A blue shirt.", 1, 10)
        agent.reset("different", {})
        right = agent.respond("different", "A blue shirt.", 1, 10)
        self.assertEqual(left, right)
        self.assertNotIn("first", agent.sessions)
        self.assertNotIn("first", agent._responses)

    def test_cached_diagnostics_belong_to_the_retried_session(self) -> None:
        agent = self.build([product("A", "Blue shirt"), product("B", "Red shirt")])
        agent.reset("one", {})
        agent.respond("one", "Blue shirt.", 1, 10)
        first = json.loads(json.dumps(agent.diagnostics))
        agent.diagnostics["active_preferences"].clear()
        agent.reset("two", {})
        agent.respond("two", "Red shirt.", 1, 10)
        agent.respond("one", "Blue shirt.", 1, 10)
        self.assertEqual(json.loads(json.dumps(agent.diagnostics)), first)

    def test_output_cap_final_turn_and_sparse_query_fallback(self) -> None:
        agent = self.build([product(f"P{index:02}", "Shirt") for index in range(12)])
        for index, (top_k, count) in enumerate(((100, 10), (3, 3), (0, 0), (-2, 0), (True, 0))):
            session = str(index)
            agent.reset(session, {})
            response = agent.respond(session, "", 10, top_k)
            self.assertEqual(len(response["recommendations"]), count)
            self.assertIsNone(response["ask_attribute"])

    def test_query_budget_is_reported_and_does_not_remove_scoring_preferences(self) -> None:
        agent = self.build([product("A", "Blue cotton shirt with pockets"),
                            product("B", "Red polyester shirt")], Config(max_constraints=1))
        agent.reset("s", {})
        agent.respond("s", "A blue cotton shirt with pockets.", 1, 10)
        self.assertEqual(agent.diagnostics["constraints_omitted_from_admission"], 2)
        values = {item["value"] for item in agent.diagnostics["evidence"]["A"]}
        self.assertLessEqual({"blue", "cotton", "pockets"}, values)

    def test_config_and_catalog_validation(self) -> None:
        for fields in ({"route_limit": True}, {"max_constraints": 0},
                       {"rrf_constant": float("nan")}, {"evidence_weight": -1},
                       {"fullwidth": 1}, {"max_query_terms": 1000000},
                       {"views": ("missing",)}, {"views": ("identity", "identity")},
                       {"state_mode": "history"}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                Config(**fields)
        with self.assertRaises(ValueError):
            self.build([product("A", "Shirt"), product("A", "Duplicate")])
        self.assertFalse(replace(Config(), fullwidth=False).fullwidth)

    def test_close_clears_profiles_replay_and_diagnostics_in_both_state_modes(self) -> None:
        for mode in ("typed", "raw"):
            with self.subTest(mode=mode):
                agent = self.build([product("A", "Blue shirt")], Config(state_mode=mode))
                profile = {"preference_tags": ["blue"]}
                agent.reset("s", profile)
                profile["preference_tags"].append("red")
                self.assertEqual(agent.sessions["s"].profile["preference_tags"], ["blue"])
                agent.respond("s", "I'm looking for Shirts.", 1, 10)
                agent.close()
                self.assertFalse(agent.sessions)
                self.assertFalse(agent._responses)
                self.assertFalse(agent.diagnostics)

    def test_raw_arm_keeps_full_active_phrases_and_numeric_components(self) -> None:
        rows = [product("A", "Cotton shirt", features=["82% cotton 18% linen", "curved drop tail hem"]),
                product("B", "Cotton shirt", features=["100% cotton", "straight hem"])]
        agent = self.build(rows, Config(state_mode="raw"))
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Shirts, but I'm still exploring.", 1, 10)
        response = agent.respond(
            "s", "For that, what matters is: 82% cotton 18% linen; curved drop tail hem.", 2, 10,
        )
        state = agent.sessions["s"]
        values = {item.value for item in state.active_preferences()}
        self.assertIn("82% cotton 18% linen", values)
        self.assertIn("curved drop tail hem", values)
        self.assertIn("82% cotton 18% linen", state.query())
        self.assertEqual(self.identifiers(response)[0], "A")
        self.assertEqual(agent.diagnostics["config"]["state_mode"], "raw")

    def test_raw_arm_uses_live_replacements_without_resurrecting_history(self) -> None:
        agent = self.build([product("A", "Red cotton shirt with rounded hem"),
                            product("B", "Blue cotton shirt with rounded hem")],
                           Config(state_mode="raw"))
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Shirts.", 1, 10)
        agent.respond("s", "For that, what matters is: red; cotton; rounded hem.", 2, 10)
        response = agent.respond("s", "Actually, what I need is: blue.", 3, 10)
        self.assertEqual(self.identifiers(response)[0], "B")
        state = agent.sessions["s"]
        self.assertNotIn("red", state.query())
        self.assertIn("cotton", state.query())
        self.assertIn("rounded hem", state.query())
        self.assertTrue(any(not item.active and item.value == "red" for item in state.preferences))
        self.assertNotIn('"red"', " ".join(agent.diagnostics["queries"].values()))
        agent.respond("s", "I don't have a color preference.", 4, 10)
        self.assertNotIn("blue", state.query())
        self.assertIn("cotton", state.query())

    def test_raw_arm_preserves_or_branches_component_scope_and_exclusions(self) -> None:
        rows = [product("A", "Jacket", categories=["Jackets"],
                        details={"Construction": "cotton lining"}),
                product("B", "Jacket", categories=["Jackets"],
                        details={"Construction": "silk lining"}),
                product("C", "Jacket", categories=["Jackets"],
                        details={"Construction": "cotton upper and leather lining"})]
        agent = self.build(rows, Config(state_mode="raw"))
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Jackets.", 1, 10)
        response = agent.respond("s", "For that, what matters is: lining: cotton or silk.", 2, 10)
        state = agent.sessions["s"]
        choices = [item for item in state.active_preferences() if item.alternative_group]
        self.assertEqual({item.value for item in choices}, {"cotton", "silk"})
        self.assertEqual({item.scope for item in choices}, {"lining"})
        self.assertEqual(len({item.alternative_group for item in choices}), 1)
        self.assertEqual(set(self.identifiers(response)[:2]), {"A", "B"})
        self.assertIn("lining: cotton or silk", state.query())
        agent.respond("s", "No cotton lining.", 3, 10)
        self.assertFalse(any(item.polarity == 1 and item.value == "cotton"
                             for item in state.active_preferences()))
        self.assertTrue(any(item.polarity == -1 and item.scope == "lining"
                            for item in state.active_preferences()))
        self.assertNotIn('"cotton"', " ".join(agent.diagnostics["queries"].values()))

    def test_raw_arm_default_and_fullwidth_have_identical_ranked_results(self) -> None:
        rows = [product(f"P{index:02}", "Cotton shirt", features=["rounded hem"])
                for index in range(12)]
        normal = self.build(rows, Config(state_mode="raw", fullwidth=False))
        full = make_agent(normal.catalog.path, config=Config(state_mode="raw"), fullwidth=True)
        self.addCleanup(full.close)
        for agent in (normal, full):
            agent.reset("s", {})
        for turn, message in enumerate(("I'm looking for Shirts.",
                                        "For that, what matters is: rounded hem."), 1):
            left, right = (agent.respond("s", message, turn, 10) for agent in (normal, full))
            self.assertEqual(left, right)
            self.assertEqual(len(self.identifiers(left)), 10)

    def test_raw_alternative_identity_survives_an_unrelated_retraction(self) -> None:
        agent = self.build([product("A", "Blue cotton shirt"), product("B", "Blue linen shirt")],
                           Config(state_mode="raw"))
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Shirts.", 1, 10)
        agent.respond("s", "For that, what matters is: blue; cotton or linen.", 2, 10)
        state = agent.sessions["s"]
        before = {item.alternative_group for item in state.active_preferences()
                  if item.alternative_group is not None}
        agent.respond("s", "I don't have a preference for color.", 3, 10)
        after = {item.alternative_group for item in state.active_preferences()
                 if item.alternative_group is not None}
        self.assertEqual(len(before), 1)
        self.assertEqual(after, before)
        self.assertFalse(any(not item.active and item.alternative_group in before
                             for item in state.preferences))


if __name__ == "__main__":
    unittest.main()
