"""Authored end-to-end regressions for price guards and changing product intent."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from agent import Agent
from mercury.lexical.budgets import budgets_allow, parse_budgets
from mercury.lexical.config import DEFAULT_AGENT_CONFIG
from mercury.lexical.dialogue import Evidence, SessionState
from mercury.lexical.paging import semantic_signature
from mercury.lexical.preprocessing import build_catalog_index
from mercury.lexical.product_features import ProductFeatureStore
from mercury.lexical.retrieval import CatalogSearch


class PriceCategoryTest(unittest.TestCase):
    def agent(self, rows=None, *, paging=True, prebuilt=False):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "catalog.jsonl"
        rows = rows if rows is not None else [
            {"parent_asin": f"{category}{i}", "title": f"Blue cotton {category}",
             "categories": [category], "features": ["blue cotton", "washable"],
             "price": 20 if i < 10 else 200, "average_rating": 4, "rating_number": 10}
            for category in ("Bags", "Shoes", "Shirts") for i in range(20)
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows))
        if prebuilt:
            build_catalog_index(path)
        agent = Agent(path, config=replace(DEFAULT_AGENT_CONFIG, guarded_paging=paging))
        self.addCleanup(agent.close)
        agent.reset("s", {})
        return agent

    @staticmethod
    def ids(response):
        return [row["parent_asin"] for row in response["recommendations"]]

    def test_budget_forms_compile_complete_values(self):
        cases = {
            "under $1,250.75": ("under", 1250.75),
            "below USD 1,250.75": ("under", 1250.75),
            "under 50 dollars": ("under", 50),
            "at most $50": ("maximum", 50),
            "no more than $50": ("maximum", 50),
            "not over $50": ("maximum", 50),
            "budget is $50": ("maximum", 50),
            "budget under 50": ("under", 50),
            "at least $50": ("minimum", 50),
            "not under $50": ("minimum", 50),
            "above $50": ("over", 50),
            "around $50": ("around", 50),
            "$50": ("around", 50),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual([(b.mode, b.amount) for b in parse_budgets(text)], [expected])

    def test_invalid_amounts_and_non_price_measurements_are_not_partial_budgets(self):
        for text in ("under $1,25.00", "under $12,34,567", "$1e3", "$-50", "$50cm", "under 50 inches",
                     "size 50", "price $1.2.3", "$" + "9" * 400):
            with self.subTest(text=text):
                self.assertEqual(parse_budgets(text), ())

    def test_ranges_are_constraints_on_both_ends(self):
        for text in ("between $20 and $50", "from 20 to 50 dollars", "$20-$50"):
            with self.subTest(text=text):
                budgets = parse_budgets(text)
                self.assertEqual([(b.mode, b.amount) for b in budgets], [("minimum", 20), ("maximum", 50)])
                self.assertFalse(all(b.allows(19) for b in budgets))
                self.assertFalse(all(b.allows(51) for b in budgets))
                self.assertTrue(all(b.allows(20) for b in budgets))
                self.assertTrue(all(b.allows(50) for b in budgets))

    def test_maximum_strict_upper_lower_and_unknown_prices_share_guard_semantics(self):
        store = ProductFeatureStore()
        for wording, accepted in (("under $50", {0, 49}), ("maximum $50", {0, 49, 50}),
                                  ("at least $50", {50, 51}), ("over $50", {51}),
                                  ("around $50", {0, 49, 50, 51})):
            query = store.compile_query([Evidence(wording, 3.8, "hard_constraint", 1)])
            for price in (0, 49, 50, 51, None):
                with self.subTest(wording=wording, price=price):
                    product = store.add(f"{wording}/{price}", {}, price=price)
                    self.assertEqual(CatalogSearch._semantic_violation(product, query),
                                     price is not None and price not in accepted)

    def test_paging_preserves_budget_after_unseen_pool_exhaustion(self):
        for paging in (False, True):
            agent = self.agent(paging=paging)
            seen = set()
            for turn in range(1, 11):
                response = agent.respond("s", "I'm looking for bags. A key requirement is: under $50."
                                         if turn == 1 else "Those options aren't right.", turn, 10)
                ids = self.ids(response)
                self.assertTrue(ids)
                self.assertTrue(set(ids) <= {f"Bags{i}" for i in range(10)})
                self.assertTrue(all(c["evidence"][-1]["status"] == "supported"
                                    for c in agent.last_diagnostics["constraint_checks"]))
                if paging and turn == 2:
                    self.assertEqual(len(set(ids) & seen), 1)
                    self.assertEqual(agent.last_diagnostics["paging"]["reason"], "unseen_exhausted_ranked_fill")
                seen.update(ids)

    def test_budget_parser_and_diagnostic_witness_agree(self):
        rows = [{"parent_asin": str(price), "title": "Bag", "categories": ["Bags"], "price": price}
                for price in (20, 1250, 1251, None)]
        agent = self.agent(rows, paging=False)
        agent.respond("s", "I'm looking for bags.", 1, 10)
        agent.respond("s", "Those options aren't right. A key requirement is: under $1,250.75.", 2, 10)
        for check in agent.last_diagnostics["constraint_checks"]:
            budget = next(row for row in check["evidence"] if "$1,250.75" in row["value"])
            if check["parent_asin"] == "None":
                self.assertEqual(budget["status"], "unknown")
            else:
                self.assertEqual(budget["witnesses"][0]["requested_value"], 1250.75)
                self.assertEqual(budget["status"], "contradicted" if check["parent_asin"] == "1251" else "supported")

    def test_negative_upper_bound_wording_is_not_an_exclusion(self):
        for text, mode in (("No more than $50", "maximum"), ("Not over $50", "maximum"),
                           ("Not under $50", "minimum"), ("No less than $50", "minimum")):
            state = SessionState({})
            state.observe(text, 1)
            query = ProductFeatureStore().compile_query(state.evidence)
            self.assertEqual([(b.mode, b.amount) for b in query.budgets], [(mode, 50)], text)

    def test_budget_correction_replaces_old_bound_and_no_preference_retires_it(self):
        agent = self.agent()
        agent.respond("s", "I'm looking for bags. A key requirement is: under $50.", 1, 10)
        agent.respond("s", "Correction: budget under $250.", 2, 10)
        state = agent._sessions["s"]
        self.assertEqual([b.amount for b in agent.search.feature_store.compile_query(state.evidence).budgets], [250])
        agent.respond("s", "I have no budget preference.", 3, 10)
        self.assertFalse(agent.search.feature_store.compile_query(state.evidence).budgets)

    def test_retracting_or_replacing_inline_budget_preserves_material(self):
        for correction in ("I have no budget preference.", "Correction: under $250."):
            agent = self.agent()
            agent.respond("s", "I'm looking for bags. A key requirement is: cotton under $50.", 1, 10)
            agent.respond("s", correction, 2, 10)
            self.assertTrue(any(e.text == "cotton" for e in agent._sessions["s"].evidence))
            budgets = agent.search.feature_store.compile_query(agent._sessions["s"].evidence).budgets
            self.assertEqual([b.amount for b in budgets], [] if "no budget" in correction else [250])

    def test_negated_spending_limit_is_a_maximum_not_a_product_exclusion(self):
        for message in ("I don't want to spend more than $50.", "I cannot pay over $50.",
                        "I won't spend above $50."):
            state = SessionState({})
            state.observe(message, 1)
            self.assertEqual([(b.mode, b.amount) for b in ProductFeatureStore().compile_query(state.evidence).budgets],
                             [("maximum", 50)], message)

    def test_budget_alternatives_do_not_become_impossible_conjunctions(self):
        for message in ("under $50 or over $100", "between $20 and $50 or between $100 and $150"):
            budgets = parse_budgets(message)
            self.assertTrue(budgets_allow(25, budgets))
            self.assertTrue(budgets_allow(125, budgets))
            self.assertFalse(budgets_allow(75, budgets))
            state = SessionState({})
            state.observe(f"A key requirement is: {message}.", 1)
            query = ProductFeatureStore().compile_query(state.evidence)
            self.assertTrue(budgets_allow(25, query.budgets))
            self.assertTrue(budgets_allow(125, query.budgets))

    def test_short_category_corrections_work_in_memory_and_with_prebuilt_index(self):
        for prebuilt in (False, True):
            agent = self.agent(prebuilt=prebuilt)
            for message in ("Correction: shoes.", "Actually, I want shoes.", "Please make that shoes.",
                            "I want shoes instead.", "Actually, I'm looking for shoes."):
                with self.subTest(prebuilt=prebuilt, message=message):
                    agent.reset("s", {})
                    agent.respond("s", "I'm looking for bags. A key requirement is: under $50; cotton.", 1, 10)
                    result = agent.respond("s", message, 2, 10)
                    self.assertEqual(agent._sessions["s"].category_text.lower(), "shoes")
                    self.assertTrue(self.ids(result))
                    self.assertTrue(all(key.startswith("Shoes") for key in self.ids(result)))
                    self.assertTrue(any(e.text == "cotton" for e in agent._sessions["s"].evidence))
                    self.assertEqual([b.amount for b in agent.search.feature_store.compile_query(
                        agent._sessions["s"].evidence).budgets], [50])
                    self.assertTrue(agent.last_diagnostics["paging"]["reset"])

    def test_opening_category_and_modifiers_are_separate(self):
        agent = self.agent()
        for message in ("I want blue cotton shoes.", "I'm looking for shoes under $1,250.75."):
            agent.reset("s", {})
            agent.respond("s", message, 1, 10)
            self.assertEqual(agent._sessions["s"].category_text, "shoes")
            self.assertGreater(len(agent._sessions["s"].evidence), 1)

    def test_category_question_accepts_bare_taxonomy_answer(self):
        agent = self.agent()
        agent._sessions["s"].record_question("category")
        result = agent.respond("s", "Shoes", 1, 10)
        self.assertEqual(agent._sessions["s"].category_text, "Shoes")
        self.assertTrue(all(key.startswith("Shoes") for key in self.ids(result)))

    def test_category_alternatives_match_either_branch_and_can_replace_an_aisle(self):
        agent = self.agent()
        for opening in ("I'm looking for shoes or shirts.", "I want shoes or shirts."):
            agent.reset("s", {})
            result = agent.respond("s", opening, 1, 10)
            self.assertTrue(self.ids(result))
            self.assertTrue(all(key.startswith(("Shoes", "Shirts")) for key in self.ids(result)))
        agent.reset("s", {})
        agent.respond("s", "I'm looking for bags.", 1, 10)
        result = agent.respond("s", "Correction: shoes or shirts.", 2, 10)
        self.assertEqual(agent._sessions["s"].category_text, "shoes or shirts")
        self.assertTrue(self.ids(result))
        self.assertTrue(all(key.startswith(("Shoes", "Shirts")) for key in self.ids(result)))

    def test_component_correction_does_not_change_product_category(self):
        agent = self.agent()
        agent.respond("s", "I'm looking for shoes. A key requirement is: upper: leather.", 1, 10)
        agent.respond("s", "Correction: upper: cotton.", 2, 10)
        self.assertEqual(agent._sessions["s"].category_text, "shoes")

    def test_quoted_and_hypothetical_category_changes_do_not_mutate_intent(self):
        agent = self.agent()
        for message in ('"Correction: shoes"', "If I say correction: shoes, what would happen?",
                        "Please don't change it to shoes.", "The label says 'looking for shoes'.",
                        "If I'm looking for shoes instead, would that work?"):
            agent.reset("s", {})
            agent.respond("s", "I'm looking for bags.", 1, 10)
            before = semantic_signature(agent._sessions["s"])
            agent.respond("s", message, 2, 10)
            self.assertEqual(semantic_signature(agent._sessions["s"]), before, message)


if __name__ == "__main__":
    unittest.main()
