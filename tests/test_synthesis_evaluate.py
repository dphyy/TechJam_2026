import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.run import source_hashes
from experiments.synthesis_evaluate import (
    ExperimentConfig,
    OpenQuestionPlanner,
    SearchExperiment,
    SynthesisAgent,
)
from mercury.lexical.dialogue import SessionState
from mercury.lexical.product_features import ProductFeatureStore
from mercury.lexical.retrieval import SearchResult


def product(identifier, score, exact=1):
    return {"parent_asin": identifier, "title": "Blue cotton shirt",
            "categories": ["Shirts"], "features": ["cotton"],
            "_rank_score": score, "_hard_constraint_count": 1,
            "_hard_constraint_exact_count": exact, "_category_leaf_match": True,
            "_exact_constraint_index_match": bool(exact),
            "_constraint_sequence_match": True, "_catalog_tiebreak": (1.0, 1.0, 1)}


class ExperimentTest(unittest.TestCase):
    def setUp(self):
        self.rows = [product("a", 3), product("b", 2), product("c", 1, exact=0)]
        self.result = SearchResult([(row["parent_asin"], row["_rank_score"])
                                    for row in self.rows], self.rows)
        self.inner = SimpleNamespace(search_with_context=lambda state, limit: self.result)
        self.state = SimpleNamespace(semantic_query=lambda: "A cotton shirt")

    def test_semantic_ranking_cannot_promote_a_known_constraint_mismatch(self):
        ranker = SimpleNamespace(prompt_tokens=0,
                                 score=lambda *args, **kwargs: {"a": 0, "b": 1, "c": 100})
        search = SearchExperiment(self.inner, ExperimentConfig(
            ranking_policy="semantic_constraints"), ranker)
        result = search.search_with_context(self.state)
        self.assertEqual([item[0] for item in result.recommendations], ["b", "a", "c"])
        self.assertEqual(set(item[0] for item in result.recommendations), {"a", "b", "c"})
        self.assertEqual(result.recommendations[0][1], 2)

    def test_partial_foreign_nonfinite_and_failed_model_results_do_not_change_ranking(self):
        for logits in ({"a": 1}, {"a": 1, "b": 1, "c": 1, "foreign": 9},
                       {"a": float("nan"), "b": 1, "c": 1}):
            with self.subTest(logits=logits):
                ranker = SimpleNamespace(prompt_tokens=0, score=lambda *args, **kwargs: logits)
                search = SearchExperiment(self.inner, ExperimentConfig(
                    ranking_policy="semantic_ties"), ranker)
                self.assertIs(search.search_with_context(self.state), self.result)
                self.assertEqual(search.last_diagnostics["fallbacks"], ["ValueError"])

        def failing(*args, **kwargs):
            raise TimeoutError("temporary failure")

        search = SearchExperiment(self.inner, ExperimentConfig(ranking_policy="semantic_ties"),
                                  SimpleNamespace(prompt_tokens=0, score=failing))
        self.assertIs(search.search_with_context(self.state), self.result)
        self.assertEqual(search.last_diagnostics["fallbacks"], ["TimeoutError"])

    def test_bounded_prefix_keeps_tail_and_all_members(self):
        seen = []

        def score(query, pool, **kwargs):
            seen.extend(item.product.parent_asin for item in pool)
            return {"a": 0, "b": 1}

        search = SearchExperiment(self.inner, ExperimentConfig(
            ranking_policy="semantic_ties", neural_prefix=2),
            SimpleNamespace(prompt_tokens=0, score=score))
        result = search.search_with_context(self.state)
        self.assertEqual(seen, ["a", "b"])
        self.assertIs(result.candidates[-1], self.rows[-1])
        self.assertEqual([item[0] for item in result.recommendations], ["b", "a", "c"])

    def test_lexical_control_never_calls_model(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("control must not infer")

        search = SearchExperiment(self.inner, ExperimentConfig(),
                                  SimpleNamespace(prompt_tokens=0, score=forbidden))
        self.assertIs(search.search_with_context(self.state), self.result)

    def test_full_width_and_missing_optional_assets_use_real_agent_responses(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            catalog.write_text("\n".join(json.dumps(product(str(index), 1)) for index in range(12)))
            agent = SynthesisAgent(catalog, ExperimentConfig(
                ranking_policy="semantic_ties", full_width=True))
            try:
                agent.reset("case", {})
                response = agent.respond("case", "I'm looking for Shirts. A key requirement is: cotton.", 1, 10)
                ids = [row["parent_asin"] for row in response["recommendations"]]
                self.assertEqual(len(ids), 10)
                self.assertEqual(ids, agent.last_diagnostics["ranked_context_ids"][:10])
                self.assertEqual(agent.model_error, "ValueError")
                self.assertEqual(agent.last_diagnostics["fallbacks"], ["optional_ranker_unavailable"])
            finally:
                agent.close()

    def test_open_question_honors_no_preference_and_end_of_conversation(self):
        planner = OpenQuestionPlanner(ProductFeatureStore())
        state = SessionState({})
        candidates = [product("a", 2), product("b", 1)]
        for item in candidates:
            item["_features"] = planner.feature_store.get_or_add(
                item["parent_asin"], {"title": item["title"], "categories": "Shirts",
                                     "features": "cotton"})
        self.assertEqual(planner.choose(state, candidates, 1).attribute, "other")
        state.no_preference_attributes.update(("other", "feature", "material", "color", "budget",
                                              "size", "style", "use_case", "category", "brand"))
        self.assertIsNone(planner.choose(state, candidates, 2).attribute)
        self.assertIsNone(planner.choose(SessionState({}), candidates, 10).attribute)

    def test_runtime_receipt_covers_nested_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mercury" / "nested").mkdir(parents=True)
            source = root / "mercury" / "nested" / "runtime.py"
            source.write_text("value = 1\n")
            real_path = Path
            with patch("experiments.run.Path", side_effect=lambda value: root / real_path(value)):
                before = source_hashes()
                source.write_text("value = 2\n")
                after = source_hashes()
            self.assertNotEqual(before, after)
            self.assertIn(source.as_posix(), before)


if __name__ == "__main__":
    unittest.main()
