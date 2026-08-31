from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.catalog_ranker_train import (
    CatalogRow,
    Fact,
    QuerySpec,
    TrainingConfig,
    catalog_facts,
    generate_pairs,
    load_partitions,
    main,
    partition,
    supervision,
    train,
)
from mercury.lexical.catalog_ranker import (
    FEATURE_NAMES,
    FEATURE_SCALES,
    MODEL_KEYS,
    CatalogLinearRanker,
    CatalogRankingSearch,
    canonical_json,
    feature_vector,
    product_features,
    sha256_file,
)
from mercury.lexical.dialogue import Evidence, SessionState
from mercury.lexical.product_features import ProductFeatureStore
from mercury.lexical.retrieval import SearchResult


def row(identifier: str, material: str | None = "cotton", color: str | None = "red", *,
        title: str = "Cotton shirt", **extra: object) -> dict:
    details = {}
    if material is not None:
        details["Material"] = material
    if color is not None:
        details["Color"] = color
    return {"parent_asin": identifier, "title": title, "categories": ["Clothing", "Shirts"],
            "details": details, "features": [], "description": [], **extra}


def candidate(identifier: str, material: str, *, score: float = 1, tier: int = 0,
              violation: bool = False) -> dict:
    return {**row(identifier, material), "_rank_score": score, "_hard_constraint_count": tier,
            "_hard_constraint_exact_count": tier, "_category_leaf_match": True,
            "_exact_constraint_index_match": bool(tier), "_semantic_violation": violation}


class CatalogRankerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.catalog = self.root / "catalog.jsonl"
        self.config = TrainingConfig(max_train_rows=40, max_validation_rows=24, max_queries_per_split=24,
                                     max_pairs_per_split=128, epochs=20)
        counters = {"train": 0, "validation": 0}
        self.rows = []
        for index in range(500):
            identifier = f"item-{index:03d}"
            split = partition(identifier, self.config)
            if counters[split] >= 16:
                continue
            counter = counters[split]
            material = ("cotton", "linen", "leather", None)[counter % 4]
            color = ("red", "blue")[counter // 4 % 2]
            self.rows.append(row(identifier, material, color))
            counters[split] += 1
            if min(counters.values()) == 16:
                break
        self.write_catalog(self.rows)

    def write_catalog(self, rows: list[dict]) -> None:
        self.catalog.write_bytes(b"".join(canonical_json(item) + b"\n" for item in rows))

    def write_model(self, weights: list[float] | None = None, **extra: object) -> tuple[Path, dict, str]:
        payload = {"feature_names": list(FEATURE_NAMES), "scales": list(FEATURE_SCALES),
                   "weights": weights if weights is not None else [0.0] * len(FEATURE_NAMES),
                   "catalog_sha256": sha256_file(self.catalog), "config_sha256": "a" * 64, **extra}
        path = self.root / "model.json"
        path.write_bytes(canonical_json(payload))
        return path, payload, sha256_file(path)

    def ranker(self, weights: list[float] | None = None, *, prefix_limit: int = 30, **extra: object) -> CatalogLinearRanker:
        path, payload, digest = self.write_model(weights, **extra)
        ranker = CatalogLinearRanker(self.catalog, path, expected_model_sha256=digest,
                                    expected_config_sha256=payload["config_sha256"], prefix_limit=prefix_limit)
        self.addCleanup(ranker.close)
        return ranker

    @staticmethod
    def state(text: str = "cotton") -> SessionState:
        return SessionState({}, evidence=[Evidence(text, 1.0, "clarification", 1)], category_text="Shirts")

    @staticmethod
    def result(rows: list[dict]) -> SearchResult:
        return SearchResult([(item["parent_asin"], item["_rank_score"]) for item in rows], rows)

    def test_learning_and_model_serialization_are_deterministic(self) -> None:
        first, first_report = train(self.catalog, self.config)
        second, second_report = train(self.catalog, self.config)
        self.assertEqual(canonical_json(first), canonical_json(second))
        self.assertEqual(first_report, second_report)
        self.assertTrue(any(abs(weight) > 0 for weight in first["weights"]))
        self.assertLess(first_report["train_after"]["logistic_loss"], first_report["train_before"]["logistic_loss"])
        self.assertGreater(first_report["validation"]["pair_accuracy"], .5)
        self.assertGreater(first["weights"][FEATURE_NAMES.index("details_token_coverage")],
                           first["weights"][FEATURE_NAMES.index("title_token_coverage")])

    def test_split_is_disjoint_before_queries_and_negative_sampling(self) -> None:
        splits, receipt = load_partitions(self.catalog, self.config)
        train_ids = {item.identifier for item in splits["train"]}
        validation_ids = {item.identifier for item in splits["validation"]}
        self.assertFalse(train_ids & validation_ids)
        self.assertEqual(receipt["row_overlap"], 0)
        for name, identifiers in (("train", train_ids), ("validation", validation_ids)):
            pairs, _ = generate_pairs(splits[name], self.config)
            self.assertTrue(pairs)
            self.assertTrue(all(pair.positive_id in identifiers and pair.negative_id in identifiers for pair in pairs))
            self.assertTrue(all(partition(identifier, self.config) == name for identifier in identifiers))

    def test_selection_and_learning_are_independent_of_catalog_line_order(self) -> None:
        first, first_report = train(self.catalog, self.config)
        self.write_catalog(list(reversed(self.rows)))
        second, second_report = train(self.catalog, self.config)
        self.assertEqual(first["weights"], second["weights"])
        self.assertEqual(first_report["selected_row_sha256"], second_report["selected_row_sha256"])
        self.assertNotEqual(first["catalog_sha256"], second["catalog_sha256"])

    def test_ambiguous_equally_supported_rows_are_not_made_false_negatives(self) -> None:
        raw = [row("same-a", "cotton", "red"), row("same-b", "cotton", "red"),
               row("other", "leather", "blue"), row("unknown", None, None)]
        rows = [CatalogRow(item["parent_asin"], "shirts", catalog_facts(item), product_features(item)) for item in raw]
        query = QuerySpec("shirts", (Fact("material", "", ("cotton",)),))
        self.assertEqual([supervision(item, query) for item in rows], [1, 1, -1, 0])
        pairs, stats = generate_pairs(rows, self.config)
        self.assertGreater(stats["queries_with_multiple_positives"], 0)
        self.assertGreater(stats["unknown_candidates_skipped"], 0)
        self.assertFalse(any({pair.positive_id, pair.negative_id} == {"same-a", "same-b"} for pair in pairs))
        self.assertFalse(any("unknown" in (pair.positive_id, pair.negative_id) for pair in pairs))

    def test_unstructured_missing_and_negated_fields_cannot_supply_positive_labels(self) -> None:
        self.assertEqual(catalog_facts(row("unknown", None, None, features=["cotton red"])), ())
        self.assertEqual(catalog_facts(row("denied", "no cotton", "not red")), ())
        self.assertEqual(catalog_facts(row("text", details="Material cotton")), ())

    def test_component_facts_do_not_borrow_material_from_other_components(self) -> None:
        raw = row("item", details={"Shell Material": "cotton", "Lining Material": "polyester"})
        parsed = CatalogRow("item", "shirts", catalog_facts(raw), product_features(raw))
        self.assertEqual(supervision(parsed, QuerySpec("shirts", (Fact("material", "shell", ("cotton",)),))), 1)
        self.assertEqual(supervision(parsed, QuerySpec("shirts", (Fact("material", "lining", ("cotton",)),))), -1)

    def test_model_contains_only_feature_parameters_and_hashes(self) -> None:
        model, report = train(self.catalog, self.config)
        self.assertEqual(set(model), MODEL_KEYS)
        serialized = canonical_json(model).decode()
        self.assertFalse(any(item["parent_asin"] in serialized for item in self.rows))
        self.assertNotIn("queries", model)
        self.assertEqual(report["model_sha256"], hashlib.sha256(canonical_json(model)).hexdigest())

    def test_training_reads_only_the_catalog_path(self) -> None:
        original = Path.open
        observed = []
        def limited(path, *args, **kwargs):
            observed.append(path)
            if path != self.catalog:
                raise AssertionError("unexpected input file")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", limited):
            train(self.catalog, self.config)
        self.assertEqual(observed, [self.catalog])

    def test_bounded_sampling_limits_rows_queries_and_pairs(self) -> None:
        config = TrainingConfig(max_train_rows=6, max_validation_rows=4, max_queries_per_split=3,
                                max_pairs_per_split=4, epochs=1)
        splits, receipt = load_partitions(self.catalog, config)
        self.assertLessEqual(receipt["selected_rows"]["train"], 6)
        self.assertLessEqual(receipt["selected_rows"]["validation"], 4)
        for rows in splits.values():
            pairs, stats = generate_pairs(rows, config)
            self.assertLessEqual(len(pairs), 4)
            self.assertLessEqual(stats["queries_considered"], 3)

    def test_training_requires_disjoint_validation_and_rejects_duplicate_ids(self) -> None:
        self.write_catalog([self.rows[0], self.rows[0]])
        with self.assertRaises(ValueError):
            load_partitions(self.catalog, self.config)
        self.write_catalog([row("single")])
        with self.assertRaises(ValueError):
            train(self.catalog, self.config)

    def test_unknown_products_have_finite_bounded_features(self) -> None:
        state = self.state()
        compiled = ProductFeatureStore(max_size=1).compile_query(state.evidence)
        vector = feature_vector({"parent_asin": "unknown"}, compiled, state.category_text)
        self.assertEqual(len(vector), len(FEATURE_NAMES))
        self.assertTrue(all(0 <= value <= 1 for value in vector))
        self.assertEqual(vector[FEATURE_NAMES.index("missing_information")], 1)
        self.assertEqual(vector[FEATURE_NAMES.index("facet_contradiction")], 0)

    def test_trained_artifact_loads_with_external_pins(self) -> None:
        model, report = train(self.catalog, self.config)
        path = self.root / "trained.json"
        path.write_bytes(canonical_json(model))
        ranker = CatalogLinearRanker(self.catalog, path, expected_model_sha256=report["model_sha256"],
                                    expected_config_sha256=report["config_sha256"])
        self.assertTrue(ranker.enabled)
        candidates = [candidate("bad", "leather"), candidate("good", "cotton")]
        result = ranker.rerank(self.result(candidates), self.state())
        self.assertEqual(result.candidates[0]["parent_asin"], "good")

    def test_reranking_stays_within_exact_constraint_tiers_and_preserves_tail(self) -> None:
        weights = [0.0] * len(FEATURE_NAMES)
        weights[FEATURE_NAMES.index("details_token_coverage")] = 1.0
        ranker = self.ranker(weights, prefix_limit=4)
        candidates = [candidate("weak-a", "leather"), candidate("locked", "leather", tier=1),
                      candidate("strong", "cotton"), candidate("excluded", "cotton", violation=True),
                      candidate("tail", "cotton")]
        result = ranker.rerank(self.result(candidates), self.state())
        self.assertEqual([item["parent_asin"] for item in result.candidates],
                         ["strong", "locked", "weak-a", "excluded", "tail"])
        self.assertIs(result.candidates[-1], candidates[-1])
        self.assertEqual(set(item[0] for item in result.recommendations), {item["parent_asin"] for item in candidates})
        self.assertTrue(ranker.last_diagnostics["model_applied"])

    def test_equal_features_do_not_invent_a_unique_winner(self) -> None:
        ranker = self.ranker([.5] * len(FEATURE_NAMES))
        candidates = [candidate("second", "cotton"), candidate("first", "cotton")]
        result = ranker.rerank(self.result(candidates), self.state())
        self.assertEqual([item["parent_asin"] for item in result.candidates], ["second", "first"])

    def test_no_model_or_missing_pins_leaves_the_original_result_unchanged(self) -> None:
        result = self.result([candidate("a", "cotton"), candidate("b", "leather")])
        path, _, _ = self.write_model()
        for ranker in (CatalogLinearRanker(self.catalog), CatalogLinearRanker(self.catalog, path)):
            self.assertFalse(ranker.enabled)
            self.assertIs(ranker.rerank(result, self.state()), result)

    def test_digest_catalog_and_config_mismatch_disable_the_model(self) -> None:
        path, payload, digest = self.write_model()
        for expected_model, expected_config in (("b" * 64, payload["config_sha256"]), (digest, "c" * 64)):
            ranker = CatalogLinearRanker(self.catalog, path, expected_model_sha256=expected_model,
                                        expected_config_sha256=expected_config)
            self.assertFalse(ranker.enabled)
        self.write_catalog(self.rows + [row("new")])
        ranker = CatalogLinearRanker(self.catalog, path, expected_model_sha256=digest,
                                    expected_config_sha256=payload["config_sha256"])
        self.assertFalse(ranker.enabled)

    def test_foreign_model_fields_invalid_scales_and_weights_are_rejected_even_with_new_digest(self) -> None:
        mutations = [
            {"product_ids": ["a"]}, {"weights": [1.0]}, {"weights": [True] * len(FEATURE_NAMES)},
            {"weights": [9.0] * len(FEATURE_NAMES)}, {"scales": [True] * len(FEATURE_NAMES)},
            {"feature_names": list(reversed(FEATURE_NAMES))},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.assertFalse(self.ranker(**mutation).enabled)

    def test_duplicate_json_keys_and_excessive_nesting_fail_closed(self) -> None:
        path = self.root / "bad-model.json"
        for raw in (b'{"weights":[],"weights":[]}', b"[" * 2000 + b"]" * 2000):
            path.write_bytes(raw)
            ranker = CatalogLinearRanker(self.catalog, path, expected_model_sha256=hashlib.sha256(raw).hexdigest(),
                                        expected_config_sha256="a" * 64)
            self.assertFalse(ranker.enabled)

    def test_malformed_or_missing_candidate_tiers_fall_back_without_mutation(self) -> None:
        ranker = self.ranker([.5] * len(FEATURE_NAMES))
        candidates = [candidate("a", "cotton"), candidate("b", "leather")]
        for key, value in (("_semantic_violation", "false"), ("_hard_constraint_exact_count", 4)):
            malformed = deepcopy(candidates)
            malformed[0][key] = value
            result = self.result(malformed)
            self.assertIs(ranker.rerank(result, self.state()), result)
            self.assertFalse(ranker.last_diagnostics["model_applied"])
        missing = deepcopy(candidates)
        del missing[0]["_semantic_violation"]
        result = self.result(missing)
        self.assertIs(ranker.rerank(result, self.state()), result)

    def test_search_adapter_preserves_scores_and_closes_resources(self) -> None:
        weights = [0.0] * len(FEATURE_NAMES)
        weights[FEATURE_NAMES.index("details_token_coverage")] = 1
        ranker = self.ranker(weights)
        result = self.result([candidate("bad", "leather", score=9), candidate("good", "cotton", score=2)])
        closed = []
        adapter = CatalogRankingSearch(SimpleNamespace(search_with_context=lambda *args: result,
                                                        close=lambda: closed.append(True)), ranker)
        reranked = adapter.search_with_context(self.state())
        self.assertEqual(reranked.recommendations, [("good", 2.0), ("bad", 9.0)])
        adapter.close()
        self.assertEqual(closed, [True])
        self.assertFalse(ranker.enabled)

    def test_reranking_preserves_additive_search_stage_receipts(self) -> None:
        weights = [0.0] * len(FEATURE_NAMES)
        weights[FEATURE_NAMES.index("details_token_coverage")] = 1
        ranker = self.ranker(weights)
        result = self.result([candidate("bad", "leather"), candidate("good", "cotton")])
        result = replace(result, candidate_ids=("good", "bad", "outside-prefix"),
                         vector_stage={"attempted": True, "status": "cache_hit", "returned_count": 3})
        reranked = ranker.rerank(result, self.state())
        self.assertEqual(reranked.candidate_ids, result.candidate_ids)
        self.assertEqual(reranked.vector_stage, result.vector_stage)
        self.assertEqual(reranked.prompt_tokens, result.prompt_tokens)
        self.assertEqual(reranked.ranking_mode, result.ranking_mode)
        self.assertEqual(reranked.recommendations[0][0], "good")

    def test_cli_preregisters_split_and_writes_a_loadable_small_artifact(self) -> None:
        output = self.root / "trained-output"
        config_path = self.root / "config.json"
        config_path.write_bytes(canonical_json(asdict(self.config)))
        original_train = train
        def registered_train(catalog, config):
            registration = json.loads((output / "registration.json").read_text())
            self.assertEqual(registration["config"], asdict(self.config))
            self.assertEqual(registration["catalog_sha256"], sha256_file(self.catalog))
            return original_train(catalog, config)
        arguments = ["catalog-ranker", "--catalog", str(self.catalog), "--output", str(output),
                     "--config", str(config_path)]
        with patch("sys.argv", arguments), patch("builtins.print"), \
                patch("experiments.catalog_ranker_train.train", side_effect=registered_train):
            main()
        model = json.loads((output / "model.json").read_text())
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(set(model), MODEL_KEYS)
        self.assertLess((output / "model.json").stat().st_size, 4096)
        ranker = CatalogLinearRanker(self.catalog, output / "model.json",
                                    expected_model_sha256=report["model_sha256"],
                                    expected_config_sha256=report["config_sha256"])
        self.assertTrue(ranker.enabled)

    def test_unknown_query_and_repeated_reranking_are_stable(self) -> None:
        weights = [.5] * len(FEATURE_NAMES)
        ranker = self.ranker(weights)
        result = self.result([candidate("b", "cotton"), candidate("a", "cotton")])
        first = ranker.rerank(result, self.state("unrecognized-property"))
        second = ranker.rerank(result, self.state("unrecognized-property"))
        self.assertEqual(first.recommendations, second.recommendations)
        self.assertEqual(first.recommendations, result.recommendations)
        self.assertEqual([item["parent_asin"] for item in result.candidates], ["b", "a"])

    def test_actual_cli_process_trains_and_verifies_the_artifact_receipt(self) -> None:
        output = self.root / "process-output"
        config_path = self.root / "process-config.json"
        config_path.write_bytes(canonical_json(asdict(self.config)))
        completed = subprocess.run([sys.executable, "-m", "experiments.catalog_ranker_train",
                                    "--catalog", str(self.catalog), "--output", str(output),
                                    "--config", str(config_path)], capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads((output / "report.json").read_text())
        model = json.loads((output / "model.json").read_text())
        self.assertEqual(report["model_sha256"], sha256_file(output / "model.json"))
        self.assertEqual(report["config_sha256"], model["config_sha256"])
        self.assertEqual(report["row_overlap"], 0)
        self.assertGreater(report["validation"]["pairs"], 0)


if __name__ == "__main__":
    unittest.main()
