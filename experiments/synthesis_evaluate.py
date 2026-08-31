"""Matched conversational search experiments with immutable local receipts."""
from __future__ import annotations

import argparse
import json
import math
import socket
import time
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from experiments.run import ObservedAgent, peak_rss_bytes, source_hashes
from mercury.catalog import product_from_dict
from mercury.lexical import Agent, FULL_WIDTH_CONFIG
from mercury.lexical.config import DEFAULT_AGENT_CONFIG
from mercury.lexical.question_planner import AdaptiveQuestionPlanner, QuestionPlan
from mercury.lexical.retrieval import SearchResult
from mercury.model_assets import file_sha256
from mercury.neural import NeuralRanker
from mercury.types import Candidate


EVALUATOR_SHA256 = "79a5ea06f9a1b8c5036f30efa85dc1f36b8f6b06eb8feb8f545dfa767bc45564"


@dataclass(frozen=True)
class ExperimentConfig:
    question_policy: str = "adaptive"
    ranking_policy: str = "lexical"
    full_width: bool = False
    neural_prefix: int = 30

    def __post_init__(self) -> None:
        if self.question_policy not in {"adaptive", "open"}:
            raise ValueError("Unknown question policy")
        if self.ranking_policy not in {"lexical", "semantic_ties", "semantic_constraints"}:
            raise ValueError("Unknown ranking policy")
        if type(self.neural_prefix) is not int or not 1 <= self.neural_prefix <= 100:
            raise ValueError("Neural prefix must be between 1 and 100")


class OpenQuestionPlanner(AdaptiveQuestionPlanner):
    """Compare open clarification with facet selection on identical candidates."""

    def choose(self, state, candidates: list[dict], turn: int) -> QuestionPlan:
        if turn >= 10 or not candidates:
            return self._no_question()
        if "other" in state.no_preference_attributes:
            return super().choose(state, candidates, turn)
        facets = self._score_facets(candidates)
        available = [facet for facet in facets
                     if facet.attribute not in state.no_preference_attributes]
        information_gain = max((facet.information_gain for facet in available), default=0.0)
        answerability = self._answerability("other", state)
        state.record_question("other")
        return QuestionPlan("other", self._word_question("other", ()), information_gain,
                            answerability, information_gain * answerability)


def evidence_tier(product: dict, preserve_phrase_order: bool) -> tuple:
    exact = int(product.get("_hard_constraint_exact_count") or 0)
    count = int(product.get("_hard_constraint_count") or 0)
    tier = (bool(product.get("_exact_constraint_index_match")),
            count > 0 and exact == count, exact, bool(product.get("_category_leaf_match")))
    if preserve_phrase_order:
        tier += (bool(product.get("_constraint_sequence_match")),
                 tuple(product.get("_catalog_tiebreak") or (0.0, 0.0, 0)))
    return tier


class SearchExperiment:
    """Rerank only the bounded prefix without changing membership or lower tiers."""

    def __init__(self, inner, config: ExperimentConfig, ranker=None):
        self.inner = inner
        self.config = config
        self.ranker = ranker
        self.last_diagnostics: dict = {}

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def search_with_context(self, state, limit: int = 10) -> SearchResult:
        result = self.inner.search_with_context(state, limit)
        candidates = result.candidates
        ids = [str(item["parent_asin"]) for item in candidates]
        diagnostic = {"candidate_context_ids": ids, "ranked_context_ids": ids,
                      "neural_prefix_ids": [], "fallbacks": [], "neural_pairs": 0}
        self.last_diagnostics = diagnostic
        query = state.semantic_query()
        if self.config.ranking_policy == "lexical" or not query or not candidates:
            return result
        if self.ranker is None:
            diagnostic["fallbacks"].append("optional_ranker_unavailable")
            return result
        prefix = candidates[:self.config.neural_prefix]
        prefix_ids = [str(item["parent_asin"]) for item in prefix]
        diagnostic["neural_prefix_ids"] = prefix_ids
        prompt_before = self.ranker.prompt_tokens
        try:
            pool = [Candidate(product_from_dict(product), float(product["_rank_score"]))
                    for product in prefix]
            logits = self.ranker.score(query, pool, document_mode="lexical")
            if set(logits) != set(prefix_ids) or any(
                    type(value) not in (int, float) or not math.isfinite(value)
                    for value in logits.values()):
                raise ValueError("Incomplete or non-finite candidate scores")
        except Exception as error:
            diagnostic["fallbacks"].append(type(error).__name__)
            return result
        diagnostic["neural_pairs"] = len(prefix)
        preserve = self.config.ranking_policy == "semantic_ties"
        ordered = sorted(prefix, key=lambda product: (
            evidence_tier(product, preserve), logits[str(product["parent_asin"])],
            -prefix_ids.index(str(product["parent_asin"])),
        ), reverse=True) + candidates[len(prefix):]
        ordered_ids = [str(product["parent_asin"]) for product in ordered]
        if len(ordered_ids) != len(set(ordered_ids)) or set(ordered_ids) != set(ids):
            raise RuntimeError("Reranking changed candidate membership")
        diagnostic["ranked_context_ids"] = ordered_ids
        diagnostic["ordering_changed"] = ordered_ids != ids
        # Keep lexical scores unchanged so a ranking ablation does not also
        # replace the numeric confidence inputs used by the breadth policy.
        return SearchResult(
            [(str(product["parent_asin"]), float(product["_rank_score"]))
             for product in ordered[:limit]], ordered,
            result.prompt_tokens + max(0, self.ranker.prompt_tokens - prompt_before),
            result.ranking_mode,
        )


class SynthesisAgent(Agent):
    def __init__(self, catalog_path: str | Path, experiment: ExperimentConfig,
                 artifact_dir: Path | None = None, ranker=None) -> None:
        super().__init__(catalog_path, config=FULL_WIDTH_CONFIG if experiment.full_width
                         else DEFAULT_AGENT_CONFIG)
        self.experiment = experiment
        self.model_error: str | None = None
        if experiment.ranking_policy != "lexical" and ranker is None:
            try:
                if artifact_dir is None:
                    raise ValueError("No local model directory configured")
                ranker = NeuralRanker(artifact_dir, cache_capacity=4096)
            except Exception as error:
                self.model_error = type(error).__name__
        self.search = SearchExperiment(self.search, experiment, ranker)
        if experiment.question_policy == "open":
            self.question_planner = OpenQuestionPlanner(self.search.feature_store)
        self.last_diagnostics: dict = {}

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        response = super().respond(session_id, user_message, turn, top_k)
        diagnostic = deepcopy(self.search.last_diagnostics)
        diagnostic.update({"experiment": asdict(self.experiment), "model_error": self.model_error,
                           "returned_ids": [item["parent_asin"]
                                            for item in response["recommendations"]]})
        self.last_diagnostics = diagnostic
        if self.experiment.full_width:
            expected = diagnostic["ranked_context_ids"][:min(10, top_k)]
            if diagnostic["returned_ids"] != expected:
                raise AssertionError("Full-width control changed the raw ranked prefix")
        return response


def _write(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def run(catalog: Path, dataset: Path, output: Path, config: ExperimentConfig,
        artifact_dir: Path | None = None) -> dict:
    evaluator = Path("evaluator/local_evaluator.py")
    if file_sha256(evaluator) != EVALUATOR_SHA256:
        raise ValueError("Official evaluator identity does not match")
    output.mkdir(parents=True, exist_ok=False)
    registration = {"config": asdict(config), "runtime": source_hashes(),
                    "catalog_sha256": file_sha256(catalog),
                    "dataset_sha256": file_sha256(dataset),
                    "evaluator_sha256": EVALUATOR_SHA256,
                    "hypothesis": "Separate question selection and semantic ordering from breadth",
                    "scope": "Full public set; no claim of unseen generalization",
                    "network": "Python socket connection denial"}
    _write(output / "registration.json", registration)
    started = time.perf_counter()
    inner = SynthesisAgent(catalog, config, artifact_dir)
    cold = time.perf_counter() - started
    observed = ObservedAgent(inner)
    ids, categories, products = catalog_index(catalog)
    started = time.perf_counter()
    try:
        report = evaluate(observed, load_jsonl(dataset), ids, categories, products)
        elapsed = time.perf_counter() - started
        traces = [turn for session in observed.traces for turn in session]
        latencies = sorted(turn["latency_seconds"] for turn in traces)
        report["measurement"] = {
            "config": asdict(config), "cold_start_seconds": cold, "evaluation_seconds": elapsed,
            "p50_seconds": latencies[int((len(latencies) - 1) * 0.50)],
            "p95_seconds": latencies[int((len(latencies) - 1) * 0.95)],
            "peak_rss_bytes": peak_rss_bytes(),
            "errors": sum("error" in turn for turn in traces),
            "fallback_turns": sum(bool(turn.get("diagnostics", {}).get("fallbacks"))
                                  for turn in traces),
            "widths": dict(Counter(len(turn.get("response", {}).get("recommendations", []))
                                   for turn in traces)),
            "source_changed": source_hashes() != registration["runtime"],
            "catalog_changed": file_sha256(catalog) != registration["catalog_sha256"],
            "dataset_changed": file_sha256(dataset) != registration["dataset_sha256"],
            "model_available": inner.search.ranker is not None,
            "model_error": inner.model_error,
            "model_identity": getattr(inner.search.ranker, "asset_identity", None),
        }
        _write(output / "report.json", report)
        _write(output / "traces.json", observed.traces)
        measurement = report["measurement"]
        if any(measurement[key] for key in ("errors", "source_changed", "catalog_changed", "dataset_changed")):
            raise RuntimeError("Experiment failed integrity or runtime checks; inspect receipt")
        return report
    finally:
        inner.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()

    def deny_network(*args, **kwargs):
        raise RuntimeError("Network connection disabled for this local experiment")

    socket.create_connection = deny_network
    socket.socket.connect = deny_network
    config = ExperimentConfig(**json.loads(args.config.read_text()))
    result = run(args.catalog, args.dataset, args.output, config, args.artifacts)
    print({key: result[key] for key in ("recommended_technical_score", "hit_rate_at_10", "mrr", "mttc")})


if __name__ == "__main__":
    main()
