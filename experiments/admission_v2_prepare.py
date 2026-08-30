"""Freeze and benchmark the feature-equivalent precomputed admission v2 scorer."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from experiments.admission_train import _pool, _read_jsonl
from mercury.admission import (
    FEATURE_VERSION_V2,
    MODEL_SCHEMA_V2,
    AdmissionFeatureCache,
    AdmissionModel,
    admission_features,
    admission_features_v2,
    score_all_candidates,
)
from mercury.catalog import Catalog
from mercury.config import Config
from mercury.model_assets import file_sha256
from mercury.retrieval import SparseIndex


def freeze_model(source: Path, output: Path) -> dict:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != "mercury-admission-linear-v1" \
            or payload.get("feature_version") != "admission-features-v1":
        raise ValueError("Admission v2 requires the frozen v1 linear model")
    payload["schema"] = MODEL_SCHEMA_V2
    payload["feature_version"] = FEATURE_VERSION_V2
    payload["optimization"] = (
        "Feature-equivalent v1 coefficients with catalog-load-time immutable token, "
        "field-presence, metadata, and product-type features"
    )
    payload["parent_model_sha256"] = file_sha256(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def benchmark(catalog_path: Path, targets_path: Path, model_path: Path,
              sample_count: int = 80) -> dict:
    catalog = Catalog(catalog_path)
    model = AdmissionModel.load(model_path, catalog.sha256)
    cache_started = time.perf_counter()
    cache = AdmissionFeatureCache(catalog.products)
    cache_seconds = time.perf_counter() - cache_started
    config = Config.load(Path("configs/selected.json"))
    sparse = SparseIndex(catalog)
    control_times = []
    candidate_times = []
    processed_queries = 0
    exact_feature_parity = 0
    decision_parity = 0
    maximum_feature_delta = 0.0
    top30_overlaps = []
    control_ranks = []
    candidate_ranks = []
    try:
        for row in _read_jsonl(targets_path)[:sample_count]:
            processed_queries += 1
            product = catalog.by_id[row["ground_truth"]["parent_asin"]]
            query = " ".join(product.title.split()[:8])
            candidates, preferences, plan = _pool(catalog, sparse, query, config)
            started = time.perf_counter()
            control_features = admission_features(candidates, preferences, plan)
            control_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            candidate_features = admission_features_v2(candidates, preferences, plan, cache)
            candidate_times.append(time.perf_counter() - started)
            if control_features == candidate_features:
                exact_feature_parity += 1
            maximum_feature_delta = max(
                maximum_feature_delta,
                *(abs(left[name] - right[name])
                  for left, right in zip(control_features, candidate_features, strict=True)
                  for name in left),
            )
            control, _ = score_all_candidates(candidates, preferences, plan, "linear", model)
            candidate, _ = score_all_candidates(
                candidates, preferences, plan, "linear_v2", model, cache,
            )
            if [item.product.parent_asin for item in control] == [
                    item.product.parent_asin for item in candidate]:
                decision_parity += 1
            control_top30 = {item.product.parent_asin for item in control[:30]}
            candidate_top30 = {item.product.parent_asin for item in candidate[:30]}
            top30_overlaps.append(len(control_top30 & candidate_top30) / 30.0)
            target = row["ground_truth"]["parent_asin"]
            control_ranks.append(next(
                (rank for rank, item in enumerate(control, 1)
                 if item.product.parent_asin == target),
                None,
            ))
            candidate_ranks.append(next(
                (rank for rank, item in enumerate(candidate, 1)
                 if item.product.parent_asin == target),
                None,
            ))
    finally:
        sparse.close()

    def percentile(values: list[float], fraction: float) -> float:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))] if ordered else 0.0

    def rank_metrics(ranks: list[int | None]) -> dict[str, float]:
        total = len(ranks) or 1
        return {
            "recall_at_20": sum(rank is not None and rank <= 20 for rank in ranks) / total,
            "recall_at_30": sum(rank is not None and rank <= 30 for rank in ranks) / total,
            "mrr": sum(1.0 / rank for rank in ranks if rank is not None) / total,
        }

    return {
        "schema": "mercury-admission-v2-benchmark-v1",
        "catalog_sha256": catalog.sha256,
        "model_sha256": file_sha256(model_path),
        "targets_sha256": file_sha256(targets_path),
        "sample_count": processed_queries,
        "exact_feature_parity": exact_feature_parity,
        "exact_decision_parity": decision_parity,
        "maximum_feature_delta": maximum_feature_delta,
        "mean_top30_membership_overlap": statistics.mean(top30_overlaps),
        "control_admission": rank_metrics(control_ranks),
        "candidate_admission": rank_metrics(candidate_ranks),
        "cache_build_seconds": cache_seconds,
        "control_feature_p50_seconds": statistics.median(control_times),
        "control_feature_p95_seconds": percentile(control_times, 0.95),
        "candidate_feature_p50_seconds": statistics.median(candidate_times),
        "candidate_feature_p95_seconds": percentile(candidate_times, 0.95),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze and benchmark admission v2")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--targets", type=Path,
                        default=Path("artifacts/robustness-matrix-v2/training.jsonl"))
    parser.add_argument("--source-model", type=Path, default=Path("models/admission_linear_v1.json"))
    parser.add_argument("--model", type=Path, default=Path("models/admission_linear_v2.json"))
    parser.add_argument("--sample-count", type=int, default=80)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    freeze_model(args.source_model, args.model)
    result = benchmark(args.catalog, args.targets, args.model, args.sample_count)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
