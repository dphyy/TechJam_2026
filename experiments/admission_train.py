"""Train and evaluate the frozen lightweight all-pool admission model."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler

from mercury.admission import FEATURE_NAMES, FEATURE_VERSION, MODEL_SCHEMA, admission_features
from mercury.catalog import Catalog
from mercury.config import Config
from mercury.intent import IntentWeights, decide_intent
from mercury.model_assets import file_sha256
from mercury.planning import build_retrieval_plan
from mercury.retrieval import SparseIndex, fuse_routes, terms
from mercury.state import SessionState
from mercury.types import Candidate


SEED = 20260830


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _digest_rows(rows: list[dict]) -> str:
    payload = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def _query_variants(product, index: int) -> list[str]:
    categories = terms(product.fields.get("categories", ""))[-3:]
    title = terms(product.title)
    features = terms(product.fields.get("features", ""))
    facets = [value for values in product.facets.values() for value in values]
    variants = [
        " ".join(dict.fromkeys(categories + title[-4:])),
        " ".join(dict.fromkeys(categories + terms(" ".join(facets))[:3])),
        " ".join(dict.fromkeys(categories + features[:4])),
    ]
    rotated = variants[index % len(variants):] + variants[:index % len(variants)]
    return [value for value in dict.fromkeys(rotated) if value]


def _intent_weights(config: Config) -> IntentWeights:
    return IntentWeights(
        object=config.intent_object_weight,
        slots=config.intent_slot_weight,
        hard=config.intent_hard_weight,
        buying_language=config.intent_buying_language_weight,
        browsing_language=config.intent_browsing_language_weight,
        use_case_without_object=config.intent_use_case_weight,
        unresolved=config.intent_unresolved_weight,
        sparse_request=config.intent_sparse_request_weight,
    )


def _pool(catalog: Catalog, sparse: SparseIndex, query: str, config: Config) -> tuple[list[Candidate], list, object]:
    state = SessionState({}, config.state_mode, config.alternatives_mode, config.scoped_preferences)
    state.update(query, 1)
    intent = decide_intent(
        state, query, config.router_buying_threshold, config.router_browsing_threshold,
        config.router_over_general_threshold, _intent_weights(config),
    )
    plan = build_retrieval_plan(state, intent)
    routes = {"sparse": sparse.search(plan.lexical_query or query, config.sparse_limit)}
    fused = fuse_routes(routes, {"sparse": 1.0})
    candidates = [
        Candidate(catalog.by_id[identifier], score, parts)
        for identifier, score, parts in fused[:config.candidate_limit]
    ]
    return candidates, state.active_preferences(), plan


def build_records(
    catalog: Catalog, targets: list[dict], annotations: list[dict], config: Config,
) -> tuple[list[dict], dict]:
    notes = {row["sample_id"]: row for row in annotations}
    sparse = SparseIndex(catalog)
    records = []
    retrieval_misses = 0
    try:
        for index, row in enumerate(targets):
            target = row["ground_truth"]["parent_asin"]
            product = catalog.by_id[target]
            annotation = notes[row["sample_id"]]
            for variant, query in enumerate(_query_variants(product, index)):
                candidates, preferences, plan = _pool(catalog, sparse, query, config)
                identifiers = [candidate.product.parent_asin for candidate in candidates]
                if target not in identifiers:
                    retrieval_misses += 1
                    continue
                features = admission_features(candidates, preferences, plan)
                group = annotation["loose_title_family_sha256"]
                fold = int(hashlib.sha256(f"admission-validation\0{group}".encode()).hexdigest(), 16) % 4
                for rank, (candidate, vector) in enumerate(zip(candidates, features, strict=True), 1):
                    records.append({
                        "sample_id": row["sample_id"],
                        "query_variant": variant,
                        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                        "candidate_id": candidate.product.parent_asin,
                        "target": int(candidate.product.parent_asin == target),
                        "bm25_rank": rank,
                        "group": group,
                        "fold": "validation" if fold == 0 else "training",
                        "metadata_strata": annotation["metadata_strata"],
                        "features": vector,
                    })
    finally:
        sparse.close()
    return records, {"retrieval_misses": retrieval_misses}


def _arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([[row["features"][name] for name in FEATURE_NAMES] for row in rows], dtype=np.float64),
        np.asarray([row["target"] for row in rows], dtype=np.int64),
    )


def _rank_metrics(rows: list[dict], scores: np.ndarray, prefix: str) -> dict:
    grouped: dict[tuple[str, int], list[tuple[dict, float]]] = defaultdict(list)
    for row, score in zip(rows, scores, strict=True):
        grouped[(row["sample_id"], row["query_variant"])].append((row, float(score)))
    ranks = []
    slices: dict[str, list[int]] = defaultdict(list)
    for values in grouped.values():
        ordered = sorted(values, key=lambda item: (-item[1], item[0]["bm25_rank"], item[0]["candidate_id"]))
        target_row = next(row for row, _ in values if row["target"] == 1)
        rank = next(index for index, (row, _) in enumerate(ordered, 1) if row["target"] == 1)
        ranks.append(rank)
        for stratum in target_row["metadata_strata"]:
            slices[stratum].append(rank)
    return {
        "name": prefix,
        "queries": len(ranks),
        "recall_at_20": sum(rank <= 20 for rank in ranks) / len(ranks) if ranks else 0.0,
        "recall_at_30": sum(rank <= 30 for rank in ranks) / len(ranks) if ranks else 0.0,
        "conditional_mrr": sum(1.0 / rank for rank in ranks) / len(ranks) if ranks else 0.0,
        "slices": {
            name: {
                "queries": len(values),
                "recall_at_20": sum(rank <= 20 for rank in values) / len(values),
                "recall_at_30": sum(rank <= 30 for rank in values) / len(values),
            }
            for name, values in sorted(slices.items())
        },
    }


def train(
    catalog_path: Path, targets_path: Path, annotations_path: Path, output_model: Path,
) -> dict:
    catalog = Catalog(catalog_path)
    config = Config.load(Path("configs/selected.json"))
    records, build_diagnostics = build_records(
        catalog, _read_jsonl(targets_path), _read_jsonl(annotations_path), config,
    )
    training = [row for row in records if row["fold"] == "training"]
    validation = [row for row in records if row["fold"] == "validation"]
    if not training or not validation or not any(row["target"] for row in training):
        raise ValueError("Grouped admission train/validation evidence is incomplete")
    x_train, y_train = _arrays(training)
    x_validation, y_validation = _arrays(validation)
    scaler = StandardScaler().fit(x_train)
    classifier = LogisticRegression(
        C=0.1, class_weight="balanced", max_iter=1000, random_state=SEED, solver="liblinear",
    ).fit(scaler.transform(x_train), y_train)
    probabilities = classifier.predict_proba(scaler.transform(x_validation))[:, 1]
    linear_scores = classifier.decision_function(scaler.transform(x_validation))
    bm25_scores = np.asarray([-row["bm25_rank"] for row in validation], dtype=np.float64)
    fusion_scores = np.asarray([
        sum(row["features"][name] * weight for name, weight in {
            "bm25_normalized": 0.38, "rank_reciprocal": 0.12, "route_agreement": 0.05,
            "title_coverage": 0.12, "category_coverage": 0.10, "feature_coverage": 0.08,
            "detail_coverage": 0.03, "description_coverage": 0.02, "all_field_coverage": 0.05,
            "object_agreement": 0.12, "positive_coverage": 0.14, "negative_compatibility": 0.18,
            "hard_compatibility": 0.20, "metadata_completeness": 0.02, "price_compatibility": 0.04,
        }.items())
        for row in validation
    ], dtype=np.float64)
    payload = {
        "schema": MODEL_SCHEMA,
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficients": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "catalog_sha256": catalog.sha256,
        "training_sha256": _digest_rows(training),
        "validation_sha256": _digest_rows(validation),
        "source_sha256": {
            "catalog": file_sha256(catalog_path),
            "targets": file_sha256(targets_path),
            "annotations": file_sha256(annotations_path),
        },
        "seed": SEED,
        "algorithm": "StandardScaler plus L2 logistic regression, C=0.1, balanced classes, liblinear",
    }
    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_model.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    started = time.perf_counter()
    # Measure pure Python inference over the full validation matrix.
    from mercury.admission import AdmissionModel
    loaded = AdmissionModel.load(output_model, catalog.sha256)
    for row in validation:
        loaded.score(row["features"])
    scoring_seconds = time.perf_counter() - started
    return {
        "schema": "mercury-admission-training-result-v1",
        "model_sha256": file_sha256(output_model),
        "training_rows": len(training),
        "validation_rows": len(validation),
        "training_queries": len({(row["sample_id"], row["query_variant"]) for row in training}),
        "validation_queries": len({(row["sample_id"], row["query_variant"]) for row in validation}),
        **build_diagnostics,
        "validation": [
            _rank_metrics(validation, bm25_scores, "bm25_prefix"),
            _rank_metrics(validation, fusion_scores, "deterministic_fusion"),
            _rank_metrics(validation, linear_scores, "regularized_linear"),
        ],
        "brier_score": float(brier_score_loss(y_validation, probabilities)),
        "full_validation_scoring_seconds": scoring_seconds,
        "microseconds_per_candidate": scoring_seconds / len(validation) * 1e6,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train frozen all-pool admission scoring")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--targets", type=Path, default=Path("artifacts/robustness-matrix-v1/training.jsonl"))
    parser.add_argument("--annotations", type=Path, default=Path("artifacts/robustness-matrix-v1/training-annotations.jsonl"))
    parser.add_argument("--model", type=Path, default=Path("models/admission_linear_v1.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = train(args.catalog, args.targets, args.annotations, args.model)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
