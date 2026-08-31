"""Deterministic pairwise calibration using only explicit catalog properties."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from mercury.lexical.catalog_ranker import (
    FEATURE_NAMES,
    FEATURE_SCALES,
    FEATURE_VERSION,
    MAX_WEIGHT,
    canonical_json,
    feature_vector,
    product_features,
    sha256_file,
    text_field,
)
from mercury.lexical.dialogue import Evidence, SessionState
from mercury.lexical.product_features import (
    FACET_PATTERNS,
    ProductFeatureStore,
    ProductFeatures,
    affirmed_terms,
    component_scope,
    terms,
)


@dataclass(frozen=True)
class TrainingConfig:
    split_seed: str = "catalog-row-split-v1"
    validation_fraction: float = 0.2
    max_train_rows: int = 1200
    max_validation_rows: int = 300
    max_queries_per_split: int = 256
    max_pairs_per_split: int = 2048
    positives_per_query: int = 2
    negatives_per_query: int = 4
    epochs: int = 25
    learning_rate: float = 0.1
    l2: float = 0.01

    def __post_init__(self) -> None:
        if not isinstance(self.split_seed, str) or not self.split_seed:
            raise ValueError("A nonempty split seed is required")
        if (type(self.validation_fraction) not in (int, float) or not math.isfinite(self.validation_fraction)
                or not 0 < self.validation_fraction < 1):
            raise ValueError("validation_fraction must lie between zero and one")
        bounds = {"max_train_rows": 5000, "max_validation_rows": 2000, "max_queries_per_split": 2000,
                  "max_pairs_per_split": 10000, "positives_per_query": 8, "negatives_per_query": 16, "epochs": 100}
        for name, maximum in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be between 1 and {maximum}")
        for name in ("learning_rate", "l2"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and between zero and one")
        if self.learning_rate == 0:
            raise ValueError("learning_rate must be positive")


DEFAULT_CONFIG = TrainingConfig()


def _digest(seed: str, namespace: str, value: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{namespace}\0{value}".encode()).digest(), "big")


def partition(identifier: str, config: TrainingConfig) -> str:
    return "validation" if _digest(config.split_seed, "partition", identifier) / 2**256 < config.validation_fraction else "train"


@dataclass(frozen=True, order=True)
class Fact:
    attribute: str
    owner: str
    values: tuple[str, ...]

    def evidence(self) -> Evidence:
        value = " and ".join(self.values)
        return Evidence(f"{self.owner}: {value}" if self.owner else value, 1.0, "clarification", 1, self.attribute)


def catalog_facts(row: dict) -> tuple[Fact, ...]:
    """Only structured, affirmative color/material statements supply supervision."""
    details = row.get("details")
    if not isinstance(details, dict):
        return ()
    facts: dict[tuple[str, str], set[str]] = defaultdict(set)
    for key, raw in details.items():
        label, value = str(key).casefold(), text_field(raw)
        attribute = ("material" if re.search(r"\b(material|fabric|composition|textile)\b", label)
                     else "color" if re.search(r"\b(colou?r)\b", label) else None)
        if attribute is None:
            continue
        affirmed = " ".join(affirmed_terms(value))
        values = set(FACET_PATTERNS[attribute].findall(affirmed))
        if values:
            owner = component_scope(f"{label}: {value}") or ""
            facts[(attribute, owner)].update(values)
    return tuple(sorted(Fact(attribute, owner, tuple(sorted(values))) for (attribute, owner), values in facts.items()))[:8]


def category_key(row: dict) -> str:
    raw = row.get("categories")
    nodes = raw if isinstance(raw, list) else [raw]
    leaves = [part for node in nodes for part in re.split(r"[,>/|]", text_field(node)) if terms(part)]
    return " ".join(terms(leaves[-1])) if leaves else ""


@dataclass(frozen=True)
class CatalogRow:
    identifier: str
    category: str
    facts: tuple[Fact, ...]
    features: ProductFeatures


def load_partitions(catalog: Path, config: TrainingConfig) -> tuple[dict[str, list[CatalogRow]], dict]:
    heaps: dict[str, list[tuple[int, str, dict]]] = {"train": [], "validation": []}
    limits = {"train": config.max_train_rows, "validation": config.max_validation_rows}
    digest, seen, counts = hashlib.sha256(), set(), Counter()
    with catalog.open("rb") as handle:
        for line in handle:
            digest.update(line)
            row = json.loads(line)
            identifier = row.get("parent_asin") if isinstance(row, dict) else None
            if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
                raise ValueError("Catalog identifiers must be unique nonempty strings")
            seen.add(identifier)
            split = partition(identifier, config)
            counts[split] += 1
            priority = _digest(config.split_seed, "selection", identifier)
            entry = (-priority, identifier, row)
            heap = heaps[split]
            if len(heap) < limits[split]:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)
    splits = {
        split: [CatalogRow(identifier, category_key(row), catalog_facts(row), product_features(row))
                for _, identifier, row in sorted(heap, key=lambda entry: (-entry[0], entry[1]))]
        for split, heap in heaps.items()
    }
    identities = {name: {row.identifier for row in rows} for name, rows in splits.items()}
    if identities["train"] & identities["validation"]:
        raise RuntimeError("Catalog partitions overlap")
    receipt = {
        "catalog_sha256": digest.hexdigest(), "catalog_rows": len(seen),
        "partition_rows": dict(counts), "selected_rows": {name: len(rows) for name, rows in splits.items()},
        "selected_row_sha256": {name: hashlib.sha256(canonical_json(sorted(ids))).hexdigest()
                                for name, ids in identities.items()},
        "row_overlap": 0,
    }
    return splits, receipt


@dataclass(frozen=True, order=True)
class QuerySpec:
    category: str
    facts: tuple[Fact, ...]

    def key(self) -> str:
        return canonical_json(asdict(self)).decode()


def supervision(row: CatalogRow, query: QuerySpec) -> int:
    actual = {(fact.attribute, fact.owner): set(fact.values) for fact in row.facts}
    complete = True
    for fact in query.facts:
        observed = actual.get((fact.attribute, fact.owner))
        if observed and set(fact.values).isdisjoint(observed):
            return -1
        complete &= observed is not None and set(fact.values) <= observed
    return 1 if complete else 0


@dataclass(frozen=True)
class TrainingPair:
    positive_id: str
    negative_id: str
    positive: tuple[float, ...]
    negative: tuple[float, ...]


def generate_pairs(rows: list[CatalogRow], config: TrainingConfig) -> tuple[list[TrainingPair], dict]:
    buckets: dict[str, list[CatalogRow]] = defaultdict(list)
    queries = set()
    for row in rows:
        if not row.category:
            continue
        buckets[row.category].append(row)
        # A multivalue catalog property can describe alternatives or a blend.
        # Query each asserted value separately rather than inventing a joint
        # shopper requirement that the catalog does not establish.
        single_facts = [Fact(fact.attribute, fact.owner, (value,)) for fact in row.facts for value in fact.values][:16]
        for fact in single_facts:
            queries.add(QuerySpec(row.category, (fact,)))
        for first, second in zip(single_facts, single_facts[1:]):
            if first.attribute != second.attribute or first.owner != second.owner:
                queries.add(QuerySpec(row.category, (first, second)))
    ordered = sorted(queries, key=lambda query: (_digest(config.split_seed, "query", query.key()), query.key()))[:config.max_queries_per_split]
    pairs, counts = [], Counter()
    compiler = ProductFeatureStore(max_size=1)
    for query in ordered:
        positive, negative = [], []
        for row in buckets[query.category]:
            label = supervision(row, query)
            if label == 1:
                positive.append(row)
            elif label == -1:
                negative.append(row)
            else:
                counts["unknown_candidates_skipped"] += 1
        if len(positive) > 1:
            counts["queries_with_multiple_positives"] += 1
        if not positive or not negative:
            continue
        key = query.key()
        def order(row: CatalogRow) -> tuple[int, str]:
            return _digest(config.split_seed, key, row.identifier), row.identifier
        positive = sorted(positive, key=order)[:config.positives_per_query]
        negative = sorted(negative, key=order)[:config.negatives_per_query]
        state = SessionState({}, evidence=[fact.evidence() for fact in query.facts], category_text=query.category)
        compiled = compiler.compile_query(state.evidence, {})
        features = {row.identifier: feature_vector(row.features, compiled, query.category) for row in positive + negative}
        for matched in positive:
            for contradicted in negative:
                first, second = features[matched.identifier], features[contradicted.identifier]
                if first == second:
                    counts["indistinguishable_pairs_skipped"] += 1
                    continue
                pairs.append(TrainingPair(matched.identifier, contradicted.identifier, first, second))
                if len(pairs) >= config.max_pairs_per_split:
                    return pairs, {"queries_considered": len(ordered), **dict(counts)}
    return pairs, {"queries_considered": len(ordered), **dict(counts)}


def fit_weights(pairs: list[TrainingPair], config: TrainingConfig) -> tuple[float, ...]:
    if not pairs:
        raise ValueError("No unambiguous training pairs are available")
    weights = [0.0] * len(FEATURE_NAMES)
    differences = [tuple(first - second for first, second in zip(pair.positive, pair.negative, strict=True)) for pair in pairs]
    for _ in range(config.epochs):
        gradient = [0.0] * len(weights)
        for difference in differences:
            margin = sum(weight * value for weight, value in zip(weights, difference, strict=True))
            wrong_probability = 1 / (1 + math.exp(max(-50.0, min(50.0, margin))))
            for index, value in enumerate(difference):
                gradient[index] -= wrong_probability * value
        weights = [max(-MAX_WEIGHT, min(MAX_WEIGHT, weight - config.learning_rate * (gradient[index] / len(pairs) + config.l2 * weight)))
                   for index, weight in enumerate(weights)]
    return tuple(weights)


def pair_metrics(pairs: list[TrainingPair], weights: tuple[float, ...]) -> dict:
    if not pairs:
        return {"pairs": 0, "pair_accuracy": None, "logistic_loss": None}
    margins = [sum(weight * (first - second) for weight, first, second in zip(weights, pair.positive, pair.negative, strict=True))
               for pair in pairs]
    return {"pairs": len(pairs), "pair_accuracy": sum(1 if margin > 0 else .5 if margin == 0 else 0 for margin in margins) / len(margins),
            "logistic_loss": sum(max(0, -margin) + math.log1p(math.exp(-abs(margin))) for margin in margins) / len(margins)}


def train(catalog: Path, config: TrainingConfig = DEFAULT_CONFIG) -> tuple[dict, dict]:
    partitions, receipt = load_partitions(catalog, config)
    train_pairs, train_stats = generate_pairs(partitions["train"], config)
    validation_pairs, validation_stats = generate_pairs(partitions["validation"], config)
    if not validation_pairs:
        raise ValueError("No disjoint validation pairs are available; do not train without validation")
    weights = fit_weights(train_pairs, config)
    definition = {"training": asdict(config), "features": FEATURE_VERSION,
                  "feature_names": list(FEATURE_NAMES), "scales": list(FEATURE_SCALES),
                  "supervision": "structured-facet-entailment-versus-explicit-contradiction-v1"}
    config_hash = hashlib.sha256(canonical_json(definition)).hexdigest()
    model = {"feature_names": list(FEATURE_NAMES), "scales": list(FEATURE_SCALES), "weights": list(weights),
             "catalog_sha256": receipt["catalog_sha256"], "config_sha256": config_hash}
    report = {**receipt, "config": definition, "config_sha256": config_hash,
              "model_sha256": hashlib.sha256(canonical_json(model)).hexdigest(),
              "train_generation": train_stats, "validation_generation": validation_stats,
              "train_before": pair_metrics(train_pairs, (0.0,) * len(FEATURE_NAMES)),
              "train_after": pair_metrics(train_pairs, weights), "validation": pair_metrics(validation_pairs, weights),
              "limits": ["Catalog property reconstruction is a proxy task, not shopper outcome validation",
                         "Only explicit structured material and color facts supply labels",
                         "Validation rows are disjoint; property vocabulary may overlap",
                         "Unknown and indistinguishable pairs are not assigned artificial unique truth"]}
    return model, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = TrainingConfig(**json.loads(args.config.read_text())) if args.config else DEFAULT_CONFIG
    # Record the fixed split and bounded settings before any pair generation.
    args.output.mkdir(parents=True, exist_ok=False)
    catalog_digest = sha256_file(args.catalog)
    (args.output / "registration.json").write_bytes(canonical_json({
        "config": asdict(config), "features": FEATURE_VERSION, "catalog_sha256": catalog_digest,
    }))
    model, report = train(args.catalog, config)
    if report["catalog_sha256"] != catalog_digest or sha256_file(args.catalog) != catalog_digest:
        raise RuntimeError("Catalog changed during training")
    (args.output / "model.json").write_bytes(canonical_json(model))
    (args.output / "report.json").write_bytes(canonical_json(report))
    print({"train_pairs": report["train_after"]["pairs"], "validation_pairs": report["validation"]["pairs"],
           "validation_pair_accuracy": report["validation"]["pair_accuracy"],
           "model_sha256": report["model_sha256"], "config_sha256": report["config_sha256"]})


if __name__ == "__main__":
    main()
