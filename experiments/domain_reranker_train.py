"""Train and validate one fixed product-domain MiniLM reranker candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from experiments.admission_train import _pool, _query_variants, _read_jsonl
from mercury.admission import AdmissionFeatureCache, AdmissionModel, score_all_candidates
from mercury.catalog import Catalog
from mercury.config import Config
from mercury.model_assets import file_sha256
from mercury.neural import MAX_LENGTH, document_text
from mercury.product_types import classify_product
from mercury.retrieval import SparseIndex, terms


SEED = 20260830
REVISION = "mercury-product-domain-minilm-v1-seed-20260830"


def _bucket(namespace: str, value: str, modulo: int = 5) -> int:
    digest = hashlib.sha256(f"{namespace}\0{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def _split(annotation: dict) -> str:
    if _bucket("category", annotation["category_group"]) == 0:
        return "category_holdout"
    if _bucket("template", annotation["dialogue_template_family"]) == 0:
        return "template_holdout"
    return "training"


def _missing_signature(product) -> tuple[str, ...]:
    return tuple(field for field in ("features", "details", "description", "store")
                 if not product.fields.get(field, "").strip())


def _hard_negatives(target, candidates, count: int = 4):
    target_terms = set(terms(target.title))
    target_type = classify_product(target)

    def priority(candidate):
        product = candidate.product
        overlap = len(target_terms & set(terms(product.title)))
        product_type = classify_product(product)
        role_confusion = int(
            product_type.role != target_type.role
            and bool({product_type.role, target_type.role} & {"accessory", "component"})
        )
        missing_match = int(_missing_signature(product) == _missing_signature(target))
        return (-role_confusion, -overlap, -missing_match, product.parent_asin)

    eligible = [candidate for candidate in candidates
                if candidate.product.parent_asin != target.parent_asin]
    return [candidate.product for candidate in sorted(eligible, key=priority)[:count]]


def _candidate_pool(catalog, sparse, cache, model, config, query: str):
    candidates, preferences, plan = _pool(catalog, sparse, query, config)
    ordered, _ = score_all_candidates(
        candidates, preferences, plan, "linear_v2", model, cache,
    )
    return ordered[:30]


def build_examples(catalog, targets: list[dict], annotations: list[dict], limit: int):
    notes = {row["sample_id"]: row for row in annotations}
    config = Config.load("configs/admission_linear_v2.json")
    sparse = SparseIndex(catalog)
    cache = AdmissionFeatureCache(catalog.products)
    model = AdmissionModel.load(config.admission_model_path, catalog.sha256)
    examples = []
    validation: dict[str, list[dict]] = defaultdict(list)
    skipped = defaultdict(int)
    try:
        for index, row in enumerate(targets):
            target = catalog.by_id[row["ground_truth"]["parent_asin"]]
            annotation = notes[row["sample_id"]]
            split = _split(annotation)
            variants = _query_variants(target, index)
            if not variants:
                skipped["empty_query"] += 1
                continue
            query = variants[0]
            candidates = _candidate_pool(catalog, sparse, cache, model, config, query)
            identifiers = {candidate.product.parent_asin for candidate in candidates}
            if target.parent_asin not in identifiers:
                skipped[f"{split}_admission_miss"] += 1
                continue
            record = {
                "sample_id": row["sample_id"],
                "query": query,
                "target": target.parent_asin,
                "candidate_ids": [candidate.product.parent_asin for candidate in candidates],
                "group": annotation["category_group"] if split == "category_holdout"
                else annotation["dialogue_template_family"],
            }
            if split != "training":
                if len(validation[split]) < limit:
                    validation[split].append(record)
                continue
            examples.append((query, document_text(target), 1.0, row["sample_id"], "positive"))
            for negative in _hard_negatives(target, candidates):
                examples.append((query, document_text(negative), 0.0,
                                 row["sample_id"], "hard_negative"))
    finally:
        sparse.close()
    return examples, dict(validation), dict(skipped)


def _rank(model, catalog, records: list[dict]) -> dict:
    ranks = []
    for row in records:
        products = [catalog.by_id[identifier] for identifier in row["candidate_ids"]]
        pairs = [(row["query"], document_text(product)) for product in products]
        scores = model.predict(pairs, batch_size=30, show_progress_bar=False)
        order = sorted(range(len(products)), key=lambda index: (-float(scores[index]), index))
        rank = next(position for position, index in enumerate(order, 1)
                    if products[index].parent_asin == row["target"])
        ranks.append(rank)
    total = len(ranks) or 1
    return {
        "queries": len(ranks),
        "conditional_mrr": sum(1.0 / rank for rank in ranks) / total,
        "top_10_recall": sum(rank <= 10 for rank in ranks) / total,
        "top_1": sum(rank == 1 for rank in ranks) / total,
    }


def _manifest(output: Path, source_model: Path, sources: dict) -> dict:
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and ".cache" not in path.parts and path.name != "asset_manifest.json":
            files[str(path.relative_to(output))] = file_sha256(path)
    payload = {
        "schema": "mercury-model-asset-v1",
        "repo_id": "local/product-domain-ms-marco-MiniLM-L6-v2",
        "revision": REVISION,
        "base_model_sha256": file_sha256(source_model / "model.safetensors"),
        "license": "Apache-2.0",
        "files": files,
        "training": {
            "seed": SEED,
            "epochs": 1,
            "learning_rate": 1e-5,
            "maximum_sequence_length": MAX_LENGTH,
            "negative_strategy": "four catalog hard negatives per positive",
            "source_sha256": sources,
        },
    }
    (output / "asset_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return payload


def train(catalog_path: Path, targets_path: Path, annotations_path: Path,
          base_model: Path, output: Path, validation_limit: int) -> dict:
    import numpy as np
    import torch
    from sentence_transformers import CrossEncoder

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    catalog = Catalog(catalog_path)
    examples, validation, skipped = build_examples(
        catalog, _read_jsonl(targets_path), _read_jsonl(annotations_path), validation_limit,
    )
    if not examples or any(not rows for rows in validation.values()):
        raise ValueError("Domain reranker requires training and two validation splits")
    common = {
        "max_length": MAX_LENGTH,
        "device": "cpu",
        "local_files_only": True,
        "trust_remote_code": False,
        "model_kwargs": {"use_safetensors": True, "dtype": torch.float32},
    }
    baseline = CrossEncoder(str(base_model), **common)
    control = {name: _rank(baseline, catalog, rows) for name, rows in validation.items()}
    candidate = CrossEncoder(str(base_model), **common)
    if output.exists():
        shutil.rmtree(output)
    optimizer = torch.optim.AdamW(candidate.model.parameters(), lr=1e-5, weight_decay=0.01)
    candidate.model.train()
    order = list(range(len(examples)))
    random.Random(SEED).shuffle(order)
    for start in range(0, len(order), 16):
        batch = [examples[index] for index in order[start:start + 16]]
        encoded = candidate.tokenizer(
            [row[0] for row in batch], [row[1] for row in batch], padding=True,
            truncation=True, max_length=MAX_LENGTH, return_tensors="pt",
        )
        labels = torch.tensor([row[2] for row in batch], dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        logits = candidate.model(**encoded).logits.reshape(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(candidate.model.parameters(), 1.0)
        optimizer.step()
    candidate.save_pretrained(str(output), safe_serialization=True)
    candidate = CrossEncoder(str(output), **common)
    adapted = {name: _rank(candidate, catalog, rows) for name, rows in validation.items()}
    sources = {
        "catalog": file_sha256(catalog_path),
        "targets": file_sha256(targets_path),
        "annotations": file_sha256(annotations_path),
    }
    manifest = _manifest(output, base_model, sources)
    return {
        "schema": "mercury-domain-reranker-training-v1",
        "training_pairs": len(examples),
        "training_queries": len({row[3] for row in examples}),
        "label_counts": {
            "positive": sum(row[2] == 1 for row in examples),
            "negative": sum(row[2] == 0 for row in examples),
        },
        "validation": {
            name: {"control": control[name], "candidate": adapted[name]}
            for name in sorted(validation)
        },
        "validation_group_counts": {name: len({row["group"] for row in rows})
                                    for name, rows in validation.items()},
        "skipped": skipped,
        "weights_sha256": manifest["files"]["model.safetensors"],
        "revision": REVISION,
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one frozen product-domain reranker")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--targets", type=Path,
                        default=Path("artifacts/robustness-matrix-v2/training.jsonl"))
    parser.add_argument("--annotations", type=Path,
                        default=Path("artifacts/robustness-matrix-v2/training-annotations.jsonl"))
    parser.add_argument("--base-model", type=Path, default=Path("artifacts/models/reranker"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/reranker_domain_v1"))
    parser.add_argument("--validation-limit", type=int, default=40)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = train(args.catalog, args.targets, args.annotations, args.base_model,
                   args.model, args.validation_limit)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
