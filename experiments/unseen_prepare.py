"""Prepare target/user-disjoint synthetic sessions without running an agent.

The official evaluator deterministically materializes intent cards from product
metadata, so these rows exercise new catalog targets while retaining the official
dialogue protocol. Preparation is deliberately independent of Mercury rankings.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from evaluator.local_evaluator import load_jsonl
from mercury.model_assets import file_sha256


SCENARIO_WEIGHTS = (
    ("buying", 40),
    ("browsing", 40),
    ("intent_override", 15),
    ("boundary", 5),
)
PROFILE_TAGS = (
    ("comfort", "fit"),
    ("durability", "material"),
    ("style", "color"),
    ("value", "quality"),
)


def _stable_identifier(prefix: str, seed: int, index: int) -> str:
    digest = hashlib.sha256(f"{prefix}\0{seed}\0{index}".encode()).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _bucket(product: dict) -> str:
    values = product.get("categories") or []
    # The common root is "Clothing, Shoes & Jewelry" and must not make every
    # product look like shoes or jewelry. Bucket from the product-specific path.
    text = " ".join(str(value).lower() for value in values[1:])
    for name, markers in (
        ("shoes", ("shoe", "boot", "sandal", "slipper")),
        ("jewelry", ("jewelry", "ring", "necklace", "earring", "bracelet")),
        ("bags", ("bag", "backpack", "wallet", "purse")),
        ("accessories", ("accessor", "hat", "belt", "scarf", "glove", "watch")),
    ):
        if any(marker in text for marker in markers):
            return name
    return "clothing"


def _scenario_schedule(count: int) -> list[str]:
    if count < 20:
        raise ValueError("Each split needs at least 20 sessions to preserve the 40/40/15/5 mix")
    raw = [(name, count * weight / 100) for name, weight in SCENARIO_WEIGHTS]
    counts = {name: int(value) for name, value in raw}
    remainder = count - sum(counts.values())
    for name, value in sorted(raw, key=lambda item: item[1] - int(item[1]), reverse=True)[:remainder]:
        counts[name] += 1
    schedule = [name for name, _ in SCENARIO_WEIGHTS for _ in range(counts[name])]
    if len(schedule) != count:
        raise AssertionError("Scenario allocation failed")
    return schedule


def _eligible_products(catalog: Path, excluded_targets: set[str]) -> list[dict]:
    products = []
    seen = set()
    with catalog.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            identifier = str(product.get("parent_asin", "")).strip()
            if not identifier or identifier in seen or identifier in excluded_targets:
                continue
            seen.add(identifier)
            if not str(product.get("title") or "").strip():
                continue
            products.append(product)
    return products


def _stratified_selection(products: list[dict], count: int, rng: random.Random) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for product in products:
        groups[_bucket(product)].append(product)
    for values in groups.values():
        rng.shuffle(values)
    ordered = []
    names = sorted(groups)
    while len(ordered) < count and names:
        next_names = []
        for name in names:
            if groups[name]:
                ordered.append(groups[name].pop())
                next_names.append(name)
                if len(ordered) == count:
                    break
        names = next_names
    if len(ordered) != count:
        raise ValueError(f"Catalog has only {len(ordered)} eligible products for {count} requested sessions")
    return ordered


def _samples(products: list[dict], split: str, seed: int) -> list[dict]:
    schedule = _scenario_schedule(len(products))
    rng = random.Random(f"scenario\0{seed}\0{split}")
    rng.shuffle(schedule)
    samples = []
    for index, (product, scenario) in enumerate(zip(products, schedule, strict=True), 1):
        profile_tags = PROFILE_TAGS[(index - 1) % len(PROFILE_TAGS)]
        samples.append({
            "sample_id": _stable_identifier(f"unseen_{split}", seed, index),
            "user_group_id": _stable_identifier(f"synthetic_user_{split}", seed, index),
            "scenario_type": scenario,
            "category_bucket": _bucket(product),
            "difficulty_bucket": "unseen",
            "user_profile": {
                "purchase_frequency": "synthetic isolated profile",
                "average_prior_rating": None,
                "rating_style": "unknown",
                "preference_tags": list(profile_tags),
                "summary": "Synthetic profile created before inference for disjoint evaluation.",
            },
            "ground_truth": {"parent_asin": str(product["parent_asin"])},
        })
    return samples


def validate_splits(development: list[dict], final: list[dict], excluded_targets: set[str]) -> dict:
    if not development or not final:
        raise ValueError("Development and final splits must both be nonempty")
    for name, samples in (("development", development), ("final", final)):
        ids = [sample.get("sample_id") for sample in samples]
        users = [sample.get("user_group_id") for sample in samples]
        targets = [sample.get("ground_truth", {}).get("parent_asin") for sample in samples]
        if any(not isinstance(value, str) or not value for value in (*ids, *users, *targets)):
            raise ValueError(f"{name} contains missing identifiers")
        if len(ids) != len(set(ids)) or len(users) != len(set(users)) or len(targets) != len(set(targets)):
            raise ValueError(f"{name} IDs, users, and targets must each be unique")
        if set(targets) & excluded_targets:
            raise ValueError(f"{name} contains an excluded target")
        if Counter(sample.get("scenario_type") for sample in samples) != Counter(_scenario_schedule(len(samples))):
            raise ValueError(f"{name} does not preserve the declared scenario mix")
    development_targets = {sample["ground_truth"]["parent_asin"] for sample in development}
    final_targets = {sample["ground_truth"]["parent_asin"] for sample in final}
    development_users = {sample["user_group_id"] for sample in development}
    final_users = {sample["user_group_id"] for sample in final}
    if development_targets & final_targets or development_users & final_users:
        raise ValueError("Development and final targets/users must be disjoint")
    return {
        "development_scenarios": dict(sorted(Counter(item["scenario_type"] for item in development).items())),
        "final_scenarios": dict(sorted(Counter(item["scenario_type"] for item in final).items())),
        "development_categories": dict(sorted(Counter(item["category_bucket"] for item in development).items())),
        "final_categories": dict(sorted(Counter(item["category_bucket"] for item in final).items())),
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def prepare(catalog: Path, excluded_dataset: Path, output: Path, development_count: int,
            final_count: int, seed: int) -> dict:
    if output.exists():
        raise FileExistsError(output)
    excluded_rows = load_jsonl(excluded_dataset)
    excluded_targets = {
        str(row.get("ground_truth", {}).get("parent_asin", "")).strip()
        for row in excluded_rows
    }
    if "" in excluded_targets:
        raise ValueError("Excluded dataset contains a missing target")
    eligible = _eligible_products(catalog, excluded_targets)
    rng = random.Random(f"selection\0{seed}")
    selected = _stratified_selection(eligible, development_count + final_count, rng)
    development = _samples(selected[:development_count], "development", seed)
    final = _samples(selected[development_count:], "final", seed)
    summary = validate_splits(development, final, excluded_targets)
    output.mkdir(parents=True, exist_ok=False)
    development_path = output / "development.jsonl"
    final_path = output / "final-sealed.jsonl"
    _write_jsonl(development_path, development)
    _write_jsonl(final_path, final)
    manifest = {
        "schema": "mercury-unseen-evidence-v1",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "method": "Target selection and simulator rows were frozen before any agent inference.",
        "interpretation": "Synthetic unseen-target engineering evidence; not organizer-private evidence.",
        "seed": seed,
        "development_count": development_count,
        "final_count": final_count,
        "catalog_sha256": file_sha256(catalog),
        "excluded_dataset_sha256": file_sha256(excluded_dataset),
        "excluded_target_count": len(excluded_targets),
        "development_sha256": file_sha256(development_path),
        "final_sha256": file_sha256(final_path),
        **summary,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create disjoint unseen-target evaluator sessions before inference")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--exclude", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-count", type=int, default=80)
    parser.add_argument("--final-count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    if args.development_count < 20 or args.final_count < 20:
        parser.error("both split counts must be at least 20")
    print(json.dumps(prepare(args.catalog, args.exclude, args.output, args.development_count,
                             args.final_count, args.seed), indent=2))


if __name__ == "__main__":
    main()
