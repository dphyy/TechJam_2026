"""Freeze the second source-independent robustness matrix.

This continuation matrix excludes targets, loose title families, and deepest
category groups represented in every supplied consumed dataset.  Dialogue
authorship metadata is namespaced independently from the first matrix and is
checked against any supplied prior annotation files.  The original v1 final
split is intentionally not copied or opened by this tool.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from experiments.cycle3_prepare import _catalog_rows, _digest, _read_rows, _target_ids, title_key
from experiments.robustness_matrix_prepare import (
    _consumed_key,
    _order,
    _serialize,
    category_group,
    metadata_strata,
)
from experiments.cycle5_prepare import band_label, band_of, rating_number


REPOSITORY = Path(__file__).resolve().parents[1]
SEED = "robustness-matrix-20260830-v2"
VERSION = "mercury-private-robustness-matrix-v2"
LEDGER_VERSION = "mercury-private-robustness-consumption-v2"
SPLIT_COUNTS = {"training": 480, "screening": 160, "confirmation": 80}
EVALUATION_SPLITS = ("screening", "confirmation")
SCENARIO_CYCLE = (
    "buying", "buying", "buying", "buying", "buying",
    "browsing", "browsing", "browsing",
    "intent_override", "intent_override", "boundary",
)
ANNOTATION_GROUPS = (
    "author_family", "user_family", "dialogue_template_family",
    "paraphrase_family", "unseen_wording_family",
)


def power_calculation(count: int, baseline_rate: float = 0.90,
                      minimum_effect: float = 0.05, alpha: float = 0.05) -> dict:
    """Record a conservative normal-approximation sensitivity calculation.

    This is a planning receipt, not a claim that synthetic observations are
    independent.  The effective sample size is therefore also reported at 75%.
    """
    if type(count) is not int or count < 1:
        raise ValueError("Power calculation count must be positive")
    if not 0 < baseline_rate < 1 or not 0 < minimum_effect < 1:
        raise ValueError("Power assumptions must be probabilities")
    z_alpha = 1.959963984540054
    effective_n = max(1, math.floor(count * 0.75))
    standard_error = math.sqrt(baseline_rate * (1.0 - baseline_rate) / effective_n)
    z_effect = minimum_effect / standard_error
    # Phi approximation from erf; two-sided rejection under the assumed shift.
    def phi(value: float) -> float:
        return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))
    approximate_power = phi(z_effect - z_alpha) + phi(-z_effect - z_alpha)
    return {
        "nominal_rows": count,
        "effective_rows_at_75_percent": effective_n,
        "baseline_hit_rate": baseline_rate,
        "minimum_absolute_effect": minimum_effect,
        "two_sided_alpha": alpha,
        "approximate_power": round(approximate_power, 6),
        "method": "normal approximation with a conservative 0.75 effective-sample factor",
    }


def _prior_annotation_values(annotation_rows: Sequence[Sequence[dict]]) -> dict[str, set[str]]:
    values = {key: set() for key in ANNOTATION_GROUPS}
    for rows in annotation_rows:
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Prior annotation rows must be objects")
            for key in ANNOTATION_GROUPS:
                value = row.get(key)
                if isinstance(value, str) and value:
                    values[key].add(value)
    return values


def _assign_groups(groups: dict[str, list[str]], seed: str) -> dict[str, str]:
    load = {split: 0 for split in SPLIT_COUNTS}
    assignment: dict[str, str] = {}
    for group in sorted(groups, key=lambda value: (_order(seed, "category", value), value)):
        split = min(
            SPLIT_COUNTS,
            key=lambda value: (load[value] / SPLIT_COUNTS[value], _order(seed, group, value)),
        )
        assignment[group] = split
        load[split] += min(len(groups[group]), max(4, SPLIT_COUNTS[split] // 8))
    return assignment


def _balanced_select(identifiers: Sequence[str], by_id: dict[str, dict],
                     family_sizes: dict[str, int], count: int, seed: str,
                     split: str) -> list[str]:
    ordered = sorted(identifiers, key=lambda value: (_order(seed, f"{split}-target", value), value))
    buckets: dict[str, list[str]] = defaultdict(list)
    for identifier in ordered:
        row = by_id[identifier]
        family = title_key(row.get("title") or "", loose=True)
        for stratum in metadata_strata(row, family_sizes[family]):
            buckets[f"metadata:{stratum}"].append(identifier)
        buckets[f"popularity:{band_label(band_of(rating_number(row)))}"].append(identifier)
    selected: list[str] = []
    seen: set[str] = set()
    keys = sorted(buckets)
    while len(selected) < count:
        progress = False
        for key in keys:
            while buckets[key] and buckets[key][0] in seen:
                buckets[key].pop(0)
            if buckets[key] and len(selected) < count:
                identifier = buckets[key].pop(0)
                selected.append(identifier)
                seen.add(identifier)
                progress = True
        if not progress:
            break
    selected.extend(identifier for identifier in ordered if identifier not in seen)
    selected = selected[:count]
    if len(selected) != count:
        raise ValueError(f"Split {split} has {len(selected)} eligible targets, expected {count}")
    return selected


def build_matrix(products: list[dict], public_rows: list[dict],
                 consumed: Sequence[Sequence[dict]],
                 prior_annotations: Sequence[Sequence[dict]] = (),
                 seed: str = SEED) -> dict:
    if not isinstance(seed, str) or not seed:
        raise ValueError("Seed must be a nonempty string")
    if not consumed:
        raise ValueError("At least one consumed dataset is required")
    by_id = _catalog_rows(products)
    excluded_ids = _target_ids(public_rows, by_id, "public", set())
    for index, rows in enumerate(consumed, 1):
        # The same frozen session can legitimately occur in more than one
        # historical aggregate pack; uniqueness is enforced within each source.
        excluded_ids.update(_target_ids(rows, by_id, f"consumed-{index}", set()))
    excluded_families = {
        title_key(by_id[identifier].get("title") or "", loose=True)
        for identifier in excluded_ids
    }
    excluded_categories = {category_group(by_id[identifier]) for identifier in excluded_ids}

    family_members: dict[str, list[str]] = defaultdict(list)
    excluded = Counter()
    for identifier, row in by_id.items():
        family = title_key(row.get("title") or "", loose=True)
        if not family:
            excluded["empty_title_family"] += 1
        elif identifier in excluded_ids:
            excluded["prior_target"] += 1
        elif family in excluded_families:
            excluded["prior_title_family"] += 1
        elif category_group(row) in excluded_categories:
            excluded["prior_category_group"] += 1
        else:
            family_members[family].append(identifier)
    if len(family_members) < sum(SPLIT_COUNTS.values()):
        raise ValueError("Not enough source-disjoint title families for matrix v2")

    representatives = {
        family: min(ids, key=lambda value: (_order(seed, "family-member", value), value))
        for family, ids in family_members.items()
    }
    category_members: dict[str, list[str]] = defaultdict(list)
    for identifier in representatives.values():
        category_members[category_group(by_id[identifier])].append(identifier)
    assignment = _assign_groups(category_members, seed)
    pools: dict[str, list[str]] = defaultdict(list)
    for group, identifiers in category_members.items():
        pools[assignment[group]].extend(identifiers)
    family_sizes = {family: len(ids) for family, ids in family_members.items()}
    chosen = {
        split: _balanced_select(pools[split], by_id, family_sizes, count, seed, split)
        for split, count in SPLIT_COUNTS.items()
    }

    prior_values = _prior_annotation_values(prior_annotations)
    datasets: dict[str, list[dict]] = {}
    annotations: dict[str, list[dict]] = {}
    for split, identifiers in chosen.items():
        rows = []
        notes = []
        for offset, identifier in enumerate(identifiers):
            scenario = SCENARIO_CYCLE[offset % len(SCENARIO_CYCLE)]
            sample_id = f"continuation_{split}_{offset + 1:04d}"
            rows.append({
                "sample_id": sample_id,
                "scenario_type": scenario,
                "ground_truth": {"parent_asin": identifier},
                "user_profile": {},
                "evidence_kind": "source-frozen continuation proxy; not organizer-private evidence",
            })
            product = by_id[identifier]
            family = title_key(product.get("title") or "", loose=True)
            note = {
                "sample_id": sample_id,
                "author_family": f"continuation-{split}-author-{offset % 8 + 1}",
                "user_family": f"continuation-{split}-shopper-{offset % 17 + 1}",
                "dialogue_template_family": f"continuation-{split}-{scenario}-template-{offset % 13 + 1}",
                "paraphrase_family": f"continuation-{split}-paraphrase-{offset % 19 + 1}",
                "unseen_wording_family": f"continuation-{split}-wording-{offset % 11 + 1}",
                "category_group": category_group(product),
                "loose_title_family_sha256": hashlib.sha256(family.encode()).hexdigest(),
                "metadata_strata": list(metadata_strata(product, family_sizes[family])),
                "popularity_band": band_label(band_of(rating_number(product))),
                "authoring_blind": True,
            }
            for key in ANNOTATION_GROUPS:
                if note[key] in prior_values[key]:
                    raise ValueError(f"Prior {key} leaked into matrix v2")
            notes.append(note)
        datasets[split] = rows
        annotations[split] = notes

    pairs = [
        (left, right)
        for index, left in enumerate(SPLIT_COUNTS)
        for right in list(SPLIT_COUNTS)[index + 1:]
    ]
    group_dimensions = (*ANNOTATION_GROUPS, "category_group", "loose_title_family_sha256")
    overlap = {
        dimension: sum(
            len({row[dimension] for row in annotations[left]}
                & {row[dimension] for row in annotations[right]})
            for left, right in pairs
        )
        for dimension in group_dimensions
    }
    targets = {
        split: {row["ground_truth"]["parent_asin"] for row in rows}
        for split, rows in datasets.items()
    }
    target_overlap = sum(len(targets[left] & targets[right]) for left, right in pairs)
    if target_overlap or any(overlap.values()):
        raise ValueError(f"Matrix v2 split overlap: targets={target_overlap}, groups={overlap}")
    if any(category_group(by_id[target]) in excluded_categories for values in targets.values() for target in values):
        raise ValueError("A previously consumed category group entered matrix v2")

    audit = {
        "catalog_count": len(by_id),
        "excluded_prior_target_count": len(excluded_ids),
        "excluded_prior_title_family_count": len(excluded_families),
        "excluded_prior_category_group_count": len(excluded_categories),
        "eligible_title_family_count": len(family_members),
        "eligible_category_group_count": len(category_members),
        "cross_split_target_overlap": target_overlap,
        "cross_split_group_overlap": overlap,
        "counts": dict(SPLIT_COUNTS),
        "scenario_counts": {
            split: dict(Counter(row["scenario_type"] for row in rows))
            for split, rows in datasets.items()
        },
        "power_calculation": {
            split: power_calculation(count) for split, count in SPLIT_COUNTS.items()
        },
        "excluded_reasons": dict(excluded),
    }
    return {"datasets": datasets, "annotations": annotations, "audit": audit}


def lock_matrix(catalog: Path, public_dataset: Path, consumed_datasets: Sequence[Path],
                prior_annotation_paths: Sequence[Path], output: Path,
                seed: str = SEED) -> dict:
    catalog, public_dataset, output = Path(catalog), Path(public_dataset), Path(output)
    consumed_paths = sorted((Path(path) for path in consumed_datasets), key=_consumed_key)
    annotation_paths = sorted((Path(path) for path in prior_annotation_paths), key=_consumed_key)
    if len({_consumed_key(path) for path in consumed_paths}) != len(consumed_paths):
        raise ValueError("Consumed dataset paths must be unique")
    matrix = build_matrix(
        _read_rows(catalog), _read_rows(public_dataset),
        [_read_rows(path) for path in consumed_paths],
        [_read_rows(path) for path in annotation_paths], seed,
    )
    files: dict[str, bytes] = {}
    for split in SPLIT_COUNTS:
        files[f"{split}.jsonl"] = _serialize(matrix["datasets"][split])
        files[f"{split}-annotations.jsonl"] = _serialize(matrix["annotations"][split])
    manifest = {
        "version": VERSION,
        "seed": seed,
        "policy": "screening then confirmation; original robustness v1 final remains sealed",
        "consumption": {"training": "open", "screening": "sealed", "confirmation": "sealed"},
        "audit": matrix["audit"],
        "source_sha256": {
            "catalog": _digest(catalog),
            "public_dataset": _digest(public_dataset),
            "consumed_datasets": {_consumed_key(path): _digest(path) for path in consumed_paths},
            "prior_annotations": {_consumed_key(path): _digest(path) for path in annotation_paths},
            "preparation_script": _digest(Path(__file__)),
            "official_evaluator": _digest(REPOSITORY / "evaluator/local_evaluator.py"),
        },
        "file_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True,
        ).strip(),
    }
    if output.exists():
        previous = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        comparable = {key: value for key, value in previous.items() if key != "created_at_utc"}
        if comparable != manifest:
            raise ValueError("Existing matrix v2 lock differs from requested source")
        for name, content in files.items():
            if (output / name).read_bytes() != content:
                raise ValueError(f"Locked matrix v2 file drift: {name}")
        return previous
    output.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        with (output / name).open("xb") as handle:
            handle.write(content)
    manifest["created_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    ledger = {
        "version": LEDGER_VERSION,
        "matrix_manifest_sha256": _digest(output / "manifest.json"),
        "entries": {
            "training": {"status": "open", "events": []},
            "screening": {"status": "sealed", "events": []},
            "confirmation": {"status": "sealed", "events": []},
        },
    }
    (output / "consumption-ledger.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return manifest


def record_consumption(output: Path, split: str, purpose: str, config: Path | None = None) -> dict:
    output = Path(output)
    if split not in EVALUATION_SPLITS:
        raise ValueError("Only screening or confirmation can be consumed")
    if not isinstance(purpose, str) or not purpose.strip():
        raise ValueError("Consumption purpose must be nonempty")
    path = output / "consumption-ledger.json"
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if ledger.get("version") != LEDGER_VERSION:
        raise ValueError("Unsupported matrix v2 ledger")
    entries = ledger["entries"]
    if split == "confirmation" and entries["screening"]["status"] != "consumed":
        raise ValueError("Confirmation cannot open before screening")
    event = {
        "opened_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": purpose.strip(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True,
        ).strip(),
        "config": str(config) if config is not None else None,
        "config_sha256": _digest(config) if config is not None else None,
        "dataset_sha256": _digest(output / f"{split}.jsonl"),
    }
    entry = entries[split]
    entry["status"] = "consumed"
    entry["events"].append(event)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the source-disjoint continuation matrix")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--public-dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--consumed-dataset", type=Path, action="append", default=[])
    parser.add_argument("--prior-annotations", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("artifacts/robustness-matrix-v2"))
    parser.add_argument("--seed", default=SEED)
    parser.add_argument("--consume", choices=EVALUATION_SPLITS)
    parser.add_argument("--purpose")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.consume:
        if not args.purpose:
            parser.error("--consume requires --purpose")
        print(json.dumps(record_consumption(args.output, args.consume, args.purpose, args.config), indent=2))
        return
    if not args.consumed_dataset:
        parser.error("locking requires at least one --consumed-dataset")
    result = lock_matrix(
        args.catalog, args.public_dataset, args.consumed_dataset,
        args.prior_annotations, args.output, args.seed,
    )
    print(json.dumps({
        "created_at_utc": result["created_at_utc"],
        "audit": result["audit"],
        "file_sha256": result["file_sha256"],
        "consumption": result["consumption"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
