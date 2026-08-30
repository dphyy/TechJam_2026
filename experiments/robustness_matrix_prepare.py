"""Freeze a source-independent private-robustness evaluation matrix.

The generator never imports or executes Mercury, the evaluator, or a learned
model.  It selects catalog targets using only catalog metadata and hashes, and
keeps whole category groups, loose title families, authors, dialogue templates,
and paraphrase families inside exactly one split.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from experiments.cycle3_prepare import _catalog_rows, _digest, _read_rows, _target_ids, title_key
from experiments.cycle5_prepare import band_label, band_of, rating_number


REPOSITORY = Path(__file__).resolve().parents[1]
SEED = "robustness-matrix-20260830-v1"
VERSION = "mercury-private-robustness-matrix-v1"
LEDGER_VERSION = "mercury-private-robustness-consumption-v1"
SPLIT_COUNTS = {"training": 480, "screening": 160, "confirmation": 80, "final": 80}
EVALUATION_SPLITS = ("screening", "confirmation", "final")
SCENARIO_CYCLE = (
    "buying", "buying", "buying", "buying",
    "browsing", "browsing", "browsing", "browsing",
    "intent_override", "intent_override", "boundary",
)
STRATA = (
    "missing_price", "short_title", "sparse_features",
    "contradictory_fields", "near_duplicate_document", "complete",
)
GENERIC_CATEGORIES = frozenset(
    "clothing shoes jewelry women men girls boys accessories fashion products apparel department".split()
)
OBJECT_WORDS = frozenset(
    "bag backpack belt boot bracelet bra cap coat dress earring glove hat hoodie jacket jean "
    "legging necklace pant purse ring sandal scarf shirt shoe short skirt slipper sneaker sock "
    "sweater swimsuit tie top tote wallet watch".split()
)


def _order(seed: str, domain: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{domain}\0{value}".encode()).hexdigest()


def _tokens(value: object) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower()) if isinstance(value, str) else []


def _object_tokens(value: object) -> set[str]:
    normalized: set[str] = set()
    for token in _tokens(value):
        singular = token[:-2] if token.endswith("es") else token[:-1] if token.endswith("s") else token
        if singular in OBJECT_WORDS:
            normalized.add(singular)
    return normalized


def category_group(row: dict) -> str:
    """Return a conservative deepest-category group used as the split unit."""
    categories = row.get("categories")
    values = [value for value in categories if isinstance(value, str) and value.strip()] \
        if isinstance(categories, list) else []
    if not values:
        return "__missing_category__"
    tokens = [token for token in _tokens(values[-1]) if token not in GENERIC_CATEGORIES]
    return " ".join(tokens) or " ".join(_tokens(values[-1])) or "__missing_category__"


def metadata_strata(row: dict, family_size: int) -> tuple[str, ...]:
    """Annotate ordinary catalog quality conditions without any evaluator label."""
    found: list[str] = []
    price = row.get("price")
    if price is None or isinstance(price, bool) or not isinstance(price, (int, float, str)):
        found.append("missing_price")
    if len(_tokens(row.get("title"))) <= 4:
        found.append("short_title")
    features = row.get("features")
    details = row.get("details")
    description = row.get("description")
    feature_count = len(features) if isinstance(features, list) else 0
    if feature_count <= 1 and not details and not description:
        found.append("sparse_features")
    title_objects = _object_tokens(row.get("title"))
    category_text = " ".join(row.get("categories") or [])
    category_objects = _object_tokens(category_text)
    details_text = " ".join(
        f"{key} {value}" for key, value in (row.get("details") or {}).items()
    ) if isinstance(row.get("details"), dict) else ""
    title_gender = set(_tokens(f"{row.get('title') or ''} {details_text}")) & {"men", "mens", "women", "womens"}
    category_gender = set(_tokens(category_text)) & {"men", "mens", "women", "womens"}
    gender_conflict = bool(
        ({"men", "mens"} & title_gender and {"women", "womens"} & category_gender)
        or ({"women", "womens"} & title_gender and {"men", "mens"} & category_gender)
    )
    if gender_conflict or (title_objects and category_objects and title_objects.isdisjoint(category_objects)):
        found.append("contradictory_fields")
    if family_size > 1:
        found.append("near_duplicate_document")
    return tuple(found) or ("complete",)


def _consumed_key(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY.resolve()))
    except ValueError:
        return str(path.resolve())


def _assign_category_groups(groups: dict[str, list[str]], seed: str) -> dict[str, str]:
    """Greedily balance whole category groups across requested split capacities."""
    remaining = dict(SPLIT_COUNTS)
    assignment: dict[str, str] = {}
    ordered = sorted(groups, key=lambda key: (-len(groups[key]), _order(seed, "category", key), key))
    for group in ordered:
        eligible = [name for name, capacity in remaining.items() if capacity > 0]
        if not eligible:
            break
        split = max(
            eligible,
            key=lambda name: (remaining[name] / SPLIT_COUNTS[name], _order(seed, group, name)),
        )
        assignment[group] = split
        remaining[split] -= min(remaining[split], len(groups[group]))
    return assignment


def _select_balanced(
    identifiers: Sequence[str], by_id: dict[str, dict], family_sizes: dict[str, int], count: int, seed: str, split: str,
) -> list[str]:
    """Hash-select while covering metadata strata and popularity bands early."""
    ordered = sorted(identifiers, key=lambda value: (_order(seed, f"{split}-target", value), value))
    buckets: dict[str, list[str]] = defaultdict(list)
    for identifier in ordered:
        row = by_id[identifier]
        family = title_key(row.get("title") or "", loose=True)
        for stratum in metadata_strata(row, family_sizes[family]):
            buckets[f"stratum:{stratum}"].append(identifier)
        buckets[f"popularity:{band_label(band_of(rating_number(row)))}"].append(identifier)
    selected: list[str] = []
    seen: set[str] = set()
    keys = [f"stratum:{value}" for value in STRATA] + [f"popularity:{band_label(index)}" for index in range(6)]
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
    for identifier in ordered:
        if len(selected) == count:
            break
        if identifier not in seen:
            selected.append(identifier)
            seen.add(identifier)
    if len(selected) != count:
        raise ValueError(f"Split {split} has {len(selected)} eligible targets, expected {count}")
    return selected


def _author(split: str, index: int) -> str:
    return f"independent-{split}-author-{index % 4 + 1}"


def _template(split: str, scenario: str, index: int) -> str:
    return f"{split}-{scenario}-template-{index % 7 + 1}"


def _paraphrase(split: str, index: int) -> str:
    return f"{split}-wording-family-{index % 11 + 1}"


def build_matrix(
    products: list[dict], public_rows: list[dict], consumed: Sequence[Sequence[dict]], seed: str = SEED,
) -> dict:
    if not isinstance(seed, str) or not seed:
        raise ValueError("Seed must be a nonempty string")
    if not consumed:
        raise ValueError("At least one consumed dataset is required")
    by_id = _catalog_rows(products)
    seen_samples: set[str] = set()
    excluded_ids = _target_ids(public_rows, by_id, "public", seen_samples)
    for number, rows in enumerate(consumed, 1):
        excluded_ids.update(_target_ids(rows, by_id, f"consumed-{number}", seen_samples))
    excluded_families = {title_key(by_id[value].get("title") or "", loose=True) for value in excluded_ids}

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
        else:
            family_members[family].append(identifier)
    if len(family_members) < sum(SPLIT_COUNTS.values()):
        raise ValueError("Not enough unseen loose-title families for the matrix")

    # Choose one member per loose family before category allocation.  A near
    # duplicate remains annotated from family size without leaking relatives
    # across groups.
    representatives = {
        family: min(ids, key=lambda value: (_order(seed, "family-member", value), value))
        for family, ids in family_members.items()
    }
    category_members: dict[str, list[str]] = defaultdict(list)
    for identifier in representatives.values():
        category_members[category_group(by_id[identifier])].append(identifier)
    group_assignment = _assign_category_groups(category_members, seed)

    split_pool: dict[str, list[str]] = defaultdict(list)
    for group, identifiers in category_members.items():
        split = group_assignment.get(group)
        if split:
            split_pool[split].extend(identifiers)
    chosen = {
        split: _select_balanced(
            split_pool[split], by_id, {key: len(value) for key, value in family_members.items()},
            count, seed, split,
        )
        for split, count in SPLIT_COUNTS.items()
    }

    datasets: dict[str, list[dict]] = {}
    annotations: dict[str, list[dict]] = {}
    for split, identifiers in chosen.items():
        rows: list[dict] = []
        notes: list[dict] = []
        for offset, identifier in enumerate(identifiers):
            scenario = SCENARIO_CYCLE[offset % len(SCENARIO_CYCLE)]
            sample_id = f"robustness_{split}_{offset + 1:04d}"
            rows.append({
                "sample_id": sample_id,
                "scenario_type": scenario,
                "ground_truth": {"parent_asin": identifier},
                "user_profile": {},
                "evidence_kind": "source-frozen catalog target recovery; not organizer-private evidence",
            })
            row = by_id[identifier]
            family = title_key(row.get("title") or "", loose=True)
            notes.append({
                "sample_id": sample_id,
                "author_family": _author(split, offset),
                "user_family": f"synthetic-{split}-shopper-{offset % 13 + 1}",
                "dialogue_template_family": _template(split, scenario, offset),
                "paraphrase_family": _paraphrase(split, offset),
                "category_group": category_group(row),
                "loose_title_family_sha256": hashlib.sha256(family.encode()).hexdigest(),
                "metadata_strata": list(metadata_strata(row, len(family_members[family]))),
                "popularity_band": band_label(band_of(rating_number(row))),
                "unseen_wording_family": f"{split}-attribute-phrasing-{offset % 9 + 1}",
            })
        datasets[split] = rows
        annotations[split] = notes

    def values(split: str, key: str) -> set[str]:
        return {row[key] for row in annotations[split]}

    pairs = [(left, right) for i, left in enumerate(SPLIT_COUNTS) for right in list(SPLIT_COUNTS)[i + 1:]]
    overlap = {
        dimension: sum(len(values(left, dimension) & values(right, dimension)) for left, right in pairs)
        for dimension in (
            "author_family", "user_family", "dialogue_template_family", "paraphrase_family",
            "category_group", "loose_title_family_sha256", "unseen_wording_family",
        )
    }
    if any(overlap.values()):
        raise ValueError(f"Frozen group overlap: {overlap}")
    target_sets = {split: {row["ground_truth"]["parent_asin"] for row in rows} for split, rows in datasets.items()}
    target_overlap = sum(len(target_sets[left] & target_sets[right]) for left, right in pairs)
    if target_overlap:
        raise ValueError("Target IDs cross matrix splits")
    audit = {
        "catalog_count": len(by_id),
        "excluded_prior_target_count": len(excluded_ids),
        "eligible_title_family_count": len(family_members),
        "eligible_category_group_count": len(category_members),
        "cross_split_target_overlap": target_overlap,
        "cross_split_group_overlap": overlap,
        "counts": dict(SPLIT_COUNTS),
        "scenario_counts": {split: dict(Counter(row["scenario_type"] for row in rows)) for split, rows in datasets.items()},
        "stratum_counts": {
            split: dict(Counter(value for row in annotations[split] for value in row["metadata_strata"]))
            for split in SPLIT_COUNTS
        },
        "popularity_counts": {
            split: dict(Counter(row["popularity_band"] for row in annotations[split])) for split in SPLIT_COUNTS
        },
        "excluded_reasons": dict(excluded),
    }
    return {"datasets": datasets, "annotations": annotations, "audit": audit}


def _serialize(rows: Sequence[dict]) -> bytes:
    return "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows).encode()


def lock_matrix(
    catalog: Path, public_dataset: Path, consumed_datasets: Sequence[Path], output: Path, seed: str = SEED,
) -> dict:
    catalog, public_dataset, output = Path(catalog), Path(public_dataset), Path(output)
    consumed_paths = sorted((Path(path) for path in consumed_datasets), key=_consumed_key)
    if len({_consumed_key(path) for path in consumed_paths}) != len(consumed_paths):
        raise ValueError("Consumed dataset paths must be unique")
    matrix = build_matrix(
        _read_rows(catalog), _read_rows(public_dataset), [_read_rows(path) for path in consumed_paths], seed,
    )
    files: dict[str, bytes] = {}
    for split in SPLIT_COUNTS:
        files[f"{split}.jsonl"] = _serialize(matrix["datasets"][split])
        files[f"{split}-annotations.jsonl"] = _serialize(matrix["annotations"][split])
    manifest = {
        "version": VERSION,
        "seed": seed,
        "policy": "procedural lock; confirmation and final outcomes remain unopened until their gates",
        "consumption": {"training": "open", "screening": "sealed", "confirmation": "sealed", "final": "sealed"},
        "audit": matrix["audit"],
        "source_sha256": {
            "catalog": _digest(catalog),
            "public_dataset": _digest(public_dataset),
            "consumed_datasets": {_consumed_key(path): _digest(path) for path in consumed_paths},
            "preparation_script": _digest(Path(__file__)),
            "official_evaluator": _digest(REPOSITORY / "evaluator/local_evaluator.py"),
        },
        "file_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
    }
    manifest_path = output / "manifest.json"
    if output.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
        comparable = {key: value for key, value in previous.items() if key != "created_at_utc"} \
            if isinstance(previous, dict) else None
        if comparable != manifest:
            raise ValueError("Existing matrix lock differs from the requested source")
        for name, content in files.items():
            if not (output / name).is_file() or (output / name).read_bytes() != content:
                raise ValueError(f"Locked matrix file drift: {name}")
        if not (output / "consumption-ledger.json").is_file():
            raise ValueError("Existing matrix lock is missing its consumption ledger")
        return previous
    output.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        with (output / name).open("xb") as handle:
            handle.write(content)
    manifest["created_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    ledger = {
        "version": LEDGER_VERSION,
        "matrix_manifest_sha256": _digest(manifest_path),
        "entries": {
            "training": {"status": "open", "events": []},
            "screening": {"status": "sealed", "events": []},
            "confirmation": {"status": "sealed", "events": []},
            "final": {"status": "sealed", "events": []},
        },
    }
    with (output / "consumption-ledger.json").open("x", encoding="utf-8") as handle:
        json.dump(ledger, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def verify_lock(catalog: Path, public_dataset: Path, consumed_datasets: Sequence[Path], output: Path) -> dict:
    catalog, public_dataset, output = Path(catalog), Path(public_dataset), Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION:
        raise ValueError("Unsupported robustness matrix manifest")
    sources = manifest.get("source_sha256", {})
    if _digest(catalog) != sources.get("catalog") or _digest(public_dataset) != sources.get("public_dataset"):
        raise ValueError("Catalog or public source hash drift")
    consumed_paths = sorted((Path(path) for path in consumed_datasets), key=_consumed_key)
    actual_consumed = {_consumed_key(path): _digest(path) for path in consumed_paths}
    if actual_consumed != sources.get("consumed_datasets"):
        raise ValueError("Consumed source hash drift")
    for name, digest in manifest.get("file_sha256", {}).items():
        if not (output / name).is_file() or _digest(output / name) != digest:
            raise ValueError(f"Locked matrix file hash drift: {name}")
    ledger = json.loads((output / "consumption-ledger.json").read_text(encoding="utf-8"))
    if ledger.get("version") != LEDGER_VERSION or ledger.get("matrix_manifest_sha256") != _digest(output / "manifest.json"):
        raise ValueError("Consumption ledger does not belong to this matrix lock")
    if set(ledger.get("entries", {})) != set(SPLIT_COUNTS):
        raise ValueError("Consumption ledger split set is invalid")
    return {
        "verified": True,
        "manifest_sha256": _digest(output / "manifest.json"),
        "file_sha256": manifest["file_sha256"],
        "consumption": {name: entry.get("status") for name, entry in ledger["entries"].items()},
        "outputs_rewritten": False,
    }


def record_consumption(output: Path, split: str, purpose: str, config: Path | None = None) -> dict:
    """Open one frozen split and append an auditable consumption event."""
    output = Path(output)
    if split not in SPLIT_COUNTS or split == "training":
        raise ValueError("Only sealed evaluation splits can be consumed")
    if not isinstance(purpose, str) or not purpose.strip():
        raise ValueError("Consumption purpose must be nonempty")
    path = output / "consumption-ledger.json"
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if ledger.get("version") != LEDGER_VERSION:
        raise ValueError("Unsupported consumption ledger")
    entries = ledger.get("entries", {})
    if split == "confirmation" and entries.get("screening", {}).get("status") != "consumed":
        raise ValueError("Confirmation cannot open before screening is consumed")
    if split == "final" and entries.get("confirmation", {}).get("status") != "consumed":
        raise ValueError("Final cannot open before confirmation is consumed")
    config_hash = _digest(Path(config)) if config is not None else None
    event = {
        "opened_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": purpose.strip(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
        "config": str(Path(config)) if config is not None else None,
        "config_sha256": config_hash,
        "dataset_sha256": _digest(output / f"{split}.jsonl"),
    }
    entry = entries.get(split)
    if not isinstance(entry, dict) or entry.get("status") not in {"sealed", "consumed"} \
            or not isinstance(entry.get("events"), list):
        raise ValueError("Malformed consumption ledger entry")
    if entry["events"] and entry["events"][-1].get("purpose") == event["purpose"] \
            and entry["events"][-1].get("source_commit") == event["source_commit"] \
            and entry["events"][-1].get("config_sha256") == event["config_sha256"]:
        return ledger
    entry["status"] = "consumed"
    entry["events"].append(event)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a target- and group-disjoint robustness matrix")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--public-dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--consumed-dataset", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("artifacts/robustness-matrix-v1"))
    parser.add_argument("--seed", default=SEED)
    parser.add_argument("--verify-lock", action="store_true")
    parser.add_argument("--consume", choices=EVALUATION_SPLITS)
    parser.add_argument("--purpose")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.consume:
        if not args.purpose:
            parser.error("--consume requires --purpose")
        result = record_consumption(args.output, args.consume, args.purpose, args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if not args.consumed_dataset:
        parser.error("locking or verification requires at least one --consumed-dataset")
    result = verify_lock(args.catalog, args.public_dataset, args.consumed_dataset, args.output) \
        if args.verify_lock else lock_matrix(
            args.catalog, args.public_dataset, args.consumed_dataset, args.output, args.seed,
        )
    print(json.dumps(result if args.verify_lock else {
        "created_at_utc": result["created_at_utc"],
        "audit": result["audit"],
        "file_sha256": result["file_sha256"],
        "consumption": result["consumption"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
