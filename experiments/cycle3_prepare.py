"""Lock fresh Cycle 3 synthetic target packs without evaluating an agent.

This module only chooses catalog targets from title families.  It deliberately
does not invoke the agent, evaluator, or any model, and its CLI reports hashes
and aggregate counts rather than target identifiers.
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


SEED = "cycle3-20260827-v1"
VERSION = "catalog-targets-v2"
SCREENING_SCENARIOS = ("buying",) * 64 + ("browsing",) * 64 + ("intent_override",) * 24 + ("boundary",) * 8
SMALL_SPLIT_SCENARIOS = ("buying",) * 32 + ("browsing",) * 32 + ("intent_override",) * 12 + ("boundary",) * 4
SPLITS = {
    "screening": SCREENING_SCENARIOS,
    "confirmation": SMALL_SPLIT_SCENARIOS,
    "validation": SMALL_SPLIT_SCENARIOS,
}
COLORS = frozenset(
    "black white blue navy red green yellow pink purple brown beige grey gray orange "
    "silver gold burgundy khaki cream tan teal multicolor".split()
)
REPOSITORY = Path(__file__).resolve().parents[1]


def title_key(title: str, loose: bool = False) -> str:
    """Return the documented exact or deliberately loose title-family key."""
    tokens = re.findall(r"[a-z0-9]+", title.lower())
    if loose:
        tokens = [token for token in tokens if token not in COLORS and not token.isdigit()]
    return " ".join(tokens)


def _order(seed: str, domain: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{domain}\0{value}".encode()).hexdigest()


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("JSONL rows must be objects")
    return rows


def _catalog_rows(products: list[dict]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for row in products:
        if not isinstance(row, dict):
            raise ValueError("Catalog rows must be objects")
        identifier = row.get("parent_asin")
        if not isinstance(identifier, str) or not identifier.strip() or identifier != identifier.strip():
            raise ValueError("Catalog IDs must be nonempty normalized strings")
        if identifier in rows:
            raise ValueError("Duplicate catalog ID")
        if row.get("title") is not None and not isinstance(row["title"], str):
            raise ValueError("Catalog titles must be strings or null")
        categories = row.get("categories")
        if not isinstance(categories, list) or any(not isinstance(value, str) for value in categories):
            raise ValueError("Catalog categories must be lists of strings")
        rows[identifier] = row
    return rows


def _target_ids(rows: Sequence[dict], by_id: dict[str, dict], source: str, seen_sample_ids: set[str]) -> set[str]:
    targets: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{source} rows must be objects")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"{source} sample IDs must be nonempty strings")
        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate {source} sample ID")
        seen_sample_ids.add(sample_id)
        truth = row.get("ground_truth")
        target = truth.get("parent_asin") if isinstance(truth, dict) else None
        if not isinstance(target, str) or not target.strip():
            raise ValueError(f"Invalid {source} ground truth")
        if target not in by_id:
            raise ValueError(f"A {source} target is missing from catalog")
        targets.add(target)
    return targets


def _family_keys(by_id: dict[str, dict], identifiers: set[str]) -> set[str]:
    return {title_key(by_id[identifier].get("title") or "", loose=True) for identifier in identifiers}


def build_pack(
    products: list[dict],
    released_public_samples: list[dict],
    consumed_datasets: Sequence[Sequence[dict]],
    seed: str = SEED,
) -> dict:
    """Build target bytes in memory, excluding all previously consumed targets/families.

    The returned datasets are for the caller to lock; this function does not
    write files, evaluate a system, or inspect any result.
    """
    if not isinstance(seed, str) or not seed:
        raise ValueError("Seed must be a nonempty string")
    by_id = _catalog_rows(products)
    sample_ids: set[str] = set()
    public_ids = _target_ids(released_public_samples, by_id, "public", sample_ids)
    if not public_ids:
        raise ValueError("Released public target set must not be empty")
    consumed_ids: set[str] = set()
    for dataset in consumed_datasets:
        consumed_ids.update(_target_ids(dataset, by_id, "consumed", sample_ids))
    if not consumed_datasets:
        raise ValueError("At least one consumed target dataset is required")

    public_families = _family_keys(by_id, public_ids)
    consumed_families = _family_keys(by_id, consumed_ids)
    groups: dict[str, list[str]] = defaultdict(list)
    excluded = Counter()
    for identifier, row in by_id.items():
        family = title_key(row.get("title") or "", loose=True)
        if not family:
            excluded["ineligible_title_count"] += 1
        elif identifier in public_ids:
            excluded["public_target_count"] += 1
        elif identifier in consumed_ids:
            excluded["consumed_target_count"] += 1
        elif family in public_families:
            excluded["public_family_relative_count"] += 1
        elif family in consumed_families:
            excluded["consumed_family_relative_count"] += 1
        else:
            groups[family].append(identifier)
    required = sum(len(scenarios) for scenarios in SPLITS.values())
    if len(groups) < required:
        raise ValueError(f"Need at least {required} eligible loose title families")

    families = sorted(groups, key=lambda value: (_order(seed, "family", value), value))[:required]
    chosen = [
        min(groups[family], key=lambda identifier: (_order(seed, "member", identifier), identifier))
        for family in families
    ]
    pack: dict[str, list[dict] | dict] = {}
    offset = 0
    for split, scenarios in SPLITS.items():
        selected = chosen[offset:offset + len(scenarios)]
        offset += len(scenarios)
        selected.sort(key=lambda identifier: (_order(seed, f"scenario-{split}", identifier), identifier))
        pack[split] = [
            {
                "sample_id": f"cycle3_{split}_{index:04d}",
                "scenario_type": scenario,
                "ground_truth": {"parent_asin": target},
                "user_profile": {},
                "evidence_kind": "synthetic_catalog_target_recovery_same_official_simulator",
            }
            for index, (target, scenario) in enumerate(zip(selected, scenarios, strict=True), 1)
        ]

    split_ids = {
        name: {row["ground_truth"]["parent_asin"] for row in rows}
        for name, rows in pack.items()
        if name in SPLITS
    }
    all_ids = set().union(*split_ids.values())
    split_families = {name: _family_keys(by_id, identifiers) for name, identifiers in split_ids.items()}
    cross_target_overlap = sum(
        len(split_ids[left] & split_ids[right])
        for left, right in (("screening", "confirmation"), ("screening", "validation"), ("confirmation", "validation"))
    )
    cross_family_overlap = sum(
        len(split_families[left] & split_families[right])
        for left, right in (("screening", "confirmation"), ("screening", "validation"), ("confirmation", "validation"))
    )
    pack["audit"] = {
        "catalog_count": len(by_id),
        "released_public_sample_count": len(released_public_samples),
        "released_public_unique_targets": len(public_ids),
        "consumed_dataset_count": len(consumed_datasets),
        "consumed_unique_targets": len(consumed_ids),
        "eligible_loose_title_family_count": len(groups),
        "ineligible_title_count": excluded["ineligible_title_count"],
        "excluded_public_target_count": excluded["public_target_count"],
        "excluded_consumed_target_count": excluded["consumed_target_count"],
        "excluded_public_family_relative_count": excluded["public_family_relative_count"],
        "excluded_consumed_family_relative_count": excluded["consumed_family_relative_count"],
        "public_target_overlap": len(all_ids & public_ids),
        "consumed_target_overlap": len(all_ids & consumed_ids),
        "public_loose_title_overlap": len(_family_keys(by_id, all_ids) & public_families),
        "consumed_loose_title_overlap": len(_family_keys(by_id, all_ids) & consumed_families),
        "cross_split_target_overlap": cross_target_overlap,
        "cross_split_loose_title_overlap": cross_family_overlap,
    }
    if any(value for name, value in pack["audit"].items() if name.endswith("overlap")):
        raise ValueError("Unexpected target or loose-title-family overlap")
    return pack


def _consumed_paths(paths: Sequence[Path]) -> list[Path]:
    normalized = [Path(path) for path in paths]
    if not normalized:
        raise ValueError("At least one consumed target dataset is required")
    names = [path.name for path in normalized]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Consumed dataset filenames must be unique")
    return sorted(normalized, key=lambda path: path.name)


def _serialized_pack(pack: dict) -> dict[str, bytes]:
    return {
        name: "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in pack[name]).encode()
        for name in SPLITS
    }


def _manifest(
    catalog: Path,
    public_dataset: Path,
    consumed_datasets: Sequence[Path],
    serialized: dict[str, bytes],
    pack: dict,
    seed: str,
) -> dict:
    return {
        "version": VERSION,
        "seed": seed,
        "kind": "fresh catalog targets under the unchanged synthetic official simulator",
        "selection": "hash-order loose title families, one hash-chosen member, split-local hash order",
        "family_method": "case/punctuation normalization; loose key removes listed colors and all-digit tokens",
        "family_limitation": "Title heuristic only; no manufacturer-family independence guarantee",
        "eligibility": "valid IDs/categories and nonempty loose title; exclude public and consumed targets/families",
        "validation_policy": "Procedural lock, not encryption. Open outcomes only after source/config finalist freeze.",
        "validation_outcomes_accessed": False,
        "counts": {name: len(serialized[name].splitlines()) for name in SPLITS},
        "scenarios": {
            name: dict(Counter(row["scenario_type"] for row in pack[name]))
            for name in SPLITS
        },
        "audit": pack["audit"],
        "source_sha256": {
            "catalog": _digest(catalog),
            "public_dataset": _digest(public_dataset),
            "consumed_datasets": {path.name: _digest(path) for path in consumed_datasets},
            "preparation_script": _digest(Path(__file__)),
            "official_evaluator": _digest(REPOSITORY / "evaluator/local_evaluator.py"),
        },
        "dataset_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in serialized.items()},
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
    }


def lock_pack(
    catalog: Path,
    public_dataset: Path,
    consumed_datasets: Sequence[Path],
    output: Path,
    seed: str = SEED,
) -> dict:
    """Write a single idempotent lock, refusing all replacement or tampering."""
    catalog, public_dataset, output = Path(catalog), Path(public_dataset), Path(output)
    consumed_paths = _consumed_paths(consumed_datasets)
    pack = build_pack(
        _read_rows(catalog),
        _read_rows(public_dataset),
        [_read_rows(path) for path in consumed_paths],
        seed,
    )
    serialized = _serialized_pack(pack)
    manifest = _manifest(catalog, public_dataset, consumed_paths, serialized, pack, seed)
    manifest_path = output / "manifest.json"
    if output.exists():
        if not manifest_path.is_file():
            raise ValueError("Output already exists without a lock; refusing to overwrite")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(previous, dict) or {
            key: value for key, value in previous.items() if key != "created_at_utc"
        } != manifest:
            raise ValueError("Existing lock differs from requested inputs or preparation source")
        for name, content in serialized.items():
            path = output / f"{name}.jsonl"
            if not path.is_file() or path.read_bytes() != content:
                raise ValueError(f"Locked {name} differs from recorded generation")
        return previous
    output.mkdir(parents=True, exist_ok=False)
    for name, content in serialized.items():
        with (output / f"{name}.jsonl").open("xb") as handle:
            handle.write(content)
    manifest["created_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with manifest_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify_lock(
    catalog: Path,
    public_dataset: Path,
    consumed_datasets: Sequence[Path],
    output: Path,
    preparation_source: Path | None = None,
) -> dict:
    """Read-only verification; source commit drift is allowed, byte drift is not."""
    catalog, public_dataset, output = Path(catalog), Path(public_dataset), Path(output)
    consumed_paths = _consumed_paths(consumed_datasets)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("version") != VERSION:
        raise ValueError("Unsupported target-pack manifest")
    source_hashes = manifest.get("source_sha256")
    data_hashes = manifest.get("dataset_sha256")
    if not isinstance(source_hashes, dict) or not isinstance(data_hashes, dict):
        raise ValueError("Invalid target-pack manifest hashes")

    current_source = Path(__file__)
    if preparation_source is None:
        if _digest(current_source) == source_hashes.get("preparation_script"):
            preparation_source = current_source
        else:
            preparation_source = output.parent / "provenance/cycle3_prepare-original.py"
            if not preparation_source.is_file():
                raise ValueError("A preserved preparation_script source snapshot is required; current source differs")
    preparation_source = Path(preparation_source)
    sources = {
        "catalog": catalog,
        "public_dataset": public_dataset,
        "preparation_script": preparation_source,
        "official_evaluator": REPOSITORY / "evaluator/local_evaluator.py",
    }
    for name, path in sources.items():
        if not path.is_file():
            raise ValueError(f"Missing {name} source snapshot")
        if _digest(path) != source_hashes.get(name):
            raise ValueError(f"Locked {name} source hash mismatch")
    expected_consumed = source_hashes.get("consumed_datasets")
    actual_consumed = {path.name: _digest(path) for path in consumed_paths}
    if not isinstance(expected_consumed, dict) or actual_consumed != expected_consumed:
        raise ValueError("Locked consumed_dataset source hash mismatch")
    for name in SPLITS:
        path = output / f"{name}.jsonl"
        if not path.is_file() or _digest(path) != data_hashes.get(name):
            raise ValueError(f"Locked {name} data hash mismatch")
    current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip()
    return {
        "verified": True,
        "verification": "read-only locked data, source inputs, and original preparation source hashes",
        "manifest_sha256": _digest(manifest_path),
        "dataset_sha256": data_hashes,
        "preparation_source_sha256": source_hashes["preparation_script"],
        "verification_source_sha256": _digest(current_source),
        "original_source_commit": manifest.get("source_commit"),
        "current_source_commit": current_commit,
        "commit_changed": current_commit != manifest.get("source_commit"),
        "outputs_rewritten": False,
        "validation_outcomes_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock fresh Cycle 3 synthetic targets without evaluation")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--public-dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--consumed-dataset", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/cycle3/synthetic-targets"))
    parser.add_argument("--seed", default=SEED)
    parser.add_argument("--verify-lock", action="store_true", help="Check recorded bytes without regenerating data")
    parser.add_argument("--preparation-source", type=Path, help="Preserved original generator source after a source change")
    args = parser.parse_args()
    if args.verify_lock:
        receipt = verify_lock(
            args.catalog, args.public_dataset, args.consumed_dataset, args.output, args.preparation_source
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return
    manifest = lock_pack(args.catalog, args.public_dataset, args.consumed_dataset, args.output, args.seed)
    print(
        json.dumps(
            {key: manifest[key] for key in ("created_at_utc", "counts", "scenarios", "audit", "dataset_sha256")},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
