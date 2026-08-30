"""Lock a popularity-matched Cycle 5 synthetic target pack.

This module only chooses catalog targets from title families.  It deliberately
does not invoke the agent, evaluator, or any model, and its CLI reports hashes
and aggregate counts rather than target identifiers.

It reuses the Cycle 3 eligibility, family, hashing and audit machinery
unchanged.  The single difference is the draw: Cycle 3 takes families in hash
order, which samples the catalog uniformly, while this module fills per-band
quotas derived at runtime from the released-public targets.  See
``docs/CYCLE5_TARGET_POOL_PROTOCOL.md`` for the registration and the
pre-declared upper-tail limitation.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from experiments.cycle3_prepare import (
    REPOSITORY,
    _catalog_rows,
    _consumed_paths,
    _digest,
    _family_keys,
    _order,
    _read_rows,
    _target_ids,
    title_key,
)


SEED = "cycle5-20260829-v1"
VERSION = "catalog-targets-v3-popularity-matched"
SCREENING_SCENARIOS = ("buying",) * 64 + ("browsing",) * 64 + ("intent_override",) * 24 + ("boundary",) * 8
SMALL_SPLIT_SCENARIOS = ("buying",) * 32 + ("browsing",) * 32 + ("intent_override",) * 12 + ("boundary",) * 4
SPLITS = {
    "screening": SCREENING_SCENARIOS,
    "confirmation": SMALL_SPLIT_SCENARIOS,
    "validation": SMALL_SPLIT_SCENARIOS,
}
# Popularity bands over rating_number, low inclusive and high exclusive.
BAND_EDGES = ((0, 5), (5, 100), (100, 1000), (1000, 5000), (5000, 20000), (20000, None))
# Repeating split cycle; 320 targets over a 4-slot cycle yields 160/80/80.
SPLIT_CYCLE = ("screening", "screening", "confirmation", "validation")


def band_label(index: int) -> str:
    low, high = BAND_EDGES[index]
    return f"{low}-{high}" if high is not None else f"{low}+"


def rating_number(row: dict) -> int:
    """Ratings count as a nonnegative integer; anything unusable reads as zero."""
    value = row.get("rating_number")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    if value != value or value in (float("inf"), float("-inf")) or value < 0:
        return 0
    return int(value)


def band_of(count: int) -> int:
    for index, (low, high) in enumerate(BAND_EDGES):
        if low <= count and (high is None or count < high):
            return index
    raise ValueError(f"Unbandable rating count: {count!r}")


def derive_quotas(counts: Sequence[int], total: int) -> list[int]:
    """Largest-remainder allocation of ``total`` across the reference shares."""
    reference = sum(counts)
    if reference <= 0:
        raise ValueError("Reference distribution must be nonempty")
    if not isinstance(total, int) or total < 1:
        raise ValueError("Total must be a positive integer")
    exact = [count * total / reference for count in counts]
    quotas = [int(value) for value in exact]
    remainder = total - sum(quotas)
    order = sorted(range(len(counts)), key=lambda index: (-(exact[index] - quotas[index]), index))
    for index in order[:remainder]:
        quotas[index] += 1
    return quotas


def allocate(quotas: Sequence[int], available: Sequence[int]) -> tuple[list[int], list[int]]:
    """Fill bands from the top down, cascading any shortfall into the next band.

    Returns the realised per-band counts and the per-band shortfall that had to
    be served elsewhere.  A band that cannot be filled never silently reduces the
    total; the deficit is carried, and a final pass spends it wherever capacity
    remains.
    """
    if len(quotas) != len(available):
        raise ValueError("Quota and availability lengths differ")
    taken = [0] * len(quotas)
    shortfall = [0] * len(quotas)
    carry = 0
    for index in range(len(quotas) - 1, -1, -1):
        want = quotas[index] + carry
        take = min(want, available[index])
        taken[index] = take
        deficit = want - take
        shortfall[index] = max(0, quotas[index] - take)
        carry = deficit
    for index in range(len(quotas) - 1, -1, -1):
        if carry <= 0:
            break
        spare = available[index] - taken[index]
        extra = min(spare, carry)
        taken[index] += extra
        carry -= extra
    if carry:
        raise ValueError("Eligible families cannot supply the requested target count")
    return taken, shortfall


def build_pack(
    products: list[dict],
    released_public_samples: list[dict],
    consumed_datasets: Sequence[Sequence[dict]],
    seed: str = SEED,
) -> dict:
    """Build target bytes in memory under a popularity-matched stratified draw.

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
    if not consumed_datasets:
        raise ValueError("At least one consumed target dataset is required")
    consumed_ids: set[str] = set()
    for dataset in consumed_datasets:
        consumed_ids.update(_target_ids(dataset, by_id, "consumed", sample_ids))

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

    # One hash-chosen member per family, exactly as Cycle 3, then band by that
    # member so the recorded distribution describes the products actually used.
    members = {
        family: min(identifiers, key=lambda identifier: (_order(seed, "member", identifier), identifier))
        for family, identifiers in groups.items()
    }
    banded: dict[int, list[str]] = {index: [] for index in range(len(BAND_EDGES))}
    for family, identifier in members.items():
        banded[band_of(rating_number(by_id[identifier]))].append(identifier)
    for index, identifiers in banded.items():
        identifiers.sort(key=lambda identifier: (_order(seed, f"band-{index}", identifier), identifier))

    reference = [0] * len(BAND_EDGES)
    for identifier in public_ids:
        reference[band_of(rating_number(by_id[identifier]))] += 1
    quotas = derive_quotas(reference, required)
    available = [len(banded[index]) for index in range(len(BAND_EDGES))]
    taken, shortfall = allocate(quotas, available)

    # Deal every band across splits on one continuous cycle, so each split holds
    # a proportional share of each band rather than a popularity-sorted slice.
    assigned: dict[str, list[str]] = {name: [] for name in SPLITS}
    position = 0
    for index in range(len(BAND_EDGES) - 1, -1, -1):
        for identifier in banded[index][:taken[index]]:
            assigned[SPLIT_CYCLE[position % len(SPLIT_CYCLE)]].append(identifier)
            position += 1
    for name, scenarios in SPLITS.items():
        if len(assigned[name]) != len(scenarios):
            raise ValueError(f"Split {name} received {len(assigned[name])} targets, expected {len(scenarios)}")

    pack: dict[str, list[dict] | dict] = {}
    for split, scenarios in SPLITS.items():
        selected = sorted(
            assigned[split], key=lambda identifier: (_order(seed, f"scenario-{split}", identifier), identifier)
        )
        pack[split] = [
            {
                "sample_id": f"cycle5_{split}_{index:04d}",
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
    pairs = (("screening", "confirmation"), ("screening", "validation"), ("confirmation", "validation"))
    achieved = [0] * len(BAND_EDGES)
    for identifier in all_ids:
        achieved[band_of(rating_number(by_id[identifier]))] += 1
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
        "cross_split_target_overlap": sum(len(split_ids[left] & split_ids[right]) for left, right in pairs),
        "cross_split_loose_title_overlap": sum(len(split_families[left] & split_families[right]) for left, right in pairs),
        "popularity_bands": {
            band_label(index): {
                "released_public_targets": reference[index],
                "released_public_share": round(reference[index] / len(public_ids), 6),
                "quota": quotas[index],
                "eligible_families": available[index],
                "selected": achieved[index],
                "unfilled_quota_served_elsewhere": shortfall[index],
            }
            for index in range(len(BAND_EDGES))
        },
        "band_shortfall_total": sum(shortfall),
    }
    if any(value for name, value in pack["audit"].items() if name.endswith("overlap")):
        raise ValueError("Unexpected target or loose-title-family overlap")
    if achieved != taken:
        raise ValueError("Realised band counts do not match the allocation")
    return pack


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
        "kind": "popularity-matched catalog targets under the unchanged synthetic official simulator",
        "selection": "band eligible families by chosen-member rating_number, fill runtime quotas derived "
                     "from released-public shares, cascade shortfall downward, deal bands across splits",
        "reference_population": "released public 200 targets; the only available sample of the organizer pool",
        "family_method": "case/punctuation normalization; loose key removes listed colors and all-digit tokens",
        "family_limitation": "Title heuristic only; no manufacturer-family independence guarantee",
        "eligibility": "valid IDs/categories and nonempty loose title; exclude public and consumed targets/families",
        "tail_limitation": "The catalog cannot supply the organizer's extreme upper tail; the shortfall is "
                           "recorded per band in audit.popularity_bands and cascades into the next lower band",
        "comparability": "Not comparable to the Cycle 3 pack; every arm must state which pack it used",
        "validation_policy": "Procedural lock, not encryption. Open outcomes only after source/config finalist freeze.",
        "validation_outcomes_accessed": False,
        "counts": {name: len(serialized[name].splitlines()) for name in SPLITS},
        "scenarios": {name: dict(Counter(row["scenario_type"] for row in pack[name])) for name in SPLITS},
        "audit": pack["audit"],
        "source_sha256": {
            "catalog": _digest(catalog),
            "public_dataset": _digest(public_dataset),
            "consumed_datasets": {path.name: _digest(path) for path in consumed_datasets},
            "preparation_script": _digest(Path(__file__)),
            "cycle3_preparation_script": _digest(REPOSITORY / "experiments/cycle3_prepare.py"),
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
) -> dict:
    """Read-only verification; source commit drift is allowed, byte drift is not."""
    catalog, public_dataset, output = Path(catalog), Path(public_dataset), Path(output)
    consumed_paths = _consumed_paths(consumed_datasets)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("version") != VERSION:
        raise ValueError("Unsupported target-pack manifest")
    source_hashes = manifest.get("source_sha256")
    data_hashes = manifest.get("dataset_sha256")
    if not isinstance(source_hashes, dict) or not isinstance(data_hashes, dict):
        raise ValueError("Invalid target-pack manifest hashes")
    sources = {
        "catalog": catalog,
        "public_dataset": public_dataset,
        "cycle3_preparation_script": REPOSITORY / "experiments/cycle3_prepare.py",
        "official_evaluator": REPOSITORY / "evaluator/local_evaluator.py",
    }
    for name, path in sources.items():
        if not path.is_file():
            raise ValueError(f"Missing {name} source snapshot")
        if _digest(path) != source_hashes.get(name):
            raise ValueError(f"Locked {name} source hash mismatch")
    actual_consumed = {path.name: _digest(path) for path in consumed_paths}
    if actual_consumed != source_hashes.get("consumed_datasets"):
        raise ValueError("Locked consumed_dataset source hash mismatch")
    for name in SPLITS:
        path = output / f"{name}.jsonl"
        if not path.is_file() or _digest(path) != data_hashes.get(name):
            raise ValueError(f"Locked {name} data hash mismatch")
    return {"verified": True, "version": manifest["version"], "counts": manifest["counts"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock or verify the popularity-matched Cycle 5 target pack")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--public", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--consumed", type=Path, nargs="+", required=True,
                        help="Every previously consumed target dataset, including the Cycle 3 splits")
    parser.add_argument("--output", type=Path, default=Path("artifacts/cycle5/synthetic-targets"))
    parser.add_argument("--seed", default=SEED)
    parser.add_argument("--verify", action="store_true", help="Verify an existing lock without writing")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_lock(args.catalog, args.public, args.consumed, args.output), indent=2))
        return
    manifest = lock_pack(args.catalog, args.public, args.consumed, args.output, args.seed)
    print(json.dumps({key: manifest[key] for key in ("version", "seed", "counts", "scenarios", "dataset_sha256")}, indent=2))
    print(json.dumps(manifest["audit"]["popularity_bands"], indent=2))


if __name__ == "__main__":
    main()
