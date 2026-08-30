from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


SEED = "cycle2-20260826-v1"
VERSION = "catalog-targets-v1"
SCENARIOS = ("buying",) * 12 + ("browsing",) * 12 + ("intent_override",) * 6 + ("boundary",) * 2
COLORS = frozenset("black white blue navy red green yellow pink purple brown beige grey gray orange "
                   "silver gold burgundy khaki cream tan teal multicolor".split())
REPOSITORY = Path(__file__).resolve().parents[1]


def title_key(title: str, loose: bool = False) -> str:
    tokens = re.findall(r"[a-z0-9]+", title.lower())
    if loose:
        tokens = [token for token in tokens if token not in COLORS and not token.isdigit()]
    return " ".join(tokens)


def _order(seed: str, domain: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{domain}\0{value}".encode()).hexdigest()


def _catalog_rows(products: list[dict]) -> dict[str, dict]:
    result = {}
    for row in products:
        if not isinstance(row, dict):
            raise ValueError("Catalog rows must be objects")
        identifier = row.get("parent_asin")
        if not isinstance(identifier, str) or not identifier.strip() or identifier != identifier.strip():
            raise ValueError("Catalog IDs must be nonempty normalized strings")
        if identifier in result:
            raise ValueError(f"Duplicate catalog ID: {identifier}")
        if row.get("title") is not None and not isinstance(row["title"], str):
            raise ValueError("Catalog titles must be strings or null")
        categories = row.get("categories")
        if not isinstance(categories, list) or any(not isinstance(value, str) for value in categories):
            raise ValueError("Catalog categories must be lists of strings")
        result[identifier] = row
    return result


def build_pack(products: list[dict], old_samples: list[dict], seed: str = SEED) -> dict:
    """Prepare new target groups; do not run an agent or inspect any outcome."""
    if not isinstance(seed, str) or not seed:
        raise ValueError("Seed must be a nonempty string")
    by_id = _catalog_rows(products)
    old_ids: set[str] = set()
    sample_ids: set[str] = set()
    for row in old_samples:
        if not isinstance(row, dict):
            raise ValueError("Old samples must be objects")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("Old sample IDs must be nonempty strings")
        if sample_id in sample_ids:
            raise ValueError("Duplicate old sample ID")
        sample_ids.add(sample_id)
        truth = row.get("ground_truth")
        if not isinstance(truth, dict) or not isinstance(truth.get("parent_asin"), str):
            raise ValueError("Invalid old ground truth")
        target = truth["parent_asin"]
        if target not in by_id:
            raise ValueError("An old target is missing from catalog")
        old_ids.add(target)
    if not old_ids:
        raise ValueError("Old target set must not be empty")
    old_exact = {title_key(by_id[target].get("title") or "") for target in old_ids}
    old_loose = {title_key(by_id[target].get("title") or "", loose=True) for target in old_ids}
    groups: dict[str, list[str]] = defaultdict(list)
    excluded = Counter()
    for identifier, row in by_id.items():
        exact = title_key(row.get("title") or "")
        family = title_key(row.get("title") or "", loose=True)
        if not exact or not family:
            excluded["ineligible_title_count"] += 1
        elif identifier in old_ids:
            excluded["old_target_count"] += 1
        elif exact in old_exact or family in old_loose:
            excluded["old_family_relative_count"] += 1
        else:
            groups[family].append(identifier)
    if len(groups) < 64:
        raise ValueError("Need at least 64 eligible title families")
    families = sorted(groups, key=lambda family: (_order(seed, "family", family), family))[:64]
    chosen = [min(groups[family], key=lambda target: (_order(seed, "member", target), target))
              for family in families]
    pack = {}
    for split, selected in (("development", chosen[::2]), ("validation", chosen[1::2])):
        selected.sort(key=lambda target: (_order(seed, f"scenario-{split}", target), target))
        pack[split] = [
            {"sample_id": f"cycle2_{split}_{index:04d}", "scenario_type": scenario,
             "ground_truth": {"parent_asin": target}, "user_profile": {},
             "evidence_kind": "synthetic_catalog_target_recovery_same_official_simulator"}
            for index, (target, scenario) in enumerate(zip(selected, SCENARIOS, strict=True), 1)
        ]
    dev = {row["ground_truth"]["parent_asin"] for row in pack["development"]}
    validation = {row["ground_truth"]["parent_asin"] for row in pack["validation"]}

    def keys(identifiers: set[str], loose: bool) -> set[str]:
        return {title_key(by_id[target]["title"], loose=loose) for target in identifiers}

    pack["audit"] = {
        "catalog_count": len(by_id), "old_sample_count": len(old_samples), "old_unique_targets": len(old_ids),
        "eligible_title_family_count": len(groups),
        "ineligible_title_count": excluded["ineligible_title_count"],
        "excluded_old_target_count": excluded["old_target_count"],
        "excluded_old_family_relative_count": excluded["old_family_relative_count"],
        "old_target_overlap": len((dev | validation) & old_ids),
        "old_exact_title_overlap": len(keys(dev | validation, False) & old_exact),
        "old_loose_title_overlap": len(keys(dev | validation, True) & old_loose),
        "cross_split_target_overlap": len(dev & validation),
        "cross_split_exact_title_overlap": len(keys(dev, False) & keys(validation, False)),
        "cross_split_loose_title_overlap": len(keys(dev, True) & keys(validation, True)),
    }
    if any(value for key, value in pack["audit"].items() if key.endswith("overlap")):
        raise ValueError("Unexpected target or title-family overlap")
    return pack


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def lock_pack(catalog: Path, old_dataset: Path, output: Path, seed: str = SEED) -> dict:
    pack = build_pack(_read_rows(catalog), _read_rows(old_dataset), seed)
    serialized = {name: "".join(json.dumps(row, sort_keys=True) + "\n" for row in pack[name]).encode()
                  for name in ("development", "validation")}
    manifest = {
        "version": VERSION, "seed": seed,
        "kind": "new catalog targets under the unchanged synthetic official simulator",
        "selection": "hash-order title families, one hash-chosen member, alternating split assignment",
        "family_method": "case/punctuation normalization; loose key removes listed colors and all-digit tokens",
        "family_limitation": "Title heuristic only; no manufacturer-family independence guarantee",
        "eligibility": "valid IDs/categories and nonempty normalized and loose title; exclude all old targets/families",
        "validation_policy": "Procedural lock, not encryption. Open outcomes only after source/config finalist freeze.",
        "validation_outcomes_accessed": False,
        "counts": {name: len(pack[name]) for name in serialized},
        "scenarios": {name: dict(Counter(row["scenario_type"] for row in pack[name])) for name in serialized},
        "audit": pack["audit"],
        "source_sha256": {"catalog": _digest(catalog), "old_dataset": _digest(old_dataset),
                          "preparation_script": _digest(Path(__file__)),
                          "official_evaluator": _digest(REPOSITORY / "evaluator/local_evaluator.py")},
        "dataset_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in serialized.items()},
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
    }
    manifest_path = output / "manifest.json"
    if output.exists():
        if not manifest_path.is_file():
            raise ValueError("Output already exists without a lock; refusing to overwrite")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if {key: value for key, value in previous.items() if key != "created_at_utc"} != manifest:
            raise ValueError("Existing lock differs from requested inputs or preparation source")
        for name, content in serialized.items():
            if (output / f"{name}.jsonl").read_bytes() != content:
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


def verify_lock(catalog: Path, old_dataset: Path, output: Path, preparation_source: Path | None = None) -> dict:
    """Verify locked bytes without regenerating samples or rewriting provenance."""
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("version") != VERSION:
        raise ValueError("Unsupported target-pack manifest")
    for key in ("source_sha256", "dataset_sha256"):
        if not isinstance(manifest.get(key), dict):
            raise ValueError(f"Invalid manifest {key}")
    if preparation_source is None:
        current_source = Path(__file__)
        if _digest(current_source) == manifest["source_sha256"].get("preparation_script"):
            preparation_source = current_source
        else:
            preparation_source = output.parent / "provenance/cycle2_prepare-original.py"
            if not preparation_source.is_file():
                raise ValueError("A preserved preparation_script source snapshot is required; current source differs")
    sources = {"catalog": catalog, "old_dataset": old_dataset,
               "preparation_script": preparation_source,
               "official_evaluator": REPOSITORY / "evaluator/local_evaluator.py"}
    for name, path in sources.items():
        if not path.is_file():
            raise ValueError(f"Missing {name} source snapshot: {path}")
        if _digest(path) != manifest["source_sha256"].get(name):
            raise ValueError(f"Locked {name} source hash mismatch")
    for name in ("development", "validation"):
        path = output / f"{name}.jsonl"
        if not path.is_file() or _digest(path) != manifest["dataset_sha256"].get(name):
            raise ValueError(f"Locked {name} data hash mismatch")
    current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip()
    return {
        "verified": True,
        "verification": "read-only locked data, source inputs, and original preparation source hashes",
        "manifest_sha256": _digest(manifest_path),
        "dataset_sha256": manifest["dataset_sha256"],
        "preparation_source_sha256": manifest["source_sha256"]["preparation_script"],
        "verification_source_sha256": _digest(Path(__file__)),
        "original_source_commit": manifest.get("source_commit"),
        "current_source_commit": current_commit,
        "commit_changed": current_commit != manifest.get("source_commit"),
        "outputs_rewritten": False,
        "validation_outcomes_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock fresh synthetic targets without evaluating them")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--old-dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/cycle2/synthetic-targets"))
    parser.add_argument("--seed", default=SEED)
    parser.add_argument("--verify-lock", action="store_true", help="Check recorded bytes without regenerating data")
    parser.add_argument("--preparation-source", type=Path,
                        help="Original generator snapshot; otherwise use preserved provenance or matching current source")
    args = parser.parse_args()
    if args.verify_lock:
        print(json.dumps(verify_lock(args.catalog, args.old_dataset, args.output, args.preparation_source), indent=2))
        return
    manifest = lock_pack(args.catalog, args.old_dataset, args.output, args.seed)
    print(json.dumps({key: manifest[key] for key in ("created_at_utc", "counts", "scenarios", "audit", "dataset_sha256")}, indent=2))


if __name__ == "__main__":
    main()
