from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from mercury.model_assets import file_sha256

SEED = "20260826"


def partition_samples(samples: list[dict]) -> tuple[list[dict], list[dict]]:
    if len({row["sample_id"] for row in samples}) != len(samples):
        raise ValueError("Sample IDs must be unique")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in samples:
        groups[str(row["ground_truth"]["parent_asin"])].append(row)
    counts = Counter(row["scenario_type"] for row in samples)
    quotas = {scenario: max(1, round(count * 0.2)) for scenario, count in counts.items()}
    used: Counter = Counter()
    chosen: set[str] = set()
    for target in sorted(groups, key=lambda key: hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest()):
        group_counts = Counter(row["scenario_type"] for row in groups[target])
        if all(used[key] + value <= quotas[key] for key, value in group_counts.items()):
            chosen.add(target)
            used.update(group_counts)
    development, reserved = [], []
    for row in samples:
        (reserved if str(row["ground_truth"]["parent_asin"]) in chosen else development).append(row)
    if not development or not reserved:
        raise ValueError("Cannot construct nonempty target-disjoint development and reserved sets")
    return development, reserved


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze public-development partitions and source hashes")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="artifacts/splits")
    args = parser.parse_args()
    source = Path(args.dataset)
    samples = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    development, reserved = partition_samples(samples)
    output = Path(args.output)
    manifest_path = output / "manifest.json"
    files = [Path(args.catalog), source, Path("evaluator/local_evaluator.py"),
             Path("docs/evaluation_config.json"), Path("baselines/official.py")]
    fingerprints = {str(file): file_sha256(file) for file in files}
    manifest = {
        "seed": SEED,
        "source_sha256": fingerprints,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "development_ids": [r["sample_id"] for r in development],
        "reserved_ids": [r["sample_id"] for r in reserved],
        "development_scenarios": dict(Counter(r["scenario_type"] for r in development)),
        "reserved_scenarios": dict(Counter(r["scenario_type"] for r in reserved)),
        "python": platform.python_version(),
        "architecture": platform.machine(),
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in ("source_sha256", "seed", "development_ids", "reserved_ids"):
            if previous[key] != manifest[key]:
                raise ValueError(f"Frozen split mismatch for {key}; refusing to overwrite")
        for name, rows in (("development", development), ("reserved", reserved)):
            expected = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
            if (output / f"{name}.jsonl").read_text() != expected:
                raise ValueError(f"Frozen {name} file differs from source")
    else:
        output.mkdir(parents=True, exist_ok=True)
        for name, rows in (("development", development), ("reserved", reserved)):
            (output / f"{name}.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"development": len(development), "reserved": len(reserved),
                      "development_scenarios": manifest["development_scenarios"],
                      "reserved_scenarios": manifest["reserved_scenarios"],
                      "source_sha256": fingerprints}, indent=2))


if __name__ == "__main__":
    main()
