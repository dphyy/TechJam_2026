"""Create a one-time source/config freeze before reserved evaluation."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from experiments.run import source_hashes
from mercury.config import Config
from mercury.model_assets import file_sha256


def freeze_configs(config_paths: list[Path], reserved: Path, output: Path, reason: str) -> dict:
    if not 1 <= len(config_paths) <= 2:
        raise ValueError("Freeze one or two development-selected finalists")
    configs = [Config.load(path).to_dict() for path in config_paths]
    if len({json.dumps(value, sort_keys=True) for value in configs}) != len(configs):
        raise ValueError("Finalists must be distinct configurations")
    if not reason.strip():
        raise ValueError("Record the development-only selection reason")
    manifest = {
        "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "reason": reason, "configs": configs,
        "config_paths": [str(path) for path in config_paths],
        "reserved_sha256": file_sha256(reserved), "source_hashes": source_hashes(),
        "protocol": "docs/EXPERIMENT_PROTOCOL.md", "maximum_finalists": 2,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", type=Path, required=True)
    parser.add_argument("--reserved", type=Path, default=Path("artifacts/splits/reserved.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/finalists.json"))
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    manifest = freeze_configs(args.config, args.reserved, args.output, args.reason)
    print(json.dumps({"output": str(args.output), "config_count": len(manifest["configs"]),
                      "reserved_sha256": manifest["reserved_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
