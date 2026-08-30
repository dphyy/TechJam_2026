from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.cycle2_capabilities import run_capabilities
from mercury.model_assets import file_sha256


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPOSITORY / "data/private_like_capabilities.json"


def run_private_like(config: Path, output: Path, dataset: Path = DEFAULT_DATASET) -> dict:
    return run_capabilities(dataset, config, output, "development", provenance={
        "schema": "private-like-engineering-validation-v1",
        "dataset_sha256": file_sha256(dataset),
        "purpose": "Judge-visible engineering validation, not tuning data or private-test evidence.",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the private-like authored engineering validation pack.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_private_like(args.config, args.output, args.dataset), indent=2))


if __name__ == "__main__":
    main()
