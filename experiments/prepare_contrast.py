from __future__ import annotations

import argparse
import json
from pathlib import Path

from mercury.catalog import Catalog
from mercury.contrast import write_contrasts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build grounded catalog-neighbor contrast evidence.")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/contrast"))
    args = parser.parse_args()
    print(json.dumps(write_contrasts(Catalog(args.catalog), args.output), indent=2))


if __name__ == "__main__":
    main()
