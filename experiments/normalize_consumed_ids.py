from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize(inputs: list[Path], output: Path) -> dict:
    if not inputs:
        raise ValueError("At least one consumed dataset is required")
    if output.exists():
        raise ValueError("Output already exists; refusing to overwrite")
    names = [path.name for path in inputs]
    if len(names) != len(set(names)):
        raise ValueError("Consumed dataset basenames must be unique")
    output.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict[str, str | int]] = {}
    all_ids: set[str] = set()
    for source in inputs:
        rows = []
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            row = json.loads(line)
            identifier = row.get("sample_id")
            if not isinstance(identifier, str) or not identifier:
                raise ValueError(f"{source}:{number} has no valid sample_id")
            row["sample_id"] = f"{source.stem}__{identifier}"
            if row["sample_id"] in all_ids:
                raise ValueError("Normalized sample IDs are not unique")
            all_ids.add(row["sample_id"])
            rows.append(row)
        serialized = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
        ).encode()
        destination = output / source.name
        destination.write_bytes(serialized)
        source_bytes = source.read_bytes()
        files[source.name] = {
            "row_count": len(rows),
            "source_path": str(source.resolve()),
            "source_sha256": _digest(source_bytes),
            "normalized_sha256": _digest(serialized),
        }
    manifest = {
        "version": "consumed-sample-id-normalization-v1",
        "transformation": "prefix sample_id with source stem; preserve every other field",
        "files": files,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Give reused consumed-pack sample IDs unique provenance prefixes"
    )
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(normalize(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
