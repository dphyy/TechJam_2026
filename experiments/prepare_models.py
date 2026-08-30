from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mercury.model_assets import MODELS, file_sha256, verify_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire explicitly pinned public model assets")
    parser.add_argument("--output", default="artifacts/models")
    parser.add_argument("--model", choices=("all", "embedding", "reranker"), default="all")
    parser.add_argument("--download", action="store_true", help="Allow public model downloads")
    args = parser.parse_args()
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    for kind, spec in MODELS.items():
        if args.model != "all" and kind != args.model:
            continue
        destination = Path(args.output) / kind
        if (destination / "asset_manifest.json").exists():
            manifest = verify_model(destination, kind)
        else:
            if not args.download:
                raise FileNotFoundError(f"{kind} is not prepared; use --download to acquire it")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=spec["repo_id"], revision=spec["revision"],
                local_dir=destination, token=False,
                allow_patterns=["model.safetensors", "*.json", "vocab.txt", "README.md",
                                "1_Pooling/config.json"],
                ignore_patterns=["onnx/*", "openvino/*"],
            )
            checksums = {
                str(file.relative_to(destination)): file_sha256(file)
                for file in sorted(destination.rglob("*"))
                if file.is_file() and ".cache" not in file.parts
                and file.name != "asset_manifest.json"
            }
            if checksums.get("model.safetensors") != spec["weights_sha256"]:
                raise ValueError(f"Downloaded {kind} weights do not match the pinned checksum")
            manifest = {"repo_id": spec["repo_id"], "revision": spec["revision"],
                        "license": spec["license"], "files": checksums}
            (destination / "asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            verify_model(destination, kind)
        print(json.dumps({"kind": kind, "revision": manifest["revision"],
                          "path": str(destination), "verified": True}), flush=True)


if __name__ == "__main__":
    main()
