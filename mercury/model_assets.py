from __future__ import annotations

import hashlib
import json
from pathlib import Path


MODELS = {
    "embedding": {
        "repo_id": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "weights_sha256": "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad",
        "metadata_sha256": "b4b9461186c74ddbad9653357d78c73b12faa9b8e61e136759b353b546cd6316",
        "license": "MIT",
        "required": ["model.safetensors", "config.json", "tokenizer_config.json",
                     "modules.json", "1_Pooling/config.json"],
    },
    "reranker": {
        "repo_id": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
        "weights_sha256": "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae",
        "metadata_sha256": "ed466f09238356bd21a5b860b96cba763359329d42e8bd73338684a381143b66",
        "license": "Apache-2.0",
        "required": ["model.safetensors", "config.json", "tokenizer_config.json"],
    },
    "reranker_domain_v1": {
        "repo_id": "local/product-domain-ms-marco-MiniLM-L6-v2",
        "revision": "mercury-product-domain-minilm-v1-seed-20260830",
        "weights_sha256": "8a0fe765eb7925ea38c820a140f8eb518c3659b3596f0df1a309e70feffb4f05",
        "license": "Apache-2.0",
        "required": ["model.safetensors", "config.json", "tokenizer_config.json"],
    },
    "bge_reranker_base": {
        "repo_id": "BAAI/bge-reranker-base",
        "revision": "2cfc18c9415c912f9d8155881c133215df768a70",
        "weights_sha256": "ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd",
        "license": "MIT",
        "required": ["model.safetensors", "config.json", "tokenizer_config.json"],
    },
}

# Cross-encoder kinds selectable by configuration; the default stays first.
RERANKERS = ("reranker", "reranker_domain_v1", "bge_reranker_base")

_LOADER_SUFFIXES = (".json", ".txt", ".model")


def _mapping_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def model_asset_identity(manifest: dict) -> str:
    """Identity of verified inference inputs, independent of documentation files."""
    return _mapping_digest({
        "revision": manifest.get("revision"),
        "files": {name: checksum for name, checksum in manifest.get("files", {}).items()
                  if name.endswith((*_LOADER_SUFFIXES, ".safetensors"))},
    })


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(path: Path, kind: str) -> dict:
    spec = MODELS[kind]
    for name in spec["required"]:
        if not (path / name).is_file():
            raise FileNotFoundError(f"Missing offline {kind} asset: {name}")
    if file_sha256(path / "model.safetensors") != spec["weights_sha256"]:
        raise ValueError(f"Unexpected {kind} weights checksum")
    manifest = json.loads((path / "asset_manifest.json").read_text())
    if not isinstance(manifest, dict) or manifest.get("revision") != spec["revision"]:
        raise ValueError(f"Unexpected {kind} model revision")
    files = manifest.get("files")
    if not isinstance(files, dict) or not set(spec["required"]) <= files.keys():
        raise ValueError("Model manifest must hash every required asset")
    if not {"tokenizer.json", "vocab.txt"} & files.keys():
        raise ValueError("Model manifest is missing tokenizer vocabulary")
    for name, checksum in files.items():
        if (not isinstance(name, str) or not isinstance(checksum, str) or len(checksum) != 64
                or any(char not in "0123456789abcdef" for char in checksum)):
            raise ValueError("Invalid model file checksum entry")
        if (not name or Path(name).is_absolute() or Path(name).as_posix() != name
                or ".." in Path(name).parts or "\\" in name or name == "asset_manifest.json"):
            raise ValueError("Invalid model manifest path")
        target = (path / name).resolve()
        if not target.is_relative_to(path.resolve()):
            raise ValueError("Invalid model manifest path")
        if file_sha256(target) != checksum:
            raise ValueError(f"Changed {kind} model metadata: {name}")
    metadata = {name: checksum for name, checksum in files.items() if name.endswith(_LOADER_SUFFIXES)}
    observed_metadata = {
        file.relative_to(path).as_posix() for file in path.rglob("*")
        if file.is_file() and file.name.endswith(_LOADER_SUFFIXES)
        and file.name != "asset_manifest.json" and ".cache" not in file.relative_to(path).parts
    }
    if observed_metadata != metadata.keys():
        raise ValueError("Model loader metadata differs from manifest")
    expected_metadata = spec.get("metadata_sha256")
    if expected_metadata is not None and _mapping_digest(metadata) != expected_metadata:
        raise ValueError(f"Unexpected {kind} loader metadata checksum")
    return manifest
