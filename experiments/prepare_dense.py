from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np

from mercury.catalog import Catalog
from mercury.model_assets import MODELS, file_sha256
from mercury.neural import DOCUMENT_VERSION, DenseIndex, document_text, load_encoder, validate_vectors


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a catalog-wide, versioned dense view using pinned local weights.")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    started = time.perf_counter()
    catalog = Catalog(args.catalog)
    destination = args.artifacts / "dense"
    if destination.exists():
        DenseIndex(catalog, args.artifacts, args.device, args.threads)
        print("Existing dense index verified; no files changed.")
        return
    encoder = load_encoder(args.artifacts / "models" / "embedding", args.device, args.threads)
    vectors = encoder.encode([document_text(product) for product in catalog.products],
                             batch_size=args.batch_size, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=True)
    vectors = np.asarray(vectors, dtype=np.float32)
    validate_vectors(vectors, len(catalog.products))
    destination.mkdir(parents=True)
    np.save(destination / "vectors.npy", vectors, allow_pickle=False)
    (destination / "ids.json").write_text(json.dumps([product.parent_asin for product in catalog.products]))
    manifest = {
        "catalog_sha256": catalog.sha256, "model_revision": MODELS["embedding"]["revision"],
        "document_version": DOCUMENT_VERSION, "count": len(catalog.products), "dimensions": 384,
        "device": args.device, "threads": args.threads, "batch_size": args.batch_size,
        "python": platform.python_version(), "build_seconds": time.perf_counter() - started,
        "max_rss_native": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "sha256": {name: file_sha256(destination / name) for name in ("vectors.npy", "ids.json")},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
