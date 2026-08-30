from __future__ import annotations

import json
import os
from pathlib import Path

from mercury.catalog import Catalog
from mercury.model_assets import MODELS, file_sha256, verify_model
from mercury.types import Candidate, Product


DOCUMENT_VERSION = "fields-v1-256"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_LENGTH = 256


def document_text(product: Product) -> str:
    limits = {"title": 500, "categories": 400, "features": 1200,
              "details": 800, "description": 1200, "store": 200}
    return "\n".join(f"{field.title()}: {product.fields[field][:limit]}"
                     for field, limit in limits.items() if product.fields[field])


def validate_dense_manifest(manifest: dict, catalog_hash: str, count: int) -> None:
    expected = {"catalog_sha256": catalog_hash,
                "model_revision": MODELS["embedding"]["revision"],
                "document_version": DOCUMENT_VERSION, "count": count, "dimensions": 384}
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"Dense index mismatch: {field}")


def validate_vectors(vectors, count: int) -> None:
    import numpy as np

    if vectors.shape != (count, 384) or vectors.dtype != np.float32:
        raise ValueError("Dense index has invalid shape or dtype")
    if not np.isfinite(vectors).all():
        raise ValueError("Dense index contains non-finite values")


def _model_options(device: str, threads: int) -> dict:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch

    torch.set_num_threads(threads)
    return {"device": device, "local_files_only": True, "trust_remote_code": False,
            "model_kwargs": {"use_safetensors": True, "dtype": torch.float32}}


def load_encoder(path: Path, device: str = "cpu", threads: int = 4):
    verify_model(path, "embedding")
    options = _model_options(device, threads)
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(str(path), **options)
    encoder.max_seq_length = MAX_LENGTH
    return encoder


class DenseIndex:
    def __init__(self, catalog: Catalog, artifact_dir: Path, device: str = "cpu", threads: int = 4):
        import numpy as np

        root = artifact_dir / "dense"
        manifest = json.loads((root / "manifest.json").read_text())
        validate_dense_manifest(manifest, catalog.sha256, len(catalog.products))
        for name in ("vectors.npy", "ids.json"):
            if file_sha256(root / name) != manifest["sha256"][name]:
                raise ValueError(f"Dense index file hash mismatch: {name}")
        self.ids = json.loads((root / "ids.json").read_text())
        if self.ids != [product.parent_asin for product in catalog.products]:
            raise ValueError("Dense index IDs do not match catalog order")
        self.vectors = np.load(root / "vectors.npy", mmap_mode="r", allow_pickle=False)
        validate_vectors(self.vectors, len(self.ids))
        self.encoder = load_encoder(artifact_dir / "models" / "embedding", device, threads)
        self.prompt_tokens = 0

    def search(self, query: str, limit: int) -> list[str]:
        import numpy as np

        if not query.strip() or limit <= 0:
            return []
        text = QUERY_PREFIX + query[:2000]
        tokens = self.encoder.tokenizer(text, truncation=True, max_length=MAX_LENGTH)
        self.prompt_tokens += len(tokens["input_ids"])
        vector = self.encoder.encode([text], normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False)[0]
        if not np.isfinite(vector).all():
            raise ValueError("Embedding model returned non-finite values")
        # Explicit contraction avoids spurious BLAS floating-point flags on the
        # tested arm64 runtime; validate the result rather than hiding warnings.
        scores = np.einsum("ij,j->i", self.vectors, vector, optimize=False)
        if not np.isfinite(scores).all():
            raise ValueError("Dense retrieval produced non-finite scores")
        # Stable ordering preserves identical rows as distinct catalog IDs.
        indices = np.argsort(-scores, kind="stable")[:limit]
        return [self.ids[int(index)] for index in indices]


class NeuralRanker:
    def __init__(self, artifact_dir: Path, device: str = "cpu", threads: int = 4):
        path = artifact_dir / "models" / "reranker"
        verify_model(path, "reranker")
        options = _model_options(device, threads)
        from sentence_transformers import CrossEncoder
        import torch

        self.model = CrossEncoder(str(path), max_length=MAX_LENGTH,
                                  activation_fn=torch.nn.Identity(), **options)
        self.prompt_tokens = 0

    def rank(self, query: str, candidates: list[Candidate], limit: int, weight: float) -> list[Candidate]:
        import numpy as np

        prefix, tail = candidates[:limit], candidates[limit:]
        if not prefix:
            return candidates
        pairs = [(query[:2000], document_text(item.product)) for item in prefix]
        tokens = self.model.tokenizer([pair[0] for pair in pairs], [pair[1] for pair in pairs],
                                      truncation=True, max_length=MAX_LENGTH)
        self.prompt_tokens += sum(map(len, tokens["input_ids"]))
        logits = np.asarray(self.model.predict(pairs, batch_size=16, convert_to_numpy=True,
                                               show_progress_bar=False)).reshape(-1)
        if len(logits) != len(prefix) or not np.isfinite(logits).all():
            raise ValueError("Reranker returned malformed or non-finite scores")
        neural_order = np.argsort(-logits, kind="stable")
        neural_ranks = {int(index): rank for rank, index in enumerate(neural_order, 1)}
        result = []
        for index, item in enumerate(prefix):
            score = (1.0 - weight) * 61.0 / (61.0 + index) + weight * 61.0 / (60.0 + neural_ranks[index])
            parts = {key: value for key, value in item.route_scores.items() if key != "constraint_penalty"}
            result.append(Candidate(item.product, score,
                                    {**parts, "neural_logit": float(logits[index]),
                                     "neural_rank": float(neural_ranks[index])}))
        result.sort(key=lambda item: (-item.score, item.product.parent_asin))
        # The unreranked tail must remain below the bounded prefix on the same
        # score scale; question lookahead consumes these scores downstream.
        floor = result[-1].score
        for index, item in enumerate(tail, len(prefix) + 1):
            score = floor * (60.0 + len(prefix)) / (60.0 + index)
            parts = {key: value for key, value in item.route_scores.items() if key != "constraint_penalty"}
            result.append(Candidate(item.product, score, parts))
        return result
