from __future__ import annotations

import json
import hashlib
import math
import os
import re
from collections import OrderedDict
from pathlib import Path
from threading import RLock

from mercury.catalog import Catalog
from mercury.model_assets import MODELS, file_sha256, verify_model
from mercury.product_types import classify_product
from mercury.types import Candidate, Preference, Product


DOCUMENT_VERSION = "fields-v1-256"
STRUCTURED_DOCUMENT_VERSION = "structured-fields-v1-256"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_LENGTH = 256
DOCUMENT_MODES = frozenset({"head", "lexical", "protected"})
PROTECTED_DOCUMENT_WORD_LIMIT = 160
_NEURAL_SCORE_KEYS = frozenset({
    "neural_logit", "neural_rank", "neural_margin", "neural_fusion_weight",
})


def _head_document_text(product: Product) -> str:
    limits = {"title": 500, "categories": 400, "features": 1200,
              "details": 800, "description": 1200, "store": 200}
    return "\n".join(f"{field.title()}: {product.fields[field][:limit]}"
                     for field, limit in limits.items() if product.fields[field])


def structured_document_text(product: Product) -> str:
    """Priority-ordered product evidence for cross-encoder truncation."""
    product_type = classify_product(product)
    lines = []
    if product_type.object_type is not None:
        lines.append(f"Product type: {product_type.object_type}")
    if product_type.role != "unknown":
        lines.append(f"Product role: {product_type.role}")
    limits = {"title": 500, "categories": 500, "features": 1000,
              "details": 700, "description": 900, "store": 150}
    lines.extend(f"{field.title()}: {product.fields[field][:limit]}"
                 for field, limit in limits.items() if product.fields[field])
    if product.price is not None:
        lines.insert(min(4, len(lines)), f"Price: {product.price:g}")
    return "\n".join(lines)


def _window(text: str, start: int, end: int, radius: int = 10) -> str:
    tokens = list(re.finditer(r"\S+", text))
    if not tokens:
        return ""
    first = next((index for index, token in enumerate(tokens) if token.end() > start), 0)
    last = next((index for index, token in enumerate(tokens) if token.start() >= end), len(tokens))
    return " ".join(token.group() for token in tokens[max(0, first - radius):min(len(tokens), last + radius)])


def _matching_windows(product: Product, values: list[str], fields: tuple[str, ...]) -> list[str]:
    snippets = []
    seen = set()
    for field in fields:
        source = product.fields[field]
        for value in values:
            normalized = value.strip()
            if not normalized:
                continue
            pattern = re.compile(r"(?<!\w)" + re.escape(normalized).replace(r"\ ", r"\s+") + r"(?!\w)", re.I)
            for match in pattern.finditer(source):
                snippet = _window(source, match.start(), match.end())
                key = (field, snippet.casefold())
                if snippet and key not in seen:
                    snippets.append(f"{field.title()}: {snippet}")
                    seen.add(key)
                break
    return snippets


def _assemble_document(product: Product, snippets: list[str]) -> str:
    parts = [f"Title: {product.fields['title']}", f"Categories: {product.fields['categories']}"]
    words: list[str] = []
    for part in parts + snippets:
        remaining = PROTECTED_DOCUMENT_WORD_LIMIT - len(words)
        if remaining <= 0:
            break
        words.extend(part.split()[:remaining])
    return " ".join(words)


def _lexical_document_text(product: Product, query: str) -> str:
    values = list(dict.fromkeys(token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) > 1))
    return _assemble_document(product, _matching_windows(product, values, ("features", "details", "description", "store")))


def _protected_document_text(product: Product, preferences: list[Preference]) -> str:
    ordered = sorted(
        (preference for preference in preferences
         if preference.active and preference.polarity != 0 and preference.attribute != "budget"),
        key=lambda preference: (not preference.hard, preference.source_turn, preference.attribute, preference.value,
                                preference.polarity),
    )
    snippets = []
    seen = set()
    for preference in ordered:
        for snippet in _matching_windows(product, [preference.value], ("features", "details", "description", "store")):
            if snippet.casefold() not in seen:
                snippets.append(snippet)
                seen.add(snippet.casefold())
    return _assemble_document(product, snippets)


def document_text(product: Product, query: str = "", preferences: list[Preference] | None = None, mode: str = "head") -> str:
    """Serialize only ordinary source text under a fixed reranker-pair budget."""
    if mode not in DOCUMENT_MODES:
        raise ValueError(f"Unsupported reranker document mode: {mode!r}")
    if mode == "head":
        return _head_document_text(product)
    if mode == "lexical":
        return _lexical_document_text(product, query)
    return _protected_document_text(product, preferences or [])


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


def fuse_neural_logits(
    candidates: list[Candidate], logits: dict[str, float], weight: float,
    *, low_margin_weight: float | None = None, margin_threshold: float = 0.0,
) -> list[Candidate]:
    """Fuse any scored candidate subset while keeping every unscored item below it.

    The ordinary first pass supplies a prefix. Progressive reranking can later
    add another batch from the same fixed candidate pool; raw logits are
    comparable because the query, model, and document contract are unchanged.
    """
    identifiers = [item.product.parent_asin for item in candidates]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Neural fusion requires unique candidate identifiers")
    if not set(logits).issubset(identifiers) or any(
            type(value) not in (int, float) or not math.isfinite(value) for value in logits.values()):
        raise ValueError("Neural fusion received invalid logits")
    scored = [(index, item) for index, item in enumerate(candidates)
              if item.product.parent_asin in logits]
    if not scored:
        return list(candidates)
    neural_order = sorted(
        scored, key=lambda pair: (-logits[pair[1].product.parent_asin], pair[0]),
    )
    neural_ranks = {
        item.product.parent_asin: rank for rank, (_, item) in enumerate(neural_order, 1)
    }
    ordered_logits = sorted((float(logits[item.product.parent_asin]) for _, item in scored), reverse=True)
    margin = ordered_logits[0] - ordered_logits[1] if len(ordered_logits) > 1 else float("inf")
    applied_weight = (
        low_margin_weight
        if low_margin_weight is not None and margin < margin_threshold
        else weight
    )
    result = []
    for index, item in scored:
        identifier = item.product.parent_asin
        neural_rank = neural_ranks[identifier]
        score = ((1.0 - applied_weight) * 61.0 / (61.0 + index)
                 + applied_weight * 61.0 / (60.0 + neural_rank))
        parts = {key: value for key, value in item.route_scores.items()
                 if key != "constraint_penalty" and key not in _NEURAL_SCORE_KEYS}
        result.append(Candidate(item.product, score, {
            **parts,
            "neural_logit": float(logits[identifier]),
            "neural_rank": float(neural_rank),
            "neural_margin": margin,
            "neural_fusion_weight": float(applied_weight),
        }))
    result.sort(key=lambda item: (-item.score, item.product.parent_asin))
    floor = result[-1].score
    scored_count = len(result)
    unscored = [item for item in candidates if item.product.parent_asin not in logits]
    for rank, item in enumerate(unscored, scored_count + 1):
        score = floor * (60.0 + scored_count) / (60.0 + rank)
        parts = {key: value for key, value in item.route_scores.items()
                 if key != "constraint_penalty" and key not in _NEURAL_SCORE_KEYS}
        result.append(Candidate(item.product, score, parts))
    return result


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
    def __init__(self, artifact_dir: Path, device: str = "cpu", threads: int = 4,
                 kind: str = "reranker", cache_capacity: int = 0, batch_size: int = 16):
        if type(cache_capacity) is not int or cache_capacity < 0:
            raise ValueError("cache_capacity must be a nonnegative integer")
        if type(batch_size) is not int or batch_size not in {16, 30, 32}:
            raise ValueError("batch_size must be one of 16, 30, or 32")
        path = artifact_dir / "models" / kind
        verify_model(path, kind)
        options = _model_options(device, threads)
        from sentence_transformers import CrossEncoder
        import torch

        self.model = CrossEncoder(str(path), max_length=MAX_LENGTH,
                                  activation_fn=torch.nn.Identity(), **options)
        self.kind = kind
        self.device = device
        self.threads = threads
        self.batch_size = batch_size
        self.prompt_tokens = 0
        self._cache_capacity = cache_capacity
        self._logit_cache: OrderedDict[tuple, float] = OrderedDict()
        self._cache_lock = RLock()
        self._score_lock = RLock()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0
        self._evaluated_pairs = 0
        self._last_prediction_token_lengths: list[int] = []
        self._last_pair_receipts: list[dict] = []

    def pair_receipts(self) -> list[dict]:
        """Return bounded non-label serialization evidence for the latest call."""
        return [dict(row) for row in self._last_pair_receipts]

    def cache_stats(self) -> dict[str, int | bool]:
        """Return a consistent snapshot without exposing cached query evidence."""
        lock = getattr(self, "_cache_lock", None)
        if lock is None:
            return {
                "enabled": False, "capacity": 0, "size": 0, "hits": 0,
                "misses": 0, "evictions": 0, "evaluated_pairs": 0,
            }
        with lock:
            return {
                "enabled": self._cache_capacity > 0,
                "capacity": self._cache_capacity,
                "size": len(self._logit_cache),
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "evictions": self._cache_evictions,
                "evaluated_pairs": self._evaluated_pairs,
            }

    def _pair_cache_key(self, effective_query: str, document: str, candidate: Candidate,
                        document_mode: str, structured: bool) -> tuple:
        kind = self.kind
        serializer = (
            STRUCTURED_DOCUMENT_VERSION if structured
            else f"{DOCUMENT_VERSION}:{document_mode}"
        )
        return (
            kind,
            MODELS[kind]["revision"],
            getattr(self, "device", "cpu"),
            getattr(self, "threads", 4),
            getattr(self, "batch_size", 16),
            serializer,
            hashlib.sha256(effective_query.encode("utf-8")).digest(),
            candidate.product.parent_asin,
            hashlib.sha256(document.encode("utf-8")).digest(),
            MAX_LENGTH,
        )

    def _predict_logits(self, effective_query: str, documents: list[str]) -> list[float]:
        import numpy as np

        pairs = [(effective_query, document) for document in documents]
        tokens = self.model.tokenizer(
            [pair[0] for pair in pairs], [pair[1] for pair in pairs],
            truncation=True, max_length=MAX_LENGTH,
        )
        self._last_prediction_token_lengths = list(map(len, tokens["input_ids"]))
        self.prompt_tokens += sum(self._last_prediction_token_lengths)
        logits = np.asarray(self.model.predict(
            pairs, batch_size=getattr(self, "batch_size", 16),
            convert_to_numpy=True, show_progress_bar=False,
        )).reshape(-1)
        if len(logits) != len(documents) or not np.isfinite(logits).all():
            raise ValueError("Reranker returned malformed or non-finite scores")
        return [float(logit) for logit in logits]

    def _score_unlocked(self, query: str, candidates: list[Candidate],
                        preferences: list[Preference] | None, document_mode: str,
                        structured: bool) -> dict[str, float]:
        effective_query = query[:2000]
        documents = [
            structured_document_text(item.product) if structured else
            document_text(item.product, query, preferences, document_mode)
            for item in candidates
        ]
        receipts = [
            {
                "product_id": item.product.parent_asin,
                "serialized_characters": len(document),
                "populated_fields": [
                    field for field in ("title", "categories", "features", "details", "description", "store")
                    if item.product.fields.get(field, "").strip()
                ],
                "token_count": None,
            }
            for item, document in zip(candidates, documents, strict=True)
        ]
        if getattr(self, "_cache_capacity", 0) <= 0:
            cache_lock = getattr(self, "_cache_lock", None)
            if cache_lock is not None:
                with cache_lock:
                    self._evaluated_pairs += len(candidates)
            logits = self._predict_logits(effective_query, documents)
            for receipt, token_count in zip(
                    receipts, self._last_prediction_token_lengths, strict=True):
                receipt["token_count"] = token_count
            self._last_pair_receipts = receipts
            return {
                item.product.parent_asin: logit
                for item, logit in zip(candidates, logits, strict=True)
            }

        keys = [
            self._pair_cache_key(effective_query, document, item, document_mode, structured)
            for item, document in zip(candidates, documents, strict=True)
        ]
        restored: list[float | None] = [None] * len(candidates)
        missing_indices = []
        with self._cache_lock:
            for index, key in enumerate(keys):
                if key in self._logit_cache:
                    restored[index] = self._logit_cache[key]
                    self._logit_cache.move_to_end(key)
                    self._cache_hits += 1
                else:
                    missing_indices.append(index)
                    self._cache_misses += 1

        if missing_indices:
            with self._cache_lock:
                self._evaluated_pairs += len(missing_indices)
            fresh = self._predict_logits(
                effective_query, [documents[index] for index in missing_indices],
            )
            for index, token_count in zip(
                    missing_indices, self._last_prediction_token_lengths, strict=True):
                receipts[index]["token_count"] = token_count
            with self._cache_lock:
                for index, logit in zip(missing_indices, fresh, strict=True):
                    key = keys[index]
                    restored[index] = logit
                    self._logit_cache[key] = logit
                    self._logit_cache.move_to_end(key)
                    while len(self._logit_cache) > self._cache_capacity:
                        self._logit_cache.popitem(last=False)
                        self._cache_evictions += 1
        if any(logit is None for logit in restored):
            raise RuntimeError("Neural cache failed to restore candidate scores")
        self._last_pair_receipts = receipts
        return {
            item.product.parent_asin: float(logit)
            for item, logit in zip(candidates, restored, strict=True)
        }

    def score(self, query: str, candidates: list[Candidate],
              preferences: list[Preference] | None = None, document_mode: str = "head",
              *, structured: bool = False) -> dict[str, float]:
        if not candidates:
            return {}
        lock = getattr(self, "_score_lock", None)
        if lock is None:
            return self._score_unlocked(
                query, candidates, preferences, document_mode, structured,
            )
        with lock:
            return self._score_unlocked(
                query, candidates, preferences, document_mode, structured,
            )

    def rank(self, query: str, candidates: list[Candidate], limit: int, weight: float,
             preferences: list[Preference] | None = None, document_mode: str = "head",
             *, structured: bool = False, low_margin_weight: float | None = None,
             margin_threshold: float = 0.0) -> list[Candidate]:
        prefix = candidates[:limit]
        logits = self.score(
            query, prefix, preferences, document_mode, structured=structured,
        )
        return fuse_neural_logits(
            candidates, logits, weight,
            low_margin_weight=low_margin_weight, margin_threshold=margin_threshold,
        )
