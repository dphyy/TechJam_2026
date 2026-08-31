from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

try:
    import numpy as np
except ImportError:  # The lexical fallback must still be importable.
    np = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIMENSIONS = 256
DEFAULT_VECTOR_LIMIT = 250
DEFAULT_QUERY_CACHE_SIZE = 256
MAX_QUERY_CHARACTERS = 8000


class EmbeddingsClient(Protocol):
    class _Embeddings(Protocol):
        def create(self, **kwargs: object) -> object: ...

    embeddings: _Embeddings


class VectorIndex(Protocol):
    def search(
        self,
        structured_query: str | None,
        limit: int = DEFAULT_VECTOR_LIMIT,
    ) -> VectorSearchResult: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class VectorSearchResult:
    rows: list[tuple[int, float]]
    prompt_tokens: int = 0


def catalog_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash catalog contents canonically so Git line endings do not change identity."""
    digest = hashlib.sha256()
    pending_carriage_return = False
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            if pending_carriage_return:
                chunk = b"\r" + chunk
                pending_carriage_return = False
            if chunk.endswith(b"\r"):
                chunk = chunk[:-1]
                pending_carriage_return = True
            digest.update(chunk.replace(b"\r\n", b"\n"))
    if pending_carriage_return:
        digest.update(b"\r")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def catalog_row_identity(path: Path) -> tuple[int, str]:
    """Bind each vector position to one unique catalog product ID."""
    digest = hashlib.sha256()
    identifiers: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            identifier = row.get("parent_asin") if isinstance(row, dict) else None
            if not isinstance(identifier, str) or not identifier.strip() or identifier in identifiers:
                raise ValueError("catalog product IDs must be unique nonempty strings")
            identifiers.add(identifier)
            digest.update(json.dumps(identifier, ensure_ascii=True).encode("utf-8") + b"\n")
    return len(identifiers), digest.hexdigest()


def _normalized_text(value: str) -> str:
    return "\n".join(
        line for raw_line in str(value).splitlines() if (line := " ".join(raw_line.split()))
    )


def _response_prompt_tokens(response: object) -> int:
    usage = getattr(response, "usage", None)
    value = getattr(usage, "prompt_tokens", 0)
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def load_openai_api_key() -> bool:
    """Load optional embedding credentials from the environment."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    if not os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_APIKEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_APIKEY"]
    return bool(os.environ.get("OPENAI_API_KEY"))


def create_openai_client() -> object:
    from openai import OpenAI

    if os.environ.get("OPENAI_SYSTEM_CA_COMPAT", "").casefold() not in {"1", "true", "yes"}:
        return OpenAI()

    import ssl

    import httpx

    context = ssl.create_default_context()
    context.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return OpenAI(http_client=httpx.Client(verify=context))


class CatalogVectorIndex:
    """Memory-mapped exact vector search with a fail-open lexical fallback."""

    def __init__(
        self,
        catalog_path: str | Path,
        *,
        vectors_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        client: EmbeddingsClient | None = None,
        cache_capacity: int = DEFAULT_QUERY_CACHE_SIZE,
    ) -> None:
        if type(cache_capacity) is not int or cache_capacity < 1:
            raise ValueError("cache_capacity must be a positive integer")
        self.catalog_path = Path(catalog_path)
        self.vectors_path = Path(vectors_path or self.catalog_path.with_name("catalog_embeddings.npy"))
        self.metadata_path = Path(
            metadata_path or self.catalog_path.with_name("catalog_embeddings.meta.json")
        )
        self.client = client
        self.model = DEFAULT_MODEL
        self.dimensions = DEFAULT_DIMENSIONS
        self.vectors = None
        self._cache: OrderedDict[str, object] = OrderedDict()
        self._cache_capacity = cache_capacity
        self._disabled_reason_logged = False
        self._load()

    @property
    def enabled(self) -> bool:
        return self.vectors is not None

    def close(self) -> None:
        self._cache.clear()
        vectors = self.vectors
        self.vectors = None
        memory_map = getattr(vectors, "_mmap", None)
        if memory_map is not None:
            memory_map.close()

    def __enter__(self) -> CatalogVectorIndex:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _disable(self, reason: str) -> None:
        self.close()
        if not self._disabled_reason_logged:
            LOGGER.warning("Vector retrieval disabled: %s", reason)
            self._disabled_reason_logged = True

    def _load(self) -> None:
        if np is None:
            self._disable("NumPy is unavailable")
            return
        if not self.vectors_path.exists() or not self.metadata_path.exists():
            self._disable("catalog embedding artifact is unavailable")
            return
        vectors = None
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            required = {
                "model", "dimensions", "row_count", "catalog_sha256", "normalized",
                "vectors_sha256", "product_ids_sha256",
            }
            if not isinstance(metadata, dict) or not required.issubset(metadata):
                raise ValueError("embedding metadata is incomplete")
            if metadata["normalized"] is not True:
                raise ValueError("catalog vectors are not normalized")
            if not isinstance(metadata["model"], str) or not metadata["model"].strip():
                raise ValueError("embedding model must be a nonempty string")
            dimensions, row_count = metadata["dimensions"], metadata["row_count"]
            if type(dimensions) is not int or dimensions < 1 or type(row_count) is not int or row_count < 0:
                raise ValueError("embedding dimensions and row count must be valid integers")
            if metadata["catalog_sha256"] != catalog_sha256(self.catalog_path):
                raise ValueError("catalog checksum does not match embedding metadata")
            actual_count, product_ids_sha256 = catalog_row_identity(self.catalog_path)
            if actual_count != row_count or metadata["product_ids_sha256"] != product_ids_sha256:
                raise ValueError("embedding product rows do not match the catalog")
            # These checks establish consistency with the manifest, not trust in its author.
            if metadata["vectors_sha256"] != file_sha256(self.vectors_path):
                raise ValueError("vector checksum does not match embedding metadata")

            vectors = np.load(self.vectors_path, mmap_mode="r", allow_pickle=False)
            if vectors.dtype != np.float32:
                raise ValueError("catalog vectors must use float32")
            if vectors.shape != (row_count, dimensions):
                raise ValueError("catalog vector shape does not match metadata")
            for start in range(0, row_count, 1024):
                chunk = vectors[start:start + 1024]
                if not np.all(np.isfinite(chunk)):
                    raise ValueError("catalog vectors contain non-finite values")
                norms = np.linalg.norm(chunk, axis=1)
                if not np.allclose(
                    norms, 1.0, rtol=1e-3, atol=1e-3
                ):
                    raise ValueError("catalog vectors are not L2-normalized")
            self.model = metadata["model"]
            self.dimensions = dimensions
            self.vectors = vectors
        except Exception as exc:  # A corrupt optional artifact cannot break FTS.
            memory_map = getattr(vectors, "_mmap", None)
            if memory_map is not None:
                memory_map.close()
            self._disable(str(exc))

    def _ensure_client(self) -> EmbeddingsClient | None:
        if self.client is not None:
            return self.client
        try:
            if not load_openai_api_key():
                raise RuntimeError("OPENAI_API_KEY is unavailable")
            self.client = create_openai_client()
            return self.client
        except Exception as exc:
            LOGGER.warning("Vector retrieval unavailable for this call: %s", type(exc).__name__)
            return None

    def _embed_missing(self, texts: Sequence[str]) -> int:
        missing = list(dict.fromkeys(text for text in texts if text not in self._cache))
        if not missing:
            return 0
        client = self._ensure_client()
        if client is None or np is None:
            return 0
        prompt_tokens = 0
        try:
            response = client.embeddings.create(
                input=missing,
                model=self.model,
                dimensions=self.dimensions,
                encoding_format="float",
            )
            prompt_tokens = _response_prompt_tokens(response)
            items = list(response.data)
            if (len(items) != len(missing)
                    or any(type(item.index) is not int for item in items)
                    or {item.index for item in items} != set(range(len(missing)))):
                raise ValueError("embedding response indices do not match the request")
            items.sort(key=lambda item: item.index)
            fresh = {}
            for text, item in zip(missing, items):
                vector = np.asarray(item.embedding, dtype=np.float32)
                if vector.shape != (self.dimensions,):
                    raise ValueError("query embedding dimensions do not match catalog vectors")
                norm = float(np.linalg.norm(vector))
                if not np.isfinite(norm) or norm <= 0.0:
                    raise ValueError("query embedding has an invalid norm")
                fresh[text] = vector / norm
            for text, vector in fresh.items():
                self._cache[text] = vector
                self._cache.move_to_end(text)
                while len(self._cache) > self._cache_capacity:
                    self._cache.popitem(last=False)
            return prompt_tokens
        except Exception as exc:
            LOGGER.warning("Vector retrieval failed for this call: %s", type(exc).__name__)
            return prompt_tokens

    def search(
        self,
        structured_query: str | None,
        limit: int = DEFAULT_VECTOR_LIMIT,
    ) -> VectorSearchResult:
        if not self.enabled or np is None or not structured_query:
            return VectorSearchResult(rows=[])

        query_text = _normalized_text(structured_query)[:MAX_QUERY_CHARACTERS]
        if not query_text:
            return VectorSearchResult(rows=[])

        prompt_tokens = self._embed_missing([query_text])
        if not self.enabled or query_text not in self._cache:
            return VectorSearchResult(rows=[], prompt_tokens=prompt_tokens)

        query = self._cache[query_text]
        self._cache.move_to_end(query_text)

        scores = np.asarray(self.vectors @ query)
        if not np.all(np.isfinite(scores)):
            self._disable("vector search produced non-finite scores")
            return VectorSearchResult(rows=[], prompt_tokens=prompt_tokens)
        count = min(max(0, int(limit)), len(scores))
        if count == 0:
            return VectorSearchResult(rows=[], prompt_tokens=prompt_tokens)
        if count == len(scores):
            indexes = np.arange(len(scores))
        else:
            indexes = np.argpartition(scores, len(scores) - count)[-count:]
        indexes = indexes[np.argsort(scores[indexes], kind="stable")[::-1]]
        rows = [(int(index) + 1, float(scores[index])) for index in indexes]
        return VectorSearchResult(rows=rows, prompt_tokens=prompt_tokens)
