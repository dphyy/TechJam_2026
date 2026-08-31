from __future__ import annotations

import hashlib
import logging
import os
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


def _normalized_text(value: str) -> str:
    return "\n".join(
        line for raw_line in str(value).splitlines() if (line := " ".join(raw_line.split()))
    )


def _response_prompt_tokens(response: object) -> int:
    usage = getattr(response, "usage", None)
    value = getattr(usage, "prompt_tokens", 0)
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
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
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.vectors_path = Path(vectors_path or self.catalog_path.with_name("catalog_embeddings.npy"))
        self.metadata_path = Path(
            metadata_path or self.catalog_path.with_name("catalog_embeddings.meta.json")
        )
        self.client = client
        self.model = DEFAULT_MODEL
        self.dimensions = DEFAULT_DIMENSIONS
        self.vectors = None
        self._cache: dict[str, object] = {}
        self._api_failed = False
        self._disabled_reason_logged = False
        self._load()

    @property
    def enabled(self) -> bool:
        return self.vectors is not None and not self._api_failed

    def close(self) -> None:
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
        try:
            import json

            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            required = {"model", "dimensions", "row_count", "catalog_sha256", "normalized"}
            if not required.issubset(metadata):
                raise ValueError("embedding metadata is incomplete")
            if not metadata["normalized"]:
                raise ValueError("catalog vectors are not normalized")
            if str(metadata["catalog_sha256"]).casefold() != catalog_sha256(
                self.catalog_path
            ).casefold():
                raise ValueError("catalog checksum does not match embedding metadata")

            vectors = np.load(self.vectors_path, mmap_mode="r", allow_pickle=False)
            dimensions = int(metadata["dimensions"])
            row_count = int(metadata["row_count"])
            if vectors.dtype != np.float32:
                raise ValueError("catalog vectors must use float32")
            if vectors.shape != (row_count, dimensions):
                raise ValueError("catalog vector shape does not match metadata")
            sample_count = min(row_count, 1024)
            if sample_count:
                sample_indexes = np.linspace(0, row_count - 1, sample_count, dtype=np.int64)
                norms = np.linalg.norm(vectors[sample_indexes], axis=1)
                if not np.all(np.isfinite(norms)) or not np.allclose(
                    norms, 1.0, rtol=1e-3, atol=1e-3
                ):
                    raise ValueError("catalog vectors are not L2-normalized")
            self.model = str(metadata["model"])
            self.dimensions = dimensions
            self.vectors = vectors
        except Exception as exc:  # A corrupt optional artifact cannot break FTS.
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
            self._api_failed = True
            LOGGER.warning("Vector retrieval disabled: %s", exc)
            return None

    def _embed_missing(self, texts: Sequence[str]) -> int:
        missing = [text for text in texts if text not in self._cache]
        if not missing:
            return 0
        client = self._ensure_client()
        if client is None or np is None:
            return 0
        try:
            response = client.embeddings.create(
                input=missing,
                model=self.model,
                dimensions=self.dimensions,
                encoding_format="float",
            )
            items = sorted(response.data, key=lambda item: int(item.index))
            if len(items) != len(missing):
                raise ValueError("embedding response length mismatch")
            for text, item in zip(missing, items):
                vector = np.asarray(item.embedding, dtype=np.float32)
                if vector.shape != (self.dimensions,):
                    raise ValueError("query embedding dimensions do not match catalog vectors")
                norm = float(np.linalg.norm(vector))
                if not np.isfinite(norm) or norm <= 0.0:
                    raise ValueError("query embedding has an invalid norm")
                self._cache[text] = vector / norm
            return _response_prompt_tokens(response)
        except Exception as exc:
            self._api_failed = True
            LOGGER.warning("Vector retrieval disabled after embedding request failed: %s", exc)
            return 0

    def search(
        self,
        structured_query: str | None,
        limit: int = DEFAULT_VECTOR_LIMIT,
    ) -> VectorSearchResult:
        if not self.enabled or np is None or not structured_query:
            return VectorSearchResult(rows=[])

        query_text = _normalized_text(structured_query)
        if not query_text:
            return VectorSearchResult(rows=[])

        prompt_tokens = self._embed_missing([query_text])
        if not self.enabled or query_text not in self._cache:
            return VectorSearchResult(rows=[], prompt_tokens=prompt_tokens)

        query = self._cache[query_text]

        scores = np.asarray(self.vectors @ query)
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
