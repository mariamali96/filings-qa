"""Embedding backends.

`LocalEmbedder` runs a small open model on your machine (no API key, no per-call cost).
`HashEmbedder` is a deterministic bag-of-words stand-in used by the offline tests.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashEmbedder:
    """Hashing-trick bag of words. Not semantic; only for tests and offline demos."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class LocalEmbedder:
    """sentence-transformers model, e.g. BAAI/bge-small-en-v1.5 (384 dimensions)."""

    # bge models work best when queries (not documents) carry this instruction.
    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "Local embeddings need the optional extra: pip install -e '.[local-embeddings]'"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._is_bge = "bge" in model_name.lower()
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        prefix = self.QUERY_INSTRUCTION if self._is_bge else ""
        return self._model.encode(prefix + text, normalize_embeddings=True).tolist()


def make_embedder(kind: str, model_name: str) -> Embedder:
    if kind == "hash":
        return HashEmbedder()
    if kind == "local":
        return LocalEmbedder(model_name)
    raise ValueError(f"Unknown EMBEDDINGS={kind!r}; use 'local' or 'hash'.")
