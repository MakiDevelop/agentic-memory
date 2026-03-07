"""Embedding providers for vector-based semantic search."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Protocol, Sequence

import numpy as np


class EmbeddingProvider(Protocol):
    """Abstract embedding interface for pluggable vector generation."""

    @property
    def model_id(self) -> str:
        """Stable identifier used for persistence / compatibility checks."""
        ...

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        ...

    def fit(self, texts: Sequence[str]) -> None:
        """Optional training step. No-op for pretrained models."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        """Return L2-normalized vectors for stored documents."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Return an L2-normalized vector for a search query."""
        ...

    def dumps(self) -> bytes:
        """Serialize provider state for persistence."""
        ...

    @classmethod
    def loads(cls, payload: bytes) -> EmbeddingProvider:
        """Restore provider state from persistence payload."""
        ...


class TFIDFEmbedding:
    """Pure-numpy TF-IDF embedding. Zero external dependencies beyond numpy."""

    def __init__(
        self,
        *,
        lowercase: bool = True,
        token_pattern: str = r"\b\w+\b",
        max_features: int = 8192,
        min_df: int = 1,
    ) -> None:
        self.lowercase = lowercase
        self._token_pattern_str = token_pattern
        self._token_re = re.compile(token_pattern)
        self.max_features = max_features
        self.min_df = min_df
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray = np.empty(0, dtype=np.float32)

    @property
    def model_id(self) -> str:
        return "tfidf-v1"

    @property
    def dim(self) -> int:
        return len(self._vocab)

    def fit(self, texts: Sequence[str]) -> None:
        """Build vocabulary and IDF weights from a corpus."""
        doc_freq: Counter[str] = Counter()

        for text in texts:
            tokens = self._tokenize(text)
            doc_freq.update(set(tokens))

        filtered = [(term, df) for term, df in doc_freq.items() if df >= self.min_df]
        filtered.sort(key=lambda item: (-item[1], item[0]))
        filtered = filtered[: self.max_features]

        self._vocab = {term: idx for idx, (term, _) in enumerate(filtered)}

        n_docs = max(len(texts), 1)
        idf = np.ones(len(self._vocab), dtype=np.float32)
        for term, idx in self._vocab.items():
            df = doc_freq[term]
            idf[idx] = math.log((1 + n_docs) / (1 + df)) + 1.0
        self._idf = idf

    def embed_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        self._ensure_fitted()
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> np.ndarray:
        self._ensure_fitted()
        return self._embed_text(text)

    def dumps(self) -> bytes:
        """Serialize state using JSON (vocab) + raw numpy bytes (idf)."""
        meta = json.dumps({
            "lowercase": self.lowercase,
            "token_pattern": self._token_pattern_str,
            "max_features": self.max_features,
            "min_df": self.min_df,
            "vocab": self._vocab,
        }).encode()
        idf_bytes = self._idf.tobytes()
        # Format: 4-byte meta length (big-endian) + meta JSON + idf raw bytes
        return len(meta).to_bytes(4, "big") + meta + idf_bytes

    @classmethod
    def loads(cls, payload: bytes) -> TFIDFEmbedding:
        """Restore state from serialized payload."""
        meta_len = int.from_bytes(payload[:4], "big")
        meta = json.loads(payload[4 : 4 + meta_len].decode())
        idf_bytes = payload[4 + meta_len :]

        obj = cls(
            lowercase=meta["lowercase"],
            token_pattern=meta["token_pattern"],
            max_features=meta["max_features"],
            min_df=meta["min_df"],
        )
        obj._vocab = meta["vocab"]
        obj._idf = np.frombuffer(idf_bytes, dtype=np.float32).copy()
        return obj

    def _embed_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        if self.dim == 0:
            return vec

        counts = Counter(token for token in self._tokenize(text) if token in self._vocab)
        if not counts:
            return vec

        max_tf = max(counts.values())
        for token, count in counts.items():
            idx = self._vocab[token]
            tf = count / max_tf
            vec[idx] = tf * self._idf[idx]

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def _tokenize(self, text: str) -> list[str]:
        if self.lowercase:
            text = text.lower()
        return self._token_re.findall(text)

    def _ensure_fitted(self) -> None:
        if not self._vocab:
            raise ValueError("TFIDFEmbedding is not fitted. Call fit() first.")
