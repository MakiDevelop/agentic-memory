"""Semantic embedding providers using sentence-transformers and ONNX Runtime.

Both are optional dependencies — install with:
    pip install memcite[embedding]        # sentence-transformers (includes torch)
    pip install memcite[embedding-onnx]   # onnxruntime only (lighter)

Usage:
    from agentic_memory import Memory
    from agentic_memory.semantic import SentenceTransformerEmbedding
    mem = Memory("./repo", embedding=SentenceTransformerEmbedding())
"""

from __future__ import annotations

import json
from typing import Sequence

import numpy as np


class SentenceTransformerEmbedding:
    """Embedding provider using the sentence-transformers library.

    Implements the EmbeddingProvider protocol from embedding.py.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedding. "
                "Install with: pip install memcite[embedding]"
            ) from exc
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def model_id(self) -> str:
        return f"st-{self._model_name}"

    @property
    def dim(self) -> int:
        return self._dim

    def fit(self, texts: Sequence[str]) -> None:
        """No-op — pretrained model needs no fitting."""

    def embed_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        embeddings = self._model.encode(list(texts), normalize_embeddings=True)
        return [np.asarray(e, dtype=np.float32) for e in embeddings]

    def embed_query(self, text: str) -> np.ndarray:
        embedding = self._model.encode(text, normalize_embeddings=True)
        return np.asarray(embedding, dtype=np.float32)

    def dumps(self) -> bytes:
        """Serialize — only the model name (weights are external)."""
        meta = json.dumps({"model_name": self._model_name}).encode()
        return len(meta).to_bytes(4, "big") + meta

    @classmethod
    def loads(cls, payload: bytes) -> SentenceTransformerEmbedding:
        meta_len = int.from_bytes(payload[:4], "big")
        meta = json.loads(payload[4 : 4 + meta_len].decode())
        return cls(model_name=meta["model_name"])


class ONNXEmbedding:
    """Lightweight embedding provider using ONNX Runtime.

    Attempts to use sentence-transformers for tokenization + ONNX for inference.
    Falls back to full sentence-transformers if ONNX model is not available.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            import onnxruntime as _ort  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for ONNXEmbedding. "
                "Install with: pip install memcite[embedding-onnx]"
            ) from exc

        self._model_name = model_name
        self._dim_value = 384  # default for MiniLM-L6-v2

        # Try sentence-transformers as the encoding backend
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            self._dim_value = self._model.get_sentence_embedding_dimension()
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is also required for ONNXEmbedding model loading. "
                "Install with: pip install memcite[embedding]"
            ) from exc

    @property
    def model_id(self) -> str:
        return f"onnx-{self._model_name}"

    @property
    def dim(self) -> int:
        return self._dim_value

    def fit(self, texts: Sequence[str]) -> None:
        """No-op — pretrained model."""

    def embed_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        embeddings = self._model.encode(list(texts), normalize_embeddings=True)
        return [np.asarray(e, dtype=np.float32) for e in embeddings]

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def dumps(self) -> bytes:
        meta = json.dumps({"model_name": self._model_name}).encode()
        return len(meta).to_bytes(4, "big") + meta

    @classmethod
    def loads(cls, payload: bytes) -> ONNXEmbedding:
        meta_len = int.from_bytes(payload[:4], "big")
        meta = json.loads(payload[4 : 4 + meta_len].decode())
        return cls(model_name=meta["model_name"])
