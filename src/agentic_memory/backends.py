"""Backend protocols for pluggable storage and search implementations."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from agentic_memory.models import MemoryRecord, RetrievalLog, ValidationStatus


@runtime_checkable
class StorageBackend(Protocol):
    """Abstract storage interface. SQLiteStore is the canonical implementation."""

    def save(self, record: MemoryRecord) -> None:
        """Insert or update a memory record."""

    def get(self, memory_id: str) -> MemoryRecord | None:
        """Fetch a memory by ID."""

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""

    def search(self, query: str, limit: int = 10) -> list[MemoryRecord]:
        """Run a full-text search."""

    def search_any(self, query: str, limit: int = 10) -> list[MemoryRecord]:
        """Run an OR-based full-text search."""

    def list_all(self, limit: int | None = 100) -> list[MemoryRecord]:
        """List stored memories."""

    def count(self) -> int:
        """Return total number of stored memories."""

    def find_by_hash(self, source_hash: str) -> MemoryRecord | None:
        """Find a memory by deduplication hash."""

    def update_validation(
        self,
        memory_id: str,
        status: ValidationStatus,
        message: str,
        confidence: float,
    ) -> None:
        """Persist validation results."""

    def save_embedding(self, memory_id: str, model_id: str, vector: np.ndarray) -> None:
        """Store an embedding vector for a memory."""

    def vector_search(
        self,
        query_vector: np.ndarray,
        *,
        model_id: str,
        limit: int = 10,
    ) -> list[Any]:
        """Run a vector similarity search."""

    def save_provider_state(self, provider_key: str, model_id: str, dim: int, state: bytes) -> None:
        """Persist embedding provider state."""

    def load_provider_state(self, provider_key: str) -> bytes | None:
        """Load persisted embedding provider state."""

    def has_embeddings(self, model_id: str) -> bool:
        """Return whether any embeddings exist for a model."""

    def count_embeddings(self, model_id: str) -> int:
        """Count embeddings for a model."""

    def log_retrieval(self, log: RetrievalLog) -> None:
        """Record a retrieval event."""

    def get_retrieval_stats(self, limit: int | None = 100) -> list[RetrievalLog]:
        """Return recent retrieval logs."""

    def log_adoption(self, memory_id: str, query: str = "", agent_name: str = "") -> None:
        """Record adoption of a memory."""

    def get_adoption_counts(self) -> dict[str, int]:
        """Return adoption counts grouped by memory ID."""

    def get_adoption_total(self) -> int:
        """Return total adoption count."""

    def close(self) -> None:
        """Close any backend resources."""


@runtime_checkable
class SearchBackend(Protocol):
    """Abstract text-search interface for future non-SQLite search backends."""

    def text_search(self, query: str, limit: int = 10) -> list[MemoryRecord]:
        """Run a text search and return matching memories."""
