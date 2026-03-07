"""Main Memory class — the public API of agentic-memory."""

from __future__ import annotations

import os
from pathlib import Path

from agentic_memory.admission import AdmissionController, AlwaysAdmit
from agentic_memory.embedding import EmbeddingProvider
from agentic_memory.evidence import Evidence, FileRef
from agentic_memory.models import Citation, MemoryRecord, QueryResult, ValidationStatus
from agentic_memory.store import SQLiteStore


def _normalize_scores(items: list[tuple[str, float]]) -> dict[str, float]:
    """Normalize scores to 0-1 range."""
    if not items:
        return {}
    scores = [s for _, s in items]
    lo, hi = min(scores), max(scores)
    if hi <= lo:
        return {k: 1.0 for k, _ in items}
    return {k: (s - lo) / (hi - lo) for k, s in items}


class Memory:
    """Repository-scoped memory with citation enforcement and hybrid search.

    Usage:
        mem = Memory("./my-project")
        mem.add("Uses ruff for linting", evidence=FileRef("pyproject.toml", lines=(15, 20)))
        result = mem.query("What linter?")
    """

    def __init__(
        self,
        repo_path: str | Path,
        db_name: str = ".agentic-memory.db",
        admission: AdmissionController | None = None,
        embedding: EmbeddingProvider | None = None,
    ):
        self.repo_path = str(Path(repo_path).resolve())
        db_path = os.path.join(self.repo_path, db_name)
        self._store = SQLiteStore(db_path)
        self._admission = admission or AlwaysAdmit()
        self._embedding = embedding
        self._embedding_ready = False

        # Try to restore embedding provider state
        if self._embedding is not None:
            self._try_restore_embedding()

    def _try_restore_embedding(self) -> None:
        """Try to restore embedding provider state from DB."""
        if self._embedding is None:
            return
        state = self._store.load_provider_state("default")
        if state is not None:
            try:
                self._embedding = type(self._embedding).loads(state)
                self._embedding_ready = True
            except Exception:
                pass

    def _fit_embedding(self) -> None:
        """Fit embedding provider on all stored memories and persist state."""
        if self._embedding is None:
            return
        all_records = self._store.list_all(limit=10000)
        if not all_records:
            return

        texts = [r.content for r in all_records]
        self._embedding.fit(texts)
        self._embedding_ready = True

        # Persist provider state
        self._store.save_provider_state(
            "default", self._embedding.model_id, self._embedding.dim, self._embedding.dumps()
        )

        # Compute and store embeddings for all records
        vectors = self._embedding.embed_documents(texts)
        for record, vector in zip(all_records, vectors):
            self._store.save_embedding(record.id, self._embedding.model_id, vector)

    def add(
        self,
        content: str,
        evidence: Evidence,
        tags: list[str] | None = None,
    ) -> MemoryRecord:
        """Add a memory with required evidence.

        Args:
            content: The knowledge to remember.
            evidence: Citation source (FileRef, GitCommitRef, URLRef, ManualRef).
            tags: Optional tags for categorization.

        Returns:
            The created MemoryRecord.

        Raises:
            TypeError: If evidence is not an Evidence instance.
            ValueError: If admission control rejects the memory.
        """
        # Admission control gate
        admission_result = self._admission.check(content, tags)
        if not admission_result.admitted:
            raise ValueError(
                f"Memory rejected by admission control (score={admission_result.score}): "
                f"{admission_result.reason}"
            )

        if not isinstance(evidence, Evidence):
            raise TypeError(
                f"evidence must be an Evidence instance (FileRef, GitCommitRef, URLRef, ManualRef), "
                f"got {type(evidence).__name__}"
            )

        # Capture content hash for FileRef
        if isinstance(evidence, FileRef):
            evidence.capture_hash(self.repo_path)

        record = MemoryRecord(
            content=content,
            evidence=evidence,
            tags=tags or [],
        )

        # Validate on creation
        status, message = evidence.validate(self.repo_path)
        record.validation_status = status
        record.validation_message = message

        self._store.save(record)

        # Re-fit embedding (TF-IDF vocab changes with new documents)
        if self._embedding is not None:
            self._fit_embedding()

        return record

    def query(
        self,
        query: str,
        limit: int = 5,
        validate: bool = True,
        include_stale: bool = True,
        *,
        fts_weight: float = 0.65,
        vector_weight: float = 0.35,
    ) -> QueryResult:
        """Query memories with hybrid search and automatic citation validation.

        Combines FTS5 full-text search with vector similarity when an embedding
        provider is configured. Results are ranked using weighted score fusion.

        Args:
            query: Search query string.
            limit: Maximum number of memories to return.
            validate: Whether to re-validate citations before returning.
            include_stale: Whether to include stale/invalid memories in results.
            fts_weight: Weight for FTS5 scores in hybrid ranking (default: 0.65).
            vector_weight: Weight for vector scores in hybrid ranking (default: 0.35).

        Returns:
            QueryResult with answer, citations, and confidence.
        """
        fetch_limit = max(limit * 3, 20)

        # FTS5 search
        fts_records = self._store.search(query, limit=fetch_limit)
        fts_scores = _normalize_scores(
            [(r.id, 1.0 / (idx + 1)) for idx, r in enumerate(fts_records)]
        )

        # Vector search (if embedding provider is available and fitted)
        vector_scores: dict[str, float] = {}
        vector_records: dict[str, MemoryRecord] = {}
        if self._embedding is not None and self._embedding_ready:
            query_vec = self._embedding.embed_query(query)
            hits = self._store.vector_search(
                query_vec, model_id=self._embedding.model_id, limit=fetch_limit
            )
            vector_scores = _normalize_scores([(h.record.id, h.score) for h in hits])
            vector_records = {h.record.id: h.record for h in hits}

        # Merge results
        merged: dict[str, MemoryRecord] = {r.id: r for r in fts_records}
        merged.update(vector_records)

        # Score fusion
        scored: list[tuple[MemoryRecord, float]] = []
        for memory_id, record in merged.items():
            fs = fts_scores.get(memory_id, 0.0)
            vs = vector_scores.get(memory_id, 0.0)
            hybrid = (fts_weight * fs) + (vector_weight * vs)
            scored.append((record, hybrid))

        scored.sort(key=lambda item: item[1], reverse=True)
        records = [r for r, _ in scored[:limit]]

        # Validation
        if validate:
            for record in records:
                status, message = record.evidence.validate(self.repo_path)
                record.validation_status = status
                record.validation_message = message
                if status == ValidationStatus.STALE:
                    record.confidence = max(0.1, record.confidence * 0.5)
                elif status == ValidationStatus.INVALID:
                    record.confidence = 0.0
                self._store.update_validation(record.id, status, message, record.confidence)

        if not include_stale:
            records = [r for r in records if r.validation_status == ValidationStatus.VALID]

        # Sort by confidence descending
        records.sort(key=lambda r: r.confidence, reverse=True)

        citations = [
            Citation(
                evidence=r.evidence,
                status=r.validation_status,
                message=r.validation_message,
            )
            for r in records
        ]

        # Build answer from top memories
        answer = "\n".join(r.content for r in records) if records else ""
        avg_confidence = sum(r.confidence for r in records) / len(records) if records else 0.0

        return QueryResult(
            answer=answer,
            citations=citations,
            confidence=avg_confidence,
            memories=records,
        )

    def get(self, memory_id: str) -> MemoryRecord | None:
        """Get a specific memory by ID."""
        return self._store.get(memory_id)

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        return self._store.delete(memory_id)

    def validate(self) -> list[MemoryRecord]:
        """Validate all memories and return those that are stale or invalid."""
        all_memories = self._store.list_all(limit=10000)
        problematic = []

        for record in all_memories:
            status, message = record.evidence.validate(self.repo_path)
            record.validation_status = status
            record.validation_message = message

            if status == ValidationStatus.STALE:
                record.confidence = max(0.1, record.confidence * 0.5)
            elif status == ValidationStatus.INVALID:
                record.confidence = 0.0

            self._store.update_validation(record.id, status, message, record.confidence)

            if status in (ValidationStatus.STALE, ValidationStatus.INVALID):
                problematic.append(record)

        return problematic

    def status(self) -> dict:
        """Get summary status of all memories."""
        all_memories = self._store.list_all(limit=10000)
        counts = {s: 0 for s in ValidationStatus}
        for record in all_memories:
            counts[record.validation_status] += 1

        return {
            "total": len(all_memories),
            "valid": counts[ValidationStatus.VALID],
            "stale": counts[ValidationStatus.STALE],
            "invalid": counts[ValidationStatus.INVALID],
            "unchecked": counts[ValidationStatus.UNCHECKED],
        }

    def list_all(self, limit: int = 100) -> list[MemoryRecord]:
        """List all memories."""
        return self._store.list_all(limit=limit)

    def close(self) -> None:
        """Close the database connection."""
        self._store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
