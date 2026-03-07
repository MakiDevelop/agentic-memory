"""Main Memory class — the public API of agentic-memory."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from agentic_memory.admission import AdmissionController, AlwaysAdmit
from agentic_memory.content_validator import ContentValidator, read_evidence_content
from agentic_memory.embedding import EmbeddingProvider
from agentic_memory.evidence import Evidence, FileRef
from agentic_memory.models import (
    Citation, MemoryKind, MemoryRecord, QueryResult, RetrievalLog, ValidationStatus, _content_hash,
)
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
        content_validator: ContentValidator | None = None,
    ):
        self.repo_path = str(Path(repo_path).resolve())
        db_path = os.path.join(self.repo_path, db_name)
        self._store = SQLiteStore(db_path)
        self._admission = admission or AlwaysAdmit()
        self._embedding = embedding
        self._content_validator = content_validator
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
        evidence: Evidence | list[Evidence],
        tags: list[str] | None = None,
        kind: MemoryKind | str = MemoryKind.FACT,
        importance: int = 1,
        ttl_seconds: int | None = None,
        deduplicate: bool = True,
    ) -> MemoryRecord:
        """Add a memory with required evidence.

        Args:
            content: The knowledge to remember.
            evidence: Citation source(s). Single Evidence or list of Evidence.
            tags: Optional tags for categorization.
            kind: Memory type (fact/rule/antipattern/preference/decision).
            importance: Priority level 0-3 (0=low, 1=normal, 2=high, 3=critical).
            ttl_seconds: Time-to-live in seconds. None=never expires.
            deduplicate: If True, skip adding if identical content already exists.

        Returns:
            The created MemoryRecord.

        Raises:
            TypeError: If evidence is not an Evidence instance (or list thereof).
            ValueError: If admission control rejects the memory.
        """
        # Normalize kind
        if isinstance(kind, str):
            kind = MemoryKind(kind)

        # Admission control gate
        admission_result = self._admission.check(content, tags)
        if not admission_result.admitted:
            raise ValueError(
                f"Memory rejected by admission control (score={admission_result.score}): "
                f"{admission_result.reason}"
            )

        evidence_items = evidence if isinstance(evidence, list) else [evidence]
        if not evidence_items:
            raise ValueError("evidence must not be empty")
        for e in evidence_items:
            if not isinstance(e, Evidence):
                raise TypeError(
                    f"evidence must be an Evidence instance (FileRef, GitCommitRef, URLRef, ManualRef), "
                    f"got {type(e).__name__}"
                )
            if isinstance(e, FileRef):
                e.capture_hash(self.repo_path)

        # Deduplication check
        content_hash = _content_hash(content)
        if deduplicate:
            existing = self._store.find_by_hash(content_hash)
            if existing is not None:
                return existing

        record = MemoryRecord(
            content=content,
            evidence=evidence,
            tags=tags or [],
            kind=kind,
            importance=max(0, min(3, importance)),
            ttl_seconds=ttl_seconds,
            source_hash=content_hash,
        )

        # Validate on creation — take worst status across all evidence
        statuses = [e.validate(self.repo_path) for e in evidence_items]
        severity = {ValidationStatus.VALID: 0, ValidationStatus.UNCHECKED: 1,
                    ValidationStatus.STALE: 2, ValidationStatus.INVALID: 3}
        worst = max(statuses, key=lambda s: severity.get(s[0], 0))
        record.validation_status = worst[0]
        record.validation_message = worst[1]

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
        kind: MemoryKind | str | None = None,
        min_importance: int = 0,
        *,
        fts_weight: float = 0.65,
        vector_weight: float = 0.35,
    ) -> QueryResult:
        """Query memories with hybrid search and automatic citation validation.

        Args:
            query: Search query string.
            limit: Maximum number of memories to return.
            validate: Whether to re-validate citations before returning.
            include_stale: Whether to include stale/invalid memories in results.
            kind: Filter by memory kind (e.g., "rule", "fact").
            min_importance: Minimum importance level (0-3).
            fts_weight: Weight for FTS5 scores in hybrid ranking (default: 0.65).
            vector_weight: Weight for vector scores in hybrid ranking (default: 0.35).

        Returns:
            QueryResult with answer, citations, and confidence.
        """
        import time
        t0 = time.monotonic()

        if isinstance(kind, str):
            kind = MemoryKind(kind)

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
        records = [r for r, _ in scored]

        # Filter by kind and importance
        if kind is not None:
            records = [r for r in records if r.kind == kind]
        if min_importance > 0:
            records = [r for r in records if r.importance >= min_importance]

        # Filter expired memories (TTL)
        records = [r for r in records if not r.is_expired]

        records = records[:limit]

        # Validation
        if validate:
            for record in records:
                self._validate_record(record)

        if not include_stale:
            records = [r for r in records if r.validation_status == ValidationStatus.VALID]

        # Sort by importance (primary) then confidence (secondary)
        records.sort(key=lambda r: (r.importance, r.confidence), reverse=True)

        citations = [
            Citation(
                evidence=e,
                status=r.validation_status,
                message=r.validation_message,
            )
            for r in records
            for e in r.evidence_list
        ]

        # Build answer from top memories
        answer = "\n".join(r.content for r in records) if records else ""
        avg_confidence = sum(r.confidence for r in records) / len(records) if records else 0.0

        qr = QueryResult(
            answer=answer,
            citations=citations,
            confidence=avg_confidence,
            memories=records,
        )

        # Log retrieval
        latency = (time.monotonic() - t0) * 1000
        self._store.log_retrieval(RetrievalLog(
            query=query,
            returned_ids=[r.id for r in records],
            result_count=len(records),
            latency_ms=latency,
        ))

        return qr

    def get(self, memory_id: str) -> MemoryRecord | None:
        """Get a specific memory by ID."""
        return self._store.get(memory_id)

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        return self._store.delete(memory_id)

    def _validate_record(self, record: MemoryRecord) -> None:
        """Validate a single record: TTL + evidence check + optional content check."""
        # TTL check
        if record.is_expired:
            record.validation_status = ValidationStatus.STALE
            record.validation_message = "Memory expired (TTL exceeded)"
            record.confidence = max(0.1, record.confidence * 0.5)
            record.updated_at = datetime.now()
            self._store.save(record)
            return

        evidence_items = record.evidence_list
        if not evidence_items:
            record.validation_status = ValidationStatus.UNCHECKED
            record.validation_message = "No evidence to validate"
            return

        severity = {ValidationStatus.VALID: 0, ValidationStatus.UNCHECKED: 1,
                    ValidationStatus.STALE: 2, ValidationStatus.INVALID: 3}

        # Validate all evidence items, take worst status
        statuses = [e.validate(self.repo_path) for e in evidence_items]
        status, message = max(statuses, key=lambda s: severity.get(s[0], 0))

        # Content-level validation (only if evidence is valid and validator is configured)
        if status == ValidationStatus.VALID and self._content_validator is not None:
            for e in evidence_items:
                evidence_content = read_evidence_content(e, self.repo_path)
                if evidence_content is not None:
                    cv_result = self._content_validator.check(record.content, evidence_content)
                    if not cv_result.consistent:
                        status = ValidationStatus.STALE
                        message = f"Content mismatch: {cv_result.reason}"
                        break

        record.validation_status = status
        record.validation_message = message

        if status == ValidationStatus.STALE:
            record.confidence = max(0.1, record.confidence * 0.5)
        elif status == ValidationStatus.INVALID:
            record.confidence = 0.0

        # Single write: update timestamp + persist relocated evidence
        record.updated_at = datetime.now()
        self._store.save(record)

    def validate(self) -> list[MemoryRecord]:
        """Validate all memories and return those that are stale or invalid."""
        all_memories = self._store.list_all(limit=10000)
        problematic = []

        for record in all_memories:
            self._validate_record(record)
            if record.validation_status in (ValidationStatus.STALE, ValidationStatus.INVALID):
                problematic.append(record)

        return problematic

    def status(self) -> dict:
        """Get summary status of all memories."""
        all_memories = self._store.list_all(limit=10000)
        counts = {s: 0 for s in ValidationStatus}
        for record in all_memories:
            counts[record.validation_status] += 1

        expired = sum(1 for r in all_memories if r.is_expired)
        return {
            "total": len(all_memories),
            "valid": counts[ValidationStatus.VALID],
            "stale": counts[ValidationStatus.STALE],
            "invalid": counts[ValidationStatus.INVALID],
            "unchecked": counts[ValidationStatus.UNCHECKED],
            "expired": expired,
        }

    def retrieval_stats(self, limit: int = 100) -> list[RetrievalLog]:
        """Get recent retrieval logs."""
        return self._store.get_retrieval_stats(limit=limit)

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
