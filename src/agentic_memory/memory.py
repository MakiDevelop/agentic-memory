"""Main Memory class — the public API of agentic-memory."""

from __future__ import annotations

import os
from pathlib import Path

from agentic_memory.admission import AdmissionController, AlwaysAdmit
from agentic_memory.evidence import Evidence, FileRef
from agentic_memory.models import Citation, MemoryRecord, QueryResult, ValidationStatus
from agentic_memory.store import SQLiteStore


class Memory:
    """Repository-scoped memory with citation enforcement.

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
    ):
        self.repo_path = str(Path(repo_path).resolve())
        db_path = os.path.join(self.repo_path, db_name)
        self._store = SQLiteStore(db_path)
        self._admission = admission or AlwaysAdmit()

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
        return record

    def query(
        self,
        query: str,
        limit: int = 5,
        validate: bool = True,
        include_stale: bool = True,
    ) -> QueryResult:
        """Query memories with automatic citation validation.

        Args:
            query: Search query string.
            limit: Maximum number of memories to return.
            validate: Whether to re-validate citations before returning.
            include_stale: Whether to include stale/invalid memories in results.

        Returns:
            QueryResult with answer, citations, and confidence.
        """
        records = self._store.search(query, limit=limit)

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
