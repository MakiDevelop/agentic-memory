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
    AddResult, Citation, CompactResult, EvalMetrics, MemoryKind, MemoryRecord,
    QueryResult, RetrievalLog, ValidationStatus, _content_hash,
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
        self._adds_since_refit = 0
        self._refit_threshold = 10

        # Try to restore embedding provider state
        if self._embedding is not None:
            self._try_restore_embedding()

    def _try_restore_embedding(self) -> None:
        """Try to restore embedding provider state from DB.

        After restoring, checks if new un-embedded memories exist and triggers
        a refit if the gap exceeds the threshold.
        """
        if self._embedding is None:
            return
        state = self._store.load_provider_state("default")
        if state is not None:
            try:
                self._embedding = type(self._embedding).loads(state)
                self._embedding_ready = True
                # Check if there are un-embedded memories from previous sessions
                total = self._store.count()
                embedded = self._store.count_embeddings(self._embedding.model_id)
                gap = total - embedded
                if gap >= self._refit_threshold:
                    self._fit_embedding()
                elif gap > 0:
                    self._adds_since_refit = gap
            except Exception:
                pass

    def _fit_embedding(self) -> None:
        """Full refit: rebuild vocab from all memories and recompute all embeddings."""
        if self._embedding is None:
            return
        all_records = self._store.list_all(limit=None)
        if not all_records:
            return

        texts = [r.content for r in all_records]
        self._embedding.fit(texts)
        self._embedding_ready = True
        self._adds_since_refit = 0

        # Persist provider state
        self._store.save_provider_state(
            "default", self._embedding.model_id, self._embedding.dim, self._embedding.dumps()
        )

        # Compute and store embeddings for all records
        vectors = self._embedding.embed_documents(texts)
        for record, vector in zip(all_records, vectors):
            self._store.save_embedding(record.id, self._embedding.model_id, vector)

    def _embed_single(self, record: MemoryRecord) -> None:
        """Compute and store embedding for a single record using existing vocab."""
        if self._embedding is None or not self._embedding_ready:
            return
        vectors = self._embedding.embed_documents([record.content])
        self._store.save_embedding(record.id, self._embedding.model_id, vectors[0])

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

        # Conflict detection: search for similar existing memories with different content
        conflicts = self._detect_conflicts(content, content_hash)
        record.conflict_ids = [c.id for c in conflicts]

        self._store.save(record)

        # Embedding: incremental for existing vocab, full refit periodically
        if self._embedding is not None:
            if not self._embedding_ready:
                self._fit_embedding()
            else:
                self._embed_single(record)
                self._adds_since_refit += 1
                if self._adds_since_refit >= self._refit_threshold:
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
        old_status = record.validation_status
        old_message = record.validation_message
        old_confidence = record.confidence
        old_evidence_json = str([e.to_dict() for e in record.evidence_list])

        # TTL check
        if record.is_expired:
            record.validation_status = ValidationStatus.STALE
            record.validation_message = "Memory expired (TTL exceeded)"
            record.confidence = max(0.1, record.confidence * 0.5)
            if record.validation_status != old_status or record.confidence != old_confidence:
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

        # Only write if state actually changed (dirty check)
        new_evidence_json = str([e.to_dict() for e in record.evidence_list])
        changed = (
            record.validation_status != old_status
            or record.confidence != old_confidence
            or record.validation_message != old_message
            or new_evidence_json != old_evidence_json
        )
        if changed:
            record.updated_at = datetime.now()
            self._store.save(record)

    def validate(self) -> list[MemoryRecord]:
        """Validate all memories and return those that are stale or invalid."""
        all_memories = self._store.list_all(limit=None)
        problematic = []

        for record in all_memories:
            self._validate_record(record)
            if record.validation_status in (ValidationStatus.STALE, ValidationStatus.INVALID):
                problematic.append(record)

        return problematic

    def status(self) -> dict:
        """Get summary status of all memories."""
        all_memories = self._store.list_all(limit=None)
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

    _STOP_WORDS = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "must",
        "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
        "into", "about", "between", "through", "during", "before", "after",
        "and", "but", "or", "nor", "not", "no", "so", "yet",
        "this", "that", "these", "those", "it", "its", "we", "our",
        "uses", "use", "used", "using",
    })

    def _significant_words(self, text: str) -> set[str]:
        """Extract significant (non-stop) words from text."""
        return {w.lower() for w in text.split() if w.lower() not in self._STOP_WORDS and len(w) > 2}

    def _detect_conflicts(self, content: str, content_hash: str) -> list[MemoryRecord]:
        """Find existing memories that might conflict with new content."""
        new_words = self._significant_words(content)
        if len(new_words) < 2:
            return []
        try:
            candidates = self._store.search_any(content, limit=10)
        except Exception:
            return []

        conflicts = []
        for c in candidates:
            if c.source_hash == content_hash or c.is_expired:
                continue
            existing_words = self._significant_words(c.content)
            if not existing_words:
                continue
            # Require at least 40% word overlap to consider it a conflict
            overlap = len(new_words & existing_words)
            overlap_ratio = overlap / min(len(new_words), len(existing_words))
            if overlap_ratio >= 0.4:
                conflicts.append(c)
        return conflicts

    def detect_conflicts(self, content: str) -> list[MemoryRecord]:
        """Find existing memories that potentially conflict with the given content."""
        return self._detect_conflicts(content, _content_hash(content))

    def add_with_result(
        self,
        content: str,
        evidence: Evidence | list[Evidence],
        tags: list[str] | None = None,
        kind: MemoryKind | str = MemoryKind.FACT,
        importance: int = 1,
        ttl_seconds: int | None = None,
        deduplicate: bool = True,
    ) -> AddResult:
        """Add a memory and return rich result with conflict/dedup info.

        Same as add() but returns AddResult instead of bare MemoryRecord.
        """
        content_hash = _content_hash(content)
        was_duplicate = False

        if deduplicate:
            existing = self._store.find_by_hash(content_hash)
            if existing is not None:
                conflicts = self._detect_conflicts(content, content_hash)
                return AddResult(record=existing, was_duplicate=True, conflicts=conflicts)

        record = self.add(
            content, evidence, tags=tags, kind=kind,
            importance=importance, ttl_seconds=ttl_seconds, deduplicate=False,
        )
        conflicts = [self._store.get(cid) for cid in record.conflict_ids]
        conflicts = [c for c in conflicts if c is not None]
        return AddResult(record=record, was_duplicate=was_duplicate, conflicts=conflicts)

    def compact(self) -> CompactResult:
        """Remove expired memories and return cleanup stats."""
        all_memories = self._store.list_all(limit=None)
        total_before = len(all_memories)
        expired_removed = 0

        for record in all_memories:
            if record.is_expired:
                self._store.delete(record.id)
                expired_removed += 1

        return CompactResult(
            expired_removed=expired_removed,
            total_before=total_before,
            total_after=total_before - expired_removed,
        )

    def create_if_useful(
        self,
        content: str,
        evidence: Evidence | list[Evidence],
        tags: list[str] | None = None,
        kind: MemoryKind | str = MemoryKind.FACT,
        importance: int = 1,
        ttl_seconds: int | None = None,
        min_importance: int = 0,
    ) -> AddResult | None:
        """Add a memory only if it passes admission + importance threshold.

        Returns AddResult if added, None if rejected (too low importance or admission fail).
        """
        if importance < min_importance:
            return None

        try:
            return self.add_with_result(
                content, evidence, tags=tags, kind=kind,
                importance=importance, ttl_seconds=ttl_seconds,
            )
        except ValueError:
            return None

    def search_context(
        self,
        query: str,
        limit: int = 5,
        kind: MemoryKind | str | None = None,
        min_importance: int = 0,
    ) -> str:
        """Query memories and return a formatted context string for agent consumption.

        Returns a ready-to-use text block with memories, citations, and confidence.
        """
        result = self.query(query, limit=limit, kind=kind, min_importance=min_importance)

        if not result.memories:
            return f"No memories found for: {query}"

        lines = [f"Found {len(result.memories)} memories:"]
        for i, mem in enumerate(result.memories, 1):
            status_icon = {"valid": "✓", "stale": "⚠", "invalid": "✗", "unchecked": "?"}
            icon = status_icon.get(mem.validation_status.value, "?")
            lines.append(f"\n[{i}] {mem.content}")
            lines.append(f"    {icon} {mem.evidence_label} [{mem.validation_status.value}]")
            lines.append(f"    kind={mem.kind.value} importance={mem.importance} confidence={mem.confidence:.1f}")
            if mem.conflict_ids:
                lines.append(f"    ⚡ conflicts with: {', '.join(mem.conflict_ids)}")
        lines.append(f"\nOverall confidence: {result.confidence:.1f}")
        return "\n".join(lines)

    def mark_adopted(self, memory_id: str, query: str = "", agent_name: str = "") -> bool:
        """Mark a memory as adopted (actually used by an agent).

        Call this when an agent confirms it used a memory from query results.
        Returns True if the memory exists.
        """
        record = self._store.get(memory_id)
        if record is None:
            return False
        self._store.log_adoption(memory_id, query=query, agent_name=agent_name)
        return True

    def eval_metrics(self) -> EvalMetrics:
        """Compute evaluation metrics for the memory system."""
        all_memories = self._store.list_all(limit=None)
        logs = self._store.get_retrieval_stats(limit=None)
        s = self.status()

        avg_latency = sum(l.latency_ms for l in logs) / len(logs) if logs else 0.0
        avg_results = sum(l.result_count for l in logs) / len(logs) if logs else 0.0

        # Adoption metrics
        total_adoptions = self._store.get_adoption_total()
        total_returned = sum(l.result_count for l in logs)
        adoption_rate = total_adoptions / total_returned if total_returned > 0 else 0.0

        return EvalMetrics(
            total_queries=len(logs),
            total_memories=len(all_memories),
            avg_latency_ms=round(avg_latency, 2),
            avg_results_per_query=round(avg_results, 2),
            expired_count=s["expired"],
            stale_count=s["stale"],
            total_adoptions=total_adoptions,
            adoption_rate=round(adoption_rate, 4),
        )

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
