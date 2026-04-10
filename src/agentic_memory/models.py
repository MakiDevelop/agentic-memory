"""Core data models for agentic-memory."""

from __future__ import annotations

import enum
import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_memory.evidence import Evidence


class ValidationStatus(enum.Enum):
    """Status of a memory's evidence after validation."""

    VALID = "valid"
    STALE = "stale"  # evidence exists but content changed
    INVALID = "invalid"  # evidence no longer exists
    UNCHECKED = "unchecked"  # not yet validated


class MemoryKind(enum.Enum):
    """Type of memory content."""

    FACT = "fact"
    RULE = "rule"
    ANTIPATTERN = "antipattern"
    PREFERENCE = "preference"
    DECISION = "decision"


def _content_hash(content: str) -> str:
    """Generate a hash for deduplication."""
    return hashlib.sha256(content.strip().lower().encode()).hexdigest()[:32]


@dataclass
class MemoryRecord:
    """A single memory entry with its evidence chain."""

    content: str
    evidence: Evidence | list[Evidence]
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    validation_status: ValidationStatus = ValidationStatus.UNCHECKED
    validation_message: str = ""
    tags: list[str] = field(default_factory=list)
    kind: MemoryKind = MemoryKind.FACT
    importance: int = 1  # 0=low, 1=normal, 2=high, 3=critical
    ttl_seconds: int | None = None  # None=never expires
    source_hash: str = ""
    conflict_ids: list[str] = field(default_factory=list)  # IDs of potentially conflicting memories
    superseded_by: str | None = None  # ID of the memory that supersedes this one

    @property
    def evidence_list(self) -> list[Evidence]:
        """Always returns evidence as a list for uniform iteration."""
        if isinstance(self.evidence, list):
            return self.evidence
        return [self.evidence]

    @property
    def evidence_label(self) -> str:
        """Human-readable label for all evidence sources."""
        return " | ".join(e.short_label() for e in self.evidence_list)

    @property
    def is_expired(self) -> bool:
        """Check if memory has exceeded its TTL."""
        if self.ttl_seconds is None:
            return False
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return elapsed > self.ttl_seconds


@dataclass
class ValidationResult:
    """Rich validation result with optional diff content."""

    status: ValidationStatus
    message: str
    old_content: str | None = None
    new_content: str | None = None

    def as_tuple(self) -> tuple[ValidationStatus, str]:
        return (self.status, self.message)


@dataclass
class Citation:
    """A citation attached to a query result."""

    evidence: Evidence
    status: ValidationStatus
    message: str = ""


@dataclass
class QueryResult:
    """Result of a memory query."""

    answer: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 1.0
    memories: list[MemoryRecord] = field(default_factory=list)


@dataclass
class AddResult:
    """Result of adding a memory, including conflict information."""

    record: MemoryRecord
    was_duplicate: bool = False
    conflicts: list[MemoryRecord] = field(default_factory=list)


@dataclass
class CompactResult:
    """Result of a compact operation."""

    expired_removed: int = 0
    total_before: int = 0
    total_after: int = 0


@dataclass
class EvalMetrics:
    """Evaluation metrics for memory system quality."""

    total_queries: int = 0
    total_memories: int = 0
    avg_latency_ms: float = 0.0
    avg_results_per_query: float = 0.0
    expired_count: int = 0
    stale_count: int = 0
    total_adoptions: int = 0
    adoption_rate: float = 0.0  # adoptions / total returned results


@dataclass
class RetrievalLog:
    """Log entry for a memory query."""

    query: str
    returned_ids: list[str]
    result_count: int
    latency_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
