"""Core data models for agentic-memory."""

from __future__ import annotations

import enum
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


@dataclass
class MemoryRecord:
    """A single memory entry with its evidence chain."""

    content: str
    evidence: Evidence
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    validation_status: ValidationStatus = ValidationStatus.UNCHECKED
    validation_message: str = ""
    tags: list[str] = field(default_factory=list)


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
