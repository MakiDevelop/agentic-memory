"""agentic-memory: Repo memory for AI agents with citation verification."""

from agentic_memory.evidence import FileRef, GitCommitRef, ManualRef, URLRef
from agentic_memory.memory import Memory
from agentic_memory.models import MemoryRecord, QueryResult, ValidationStatus

__version__ = "0.1.0"

__all__ = [
    "Memory",
    "MemoryRecord",
    "QueryResult",
    "ValidationStatus",
    "FileRef",
    "GitCommitRef",
    "URLRef",
    "ManualRef",
]
