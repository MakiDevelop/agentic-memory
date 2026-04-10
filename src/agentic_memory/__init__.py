"""agentic-memory: Repo memory for AI agents with citation verification."""

from agentic_memory.admission import (
    AdmissionController,
    AlwaysAdmit,
    HeuristicAdmissionController,
    LLMAdmissionController,
)
from agentic_memory.backends import StorageBackend
from agentic_memory.content_validator import (
    ContentValidator,
    KeywordOverlapValidator,
    LLMContentValidator,
)
from agentic_memory.embedding import EmbeddingProvider, TFIDFEmbedding
from agentic_memory.evidence import FileRef, GitCommitRef, ManualRef, URLRef
from agentic_memory.graph import MemoryEdge, MemoryGraph, RelationType
from agentic_memory.memory import Memory
from agentic_memory.models import (
    AddResult,
    CompactResult,
    EvalMetrics,
    MemoryKind,
    MemoryRecord,
    QueryResult,
    RetrievalLog,
    ValidationResult,
    ValidationStatus,
)

__version__ = "0.8.0"

__all__ = [
    "Memory",
    "MemoryRecord",
    "QueryResult",
    "MemoryKind",
    "RetrievalLog",
    "ValidationResult",
    "ValidationStatus",
    "AddResult",
    "CompactResult",
    "EvalMetrics",
    "FileRef",
    "GitCommitRef",
    "URLRef",
    "ManualRef",
    "AdmissionController",
    "AlwaysAdmit",
    "HeuristicAdmissionController",
    "LLMAdmissionController",
    "StorageBackend",
    "EmbeddingProvider",
    "TFIDFEmbedding",
    "ContentValidator",
    "KeywordOverlapValidator",
    "LLMContentValidator",
    "MemoryGraph",
    "MemoryEdge",
    "RelationType",
]
