"""LangChain integration bridge for agentic-memory.

Provides MemciteRetriever — a LangChain-compatible retriever that returns
citation-backed documents from the memcite memory store.

Usage:
    pip install memcite[langchain]

    from agentic_memory.bridges.langchain import MemciteRetriever
    retriever = MemciteRetriever(repo_path="./my-project")
    docs = retriever.invoke("What linter does this project use?")
"""

from __future__ import annotations

from typing import Any

try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

from agentic_memory.memory import Memory


def _check_langchain() -> None:
    if not _LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain integration requires langchain-core. "
            "Install with: pip install memcite[langchain]"
        )


if _LANGCHAIN_AVAILABLE:

    class MemciteRetriever(BaseRetriever):
        """LangChain retriever backed by memcite's citation-validated memory.

        Each returned Document includes citation metadata (evidence source,
        validation status, confidence) so downstream chains can assess
        trustworthiness.

        Attributes:
            repo_path: Path to the repository with .agentic-memory.db
            validate: Re-validate citations on each query (default: True)
            include_stale: Include stale/invalid memories (default: True)
            kind: Filter by memory kind (default: None = all)
            min_importance: Minimum importance level 0-3 (default: 0)
        """

        repo_path: str = "."
        validate: bool = True
        include_stale: bool = True
        kind: str | None = None
        min_importance: int = 0
        max_results: int = 5

        _memory: Memory | None = None

        class Config:
            arbitrary_types_allowed = True
            underscore_attrs_are_private = True

        def _get_memory(self) -> Memory:
            if self._memory is None:
                self._memory = Memory(self.repo_path)
            return self._memory

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun | None = None,
        ) -> list[Document]:
            """Retrieve citation-backed documents from memcite."""
            mem = self._get_memory()
            result = mem.query(
                query,
                limit=self.max_results,
                validate=self.validate,
                include_stale=self.include_stale,
                kind=self.kind,
                min_importance=self.min_importance,
            )

            documents = []
            for record in result.memories:
                metadata: dict[str, Any] = {
                    "memory_id": record.id,
                    "evidence": record.evidence_label,
                    "validation_status": record.validation_status.value,
                    "confidence": record.confidence,
                    "kind": record.kind.value,
                    "importance": record.importance,
                }
                if record.validation_message:
                    metadata["validation_message"] = record.validation_message
                if record.conflict_ids:
                    metadata["conflicts"] = record.conflict_ids

                documents.append(Document(
                    page_content=record.content,
                    metadata=metadata,
                ))

            return documents

else:
    # Stub for when langchain is not installed
    class MemciteRetriever:  # type: ignore[no-redef]
        """Stub — install langchain-core to use this class."""

        def __init__(self, **kwargs: Any):
            _check_langchain()
