"""Tests for backend protocol conformance."""

from __future__ import annotations

import inspect

from agentic_memory import StorageBackend
from agentic_memory.backends import SearchBackend
from agentic_memory.store import SQLiteStore


class DummySearch:
    """Minimal runtime implementation for SearchBackend tests."""

    def text_search(self, query: str, limit: int = 10):
        return []


def test_sqlitestore_satisfies_protocol(tmp_path):
    store = SQLiteStore(tmp_path / "memory.db")
    try:
        assert isinstance(store, StorageBackend)
    finally:
        store.close()


def test_sqlitestore_is_structural_subclass():
    assert issubclass(SQLiteStore, StorageBackend)


def test_storage_backend_methods_exist_on_sqlitestore():
    expected_methods = [
        "save",
        "get",
        "delete",
        "search",
        "search_any",
        "list_all",
        "count",
        "find_by_hash",
        "update_validation",
        "save_embedding",
        "vector_search",
        "save_provider_state",
        "load_provider_state",
        "has_embeddings",
        "count_embeddings",
        "log_retrieval",
        "get_retrieval_stats",
        "log_adoption",
        "get_adoption_counts",
        "get_adoption_total",
        "close",
    ]
    for method_name in expected_methods:
        assert hasattr(SQLiteStore, method_name)
        assert callable(getattr(SQLiteStore, method_name))


def test_storage_backend_signatures_match_sqlitestore():
    for method_name in StorageBackend.__dict__:
        if method_name.startswith("_"):
            continue
        protocol_member = getattr(StorageBackend, method_name, None)
        if not callable(protocol_member):
            continue
        implementation_member = getattr(SQLiteStore, method_name)
        protocol_signature = inspect.signature(protocol_member)
        implementation_signature = inspect.signature(implementation_member)

        protocol_parameters = list(protocol_signature.parameters.values())
        implementation_parameters = list(implementation_signature.parameters.values())

        assert [param.name for param in implementation_parameters] == [param.name for param in protocol_parameters]
        assert [param.kind for param in implementation_parameters] == [param.kind for param in protocol_parameters]


def test_search_backend_is_runtime_checkable():
    assert isinstance(DummySearch(), SearchBackend)


def test_storage_backend_is_exported_from_package():
    from agentic_memory import StorageBackend as exported_protocol

    assert exported_protocol is StorageBackend
