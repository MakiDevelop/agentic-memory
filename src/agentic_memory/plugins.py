"""Plugin discovery for agentic-memory via importlib metadata entry points."""

from __future__ import annotations

import importlib.metadata
from typing import Any

from agentic_memory.evidence import Evidence, FileRef, GitCommitRef, ManualRef, URLRef

_BUILTIN_EVIDENCE: dict[str, type[Evidence]] = {
    "file": FileRef,
    "git_commit": GitCommitRef,
    "url": URLRef,
    "manual": ManualRef,
}

_evidence_registry: dict[str, type[Evidence]] | None = None
_storage_backend_registry: dict[str, type[Any]] | None = None
_search_backend_registry: dict[str, type[Any]] | None = None


def _iter_entry_points(group: str) -> list[Any]:
    try:
        entry_points = importlib.metadata.entry_points(group=group)
    except TypeError:
        try:
            entry_points = importlib.metadata.entry_points()
        except Exception:
            return []
        if hasattr(entry_points, "select"):
            return list(entry_points.select(group=group))
        if isinstance(entry_points, dict):
            return list(entry_points.get(group, []))
        return [ep for ep in entry_points if getattr(ep, "group", None) == group]
    except Exception:
        return []

    return list(entry_points)


def _load_registry(group: str, builtins: dict[str, type[Any]]) -> dict[str, type[Any]]:
    registry: dict[str, type[Any]] = dict(builtins)
    for entry_point in _iter_entry_points(group):
        if entry_point.name in registry:
            continue
        try:
            loaded = entry_point.load()
        except Exception:
            continue
        registry[entry_point.name] = loaded
    return registry


def _builtin_storage_backends() -> dict[str, type[Any]]:
    from agentic_memory.store import SQLiteStore

    return {"sqlite": SQLiteStore}


def _builtin_search_backends() -> dict[str, type[Any]]:
    from agentic_memory.embedding import TFIDFEmbedding

    return {"tfidf": TFIDFEmbedding}


def get_evidence_registry() -> dict[str, type[Evidence]]:
    """Return merged evidence registry: built-ins plus any installed plugins."""
    global _evidence_registry
    if _evidence_registry is None:
        _evidence_registry = {
            name: cls
            for name, cls in _load_registry("agentic_memory.evidence", _BUILTIN_EVIDENCE).items()
        }
    return _evidence_registry


def get_storage_backend_registry() -> dict[str, type[Any]]:
    """Return discovered storage backend implementations."""
    global _storage_backend_registry
    if _storage_backend_registry is None:
        _storage_backend_registry = _load_registry(
            "agentic_memory.backends",
            _builtin_storage_backends(),
        )
    return _storage_backend_registry


def get_search_backend_registry() -> dict[str, type[Any]]:
    """Return discovered embedding / search backend implementations."""
    global _search_backend_registry
    if _search_backend_registry is None:
        _search_backend_registry = _load_registry(
            "agentic_memory.search",
            _builtin_search_backends(),
        )
    return _search_backend_registry


def reset_registry() -> None:
    """Reset all cached registries. Primarily used by tests."""
    global _evidence_registry, _storage_backend_registry, _search_backend_registry
    _evidence_registry = None
    _storage_backend_registry = None
    _search_backend_registry = None
