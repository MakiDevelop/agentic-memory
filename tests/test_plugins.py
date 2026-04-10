"""Tests for plugin discovery registries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agentic_memory.embedding import TFIDFEmbedding
from agentic_memory.evidence import Evidence, FileRef, GitCommitRef, ManualRef, URLRef, evidence_from_dict
from agentic_memory.models import MemoryRecord, ValidationStatus
from agentic_memory.plugins import (
    get_evidence_registry,
    get_search_backend_registry,
    get_storage_backend_registry,
    reset_registry,
)
from agentic_memory.store import SQLiteStore


class DummyEntryPoint:
    """Simple test double for importlib metadata entry points."""

    def __init__(self, name: str, value: Any, *, should_raise: bool = False):
        self.name = name
        self._value = value
        self._should_raise = should_raise

    def load(self) -> Any:
        if self._should_raise:
            raise RuntimeError("broken plugin")
        return self._value


@dataclass
class JiraTicketRef(Evidence):
    """Minimal plugin evidence type used for registry tests."""

    ticket_id: str

    def validate(self, repo_path: str) -> tuple[ValidationStatus, str]:
        return ValidationStatus.VALID, f"Ticket {self.ticket_id}"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "jira", "ticket_id": self.ticket_id}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JiraTicketRef:
        return cls(ticket_id=data["ticket_id"])

    def short_label(self) -> str:
        return self.ticket_id


class DummyStorageBackend:
    """Minimal external storage backend for registry tests."""


class DummySearchBackend:
    """Minimal external search backend for registry tests."""


@pytest.fixture(autouse=True)
def clear_registries():
    reset_registry()
    yield
    reset_registry()


def test_builtin_evidence_registry():
    registry = get_evidence_registry()
    assert registry["file"] is FileRef
    assert registry["git_commit"] is GitCommitRef
    assert registry["url"] is URLRef
    assert registry["manual"] is ManualRef


@pytest.mark.parametrize(
    ("payload", "expected_type"),
    [
        ({"type": "file", "path": "README.md"}, FileRef),
        ({"type": "git_commit", "sha": "abc123"}, GitCommitRef),
        ({"type": "url", "url": "https://example.com"}, URLRef),
        ({"type": "manual", "note": "note"}, ManualRef),
    ],
)
def test_evidence_from_dict_uses_registry(payload, expected_type):
    evidence = evidence_from_dict(payload)
    assert isinstance(evidence, expected_type)


def test_unknown_evidence_type_raises():
    with pytest.raises(ValueError, match="Unknown evidence type"):
        evidence_from_dict({"type": "unknown"})


def test_registry_caching():
    registry_1 = get_evidence_registry()
    registry_2 = get_evidence_registry()
    assert registry_1 is registry_2


def test_reset_registry_clears_cache():
    registry_1 = get_evidence_registry()
    reset_registry()
    registry_2 = get_evidence_registry()
    assert registry_1 is not registry_2


def test_plugin_evidence_entry_point_is_loaded(monkeypatch):
    def fake_entry_points(*, group: str):
        if group == "agentic_memory.evidence":
            return [DummyEntryPoint("jira", JiraTicketRef)]
        return []

    monkeypatch.setattr("agentic_memory.plugins.importlib.metadata.entry_points", fake_entry_points)
    registry = get_evidence_registry()

    assert registry["jira"] is JiraTicketRef
    assert isinstance(evidence_from_dict({"type": "jira", "ticket_id": "PROJ-123"}), JiraTicketRef)


def test_broken_plugin_is_skipped(monkeypatch):
    def fake_entry_points(*, group: str):
        if group == "agentic_memory.evidence":
            return [DummyEntryPoint("broken", object, should_raise=True)]
        return []

    monkeypatch.setattr("agentic_memory.plugins.importlib.metadata.entry_points", fake_entry_points)
    registry = get_evidence_registry()

    assert "broken" not in registry


def test_builtin_evidence_cannot_be_overridden(monkeypatch):
    def fake_entry_points(*, group: str):
        if group == "agentic_memory.evidence":
            return [DummyEntryPoint("file", JiraTicketRef)]
        return []

    monkeypatch.setattr("agentic_memory.plugins.importlib.metadata.entry_points", fake_entry_points)
    registry = get_evidence_registry()

    assert registry["file"] is FileRef


def test_entry_point_fallback_supports_dict_style(monkeypatch):
    def fake_entry_points():
        return {"agentic_memory.evidence": [DummyEntryPoint("jira", JiraTicketRef)]}

    monkeypatch.setattr("agentic_memory.plugins.importlib.metadata.entry_points", fake_entry_points)
    registry = get_evidence_registry()

    assert registry["jira"] is JiraTicketRef


def test_storage_backend_registry_includes_builtin_and_plugin(monkeypatch):
    def fake_entry_points(*, group: str):
        if group == "agentic_memory.backends":
            return [DummyEntryPoint("dummy", DummyStorageBackend)]
        return []

    monkeypatch.setattr("agentic_memory.plugins.importlib.metadata.entry_points", fake_entry_points)
    registry = get_storage_backend_registry()

    assert registry["sqlite"] is SQLiteStore
    assert registry["dummy"] is DummyStorageBackend


def test_search_backend_registry_includes_builtin_and_plugin(monkeypatch):
    def fake_entry_points(*, group: str):
        if group == "agentic_memory.search":
            return [DummyEntryPoint("dummy", DummySearchBackend)]
        return []

    monkeypatch.setattr("agentic_memory.plugins.importlib.metadata.entry_points", fake_entry_points)
    registry = get_search_backend_registry()

    assert registry["tfidf"] is TFIDFEmbedding
    assert registry["dummy"] is DummySearchBackend


def test_registry_functions_return_plain_class_mappings():
    evidence_registry = get_evidence_registry()
    storage_registry = get_storage_backend_registry()
    search_registry = get_search_backend_registry()

    assert all(isinstance(name, str) and isinstance(cls, type) for name, cls in evidence_registry.items())
    assert all(isinstance(name, str) and isinstance(cls, type) for name, cls in storage_registry.items())
    assert all(isinstance(name, str) and isinstance(cls, type) for name, cls in search_registry.items())


def test_plugin_registry_does_not_break_memory_record_types():
    record = MemoryRecord(content="test", evidence=ManualRef("note"))
    assert isinstance(record.evidence, ManualRef)
