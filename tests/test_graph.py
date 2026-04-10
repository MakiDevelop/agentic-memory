"""Tests for Memory Graph — typed relationships between memories."""

from __future__ import annotations

import sqlite3

import pytest

from agentic_memory import ManualRef, Memory, RelationType


@pytest.fixture()
def mem(tmp_path):
    m = Memory(str(tmp_path))
    yield m
    m.close()


@pytest.fixture()
def three_memories(mem):
    """Create three memories and return their IDs."""
    r1 = mem.add("Project uses ruff for linting", evidence=ManualRef("team decision"))
    r2 = mem.add("Project uses black for formatting", evidence=ManualRef("old convention"))
    r3 = mem.add("Project uses ruff for both linting and formatting", evidence=ManualRef("new convention"))
    return r1.id, r2.id, r3.id


class TestEdgeCRUD:
    def test_add_edge_contradicts(self, mem, three_memories):
        a, b, _ = three_memories
        edge = mem.add_relation(a, b, "contradicts")
        assert edge.source_id == a
        assert edge.target_id == b
        assert edge.relation == RelationType.CONTRADICTS

    def test_add_edge_supports(self, mem, three_memories):
        a, _, c = three_memories
        edge = mem.add_relation(a, c, "supports")
        assert edge.relation == RelationType.SUPPORTS

    def test_add_edge_supersedes(self, mem, three_memories):
        _, b, c = three_memories
        edge = mem.add_relation(c, b, "supersedes")
        assert edge.relation == RelationType.SUPERSEDES

    def test_add_edge_depends_on(self, mem, three_memories):
        a, b, _ = three_memories
        edge = mem.add_relation(a, b, "depends_on")
        assert edge.relation == RelationType.DEPENDS_ON

    def test_self_reference_raises(self, mem, three_memories):
        a, _, _ = three_memories
        with pytest.raises(ValueError, match="self-referencing"):
            mem.add_relation(a, a, "supports")

    def test_duplicate_edge_raises(self, mem, three_memories):
        a, b, _ = three_memories
        mem.add_relation(a, b, "supports")
        with pytest.raises(sqlite3.IntegrityError):
            mem.add_relation(a, b, "supports")

    def test_supersedes_cycle_detection(self, mem, three_memories):
        a, b, _ = three_memories
        mem.add_relation(a, b, "supersedes")
        with pytest.raises(ValueError, match="Cycle detected"):
            mem.add_relation(b, a, "supersedes")

    def test_remove_edge(self, mem, three_memories):
        a, b, _ = three_memories
        mem.add_relation(a, b, "contradicts")
        assert mem.remove_relation(a, b, "contradicts") is True

    def test_remove_nonexistent_edge(self, mem, three_memories):
        a, b, _ = three_memories
        assert mem.remove_relation(a, b, "contradicts") is False

    def test_nonexistent_source_raises(self, mem, three_memories):
        _, b, _ = three_memories
        with pytest.raises(ValueError, match="Source memory not found"):
            mem.add_relation("nonexistent", b, "supports")

    def test_nonexistent_target_raises(self, mem, three_memories):
        a, _, _ = three_memories
        with pytest.raises(ValueError, match="Target memory not found"):
            mem.add_relation(a, "nonexistent", "supports")


class TestEdgeQueries:
    def test_get_edges_outgoing(self, mem, three_memories):
        a, b, c = three_memories
        mem.add_relation(a, b, "supports")
        mem.add_relation(a, c, "supports")
        edges = mem.get_relations(a, direction="outgoing")
        assert len(edges) == 2
        assert all(e.source_id == a for e in edges)

    def test_get_edges_incoming(self, mem, three_memories):
        a, b, _ = three_memories
        mem.add_relation(a, b, "supports")
        edges = mem.get_relations(b, direction="incoming")
        assert len(edges) == 1
        assert edges[0].target_id == b

    def test_get_edges_both(self, mem, three_memories):
        a, b, c = three_memories
        mem.add_relation(a, b, "supports")
        mem.add_relation(c, a, "depends_on")
        edges = mem.get_relations(a, direction="both")
        assert len(edges) == 2

    def test_get_edges_by_relation(self, mem, three_memories):
        a, b, c = three_memories
        mem.add_relation(a, b, "supports")
        mem.add_relation(a, c, "contradicts")
        edges = mem.get_relations(a, relation="supports", direction="outgoing")
        assert len(edges) == 1
        assert edges[0].relation == RelationType.SUPPORTS

    def test_get_edges_empty(self, mem, three_memories):
        a, _, _ = three_memories
        assert mem.get_relations(a) == []


class TestTraversal:
    def test_traverse_simple_chain(self, mem, three_memories):
        a, b, c = three_memories
        mem.add_relation(a, b, "supersedes")
        mem.add_relation(b, c, "supersedes")
        records = mem.traverse(a, "supersedes", max_depth=3)
        assert len(records) == 2
        ids = [r.id for r in records]
        assert b in ids
        assert c in ids

    def test_traverse_max_depth(self, mem, three_memories):
        a, b, c = three_memories
        mem.add_relation(a, b, "supersedes")
        mem.add_relation(b, c, "supersedes")
        records = mem.traverse(a, "supersedes", max_depth=1)
        assert len(records) == 1
        assert records[0].id == b

    def test_traverse_branching(self, mem, three_memories):
        a, b, c = three_memories
        mem.add_relation(a, b, "supports")
        mem.add_relation(a, c, "supports")
        records = mem.traverse(a, "supports")
        assert len(records) == 2

    def test_traverse_no_results(self, mem, three_memories):
        a, _, _ = three_memories
        records = mem.traverse(a, "supports")
        assert records == []


class TestSupersede:
    def test_supersede_creates_new_memory(self, mem, three_memories):
        _, b, _ = three_memories
        new = mem.supersede(b, "Updated convention", evidence=ManualRef("latest"))
        assert new.content == "Updated convention"

    def test_supersede_creates_edge(self, mem, three_memories):
        _, b, _ = three_memories
        new = mem.supersede(b, "Updated convention", evidence=ManualRef("latest"))
        edges = mem.get_relations(new.id, relation="supersedes", direction="outgoing")
        assert len(edges) == 1
        assert edges[0].target_id == b

    def test_supersede_marks_old_memory(self, mem, three_memories):
        _, b, _ = three_memories
        new = mem.supersede(b, "Updated convention", evidence=ManualRef("latest"))
        old = mem.get(b)
        assert old.superseded_by == new.id

    def test_supersede_nonexistent_raises(self, mem):
        with pytest.raises(ValueError, match="not found"):
            mem.supersede("nonexistent", "new content", evidence=ManualRef("test"))


class TestDeleteCleansEdges:
    def test_delete_removes_outgoing_edges(self, mem, three_memories):
        a, b, _ = three_memories
        mem.add_relation(a, b, "supports")
        mem.delete(a)
        edges = mem.get_relations(b, direction="incoming")
        assert len(edges) == 0

    def test_delete_removes_incoming_edges(self, mem, three_memories):
        a, b, _ = three_memories
        mem.add_relation(a, b, "supports")
        mem.delete(b)
        edges = mem.get_relations(a, direction="outgoing")
        assert len(edges) == 0
