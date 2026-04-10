"""Memory Graph — typed relationships between memories.

Supports four relation types:
- contradicts: two memories express conflicting information
- supports: one memory reinforces another
- supersedes: a newer memory replaces an older one
- depends_on: one memory requires another for context
"""

from __future__ import annotations

import json
import sqlite3
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class RelationType(Enum):
    """Types of directed relationships between memories."""

    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    SUPERSEDES = "supersedes"
    DEPENDS_ON = "depends_on"


@dataclass
class MemoryEdge:
    """A directed relationship between two memories."""

    id: int
    source_id: str
    target_id: str
    relation: RelationType
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class MemoryGraph:
    """Graph operations over the memory_edges table.

    Uses the same SQLite connection as SQLiteStore.
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: RelationType | str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEdge:
        """Add a directed edge between two memories.

        Raises:
            ValueError: Self-reference or supersedes cycle detected.
            sqlite3.IntegrityError: Duplicate edge or missing memory FK.
        """
        if isinstance(relation, str):
            relation = RelationType(relation)

        if source_id == target_id:
            raise ValueError("Cannot create self-referencing edge")

        if relation == RelationType.SUPERSEDES:
            reverse = self._conn.execute(
                "SELECT id FROM memory_edges WHERE source_id = ? AND target_id = ? AND relation = 'supersedes'",
                (target_id, source_id),
            ).fetchone()
            if reverse:
                raise ValueError(f"Cycle detected: {target_id} already supersedes {source_id}")

        now = datetime.now().isoformat()
        meta_json = json.dumps(metadata or {})

        cursor = self._conn.execute(
            (
                "INSERT INTO memory_edges "
                "(source_id, target_id, relation, metadata_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)"
            ),
            (source_id, target_id, relation.value, meta_json, now),
        )
        self._conn.commit()

        return MemoryEdge(
            id=cursor.lastrowid,
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            metadata=metadata or {},
            created_at=datetime.fromisoformat(now),
        )

    def remove_edge(self, source_id: str, target_id: str, relation: RelationType | str) -> bool:
        """Remove a specific edge. Returns True if it existed."""
        if isinstance(relation, str):
            relation = RelationType(relation)

        cursor = self._conn.execute(
            "DELETE FROM memory_edges WHERE source_id = ? AND target_id = ? AND relation = ?",
            (source_id, target_id, relation.value),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def get_edges(
        self,
        memory_id: str,
        relation: RelationType | str | None = None,
        direction: str = "both",
    ) -> list[MemoryEdge]:
        """Get edges for a memory, optionally filtered by relation and direction.

        Args:
            direction: "outgoing" (as source), "incoming" (as target), or "both".
        """
        if isinstance(relation, str):
            relation = RelationType(relation)

        conditions: list[str] = []
        params: list[Any] = []

        if direction in ("outgoing", "both"):
            conditions.append("source_id = ?")
            params.append(memory_id)
        if direction in ("incoming", "both"):
            conditions.append("target_id = ?")
            params.append(memory_id)

        where = " OR ".join(conditions)
        if relation is not None:
            where = f"({where}) AND relation = ?"
            params.append(relation.value)

        rows = self._conn.execute(
            f"SELECT * FROM memory_edges WHERE {where} ORDER BY created_at",
            params,
        ).fetchall()

        return [self._row_to_edge(row) for row in rows]

    def traverse(
        self,
        start_id: str,
        relation: RelationType | str,
        max_depth: int = 3,
        direction: str = "outgoing",
    ) -> list[str]:
        """BFS traversal following edges of a specific relation type.

        Returns memory IDs reachable from start_id (excluding start_id itself).
        """
        if isinstance(relation, str):
            relation = RelationType(relation)

        visited: set[str] = {start_id}
        queue: deque[tuple[str, int]] = deque([(start_id, 0)])
        result: list[str] = []

        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue

            if direction == "outgoing":
                col_from, col_to = "source_id", "target_id"
            else:
                col_from, col_to = "target_id", "source_id"

            rows = self._conn.execute(
                f"SELECT {col_to} FROM memory_edges WHERE {col_from} = ? AND relation = ?",
                (current_id, relation.value),
            ).fetchall()

            for row in rows:
                neighbor_id = row[0]
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    result.append(neighbor_id)
                    queue.append((neighbor_id, depth + 1))

        return result

    def remove_edges_for_memory(self, memory_id: str) -> int:
        """Remove all edges involving a memory. Returns count removed."""
        cursor = self._conn.execute(
            "DELETE FROM memory_edges WHERE source_id = ? OR target_id = ?",
            (memory_id, memory_id),
        )
        self._conn.commit()
        return cursor.rowcount

    def _row_to_edge(self, row: sqlite3.Row) -> MemoryEdge:
        return MemoryEdge(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            relation=RelationType(row["relation"]),
            metadata=json.loads(row["metadata_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )
