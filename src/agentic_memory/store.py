"""SQLite storage backend for agentic-memory."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from agentic_memory.evidence import evidence_from_dict
from agentic_memory.models import MemoryRecord, ValidationStatus


class SQLiteStore:
    """SQLite-based memory storage with FTS5 full-text search."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                validation_status TEXT DEFAULT 'unchecked',
                validation_message TEXT DEFAULT '',
                tags_json TEXT DEFAULT '[]'
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content,
                content='memories',
                content_rowid='rowid'
            );

            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content)
                VALUES (new.rowid, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
                INSERT INTO memories_fts(rowid, content)
                VALUES (new.rowid, new.content);
            END;
        """)
        self._conn.commit()

    def save(self, record: MemoryRecord) -> None:
        """Insert or update a memory record."""
        self._conn.execute(
            """INSERT OR REPLACE INTO memories
               (id, content, evidence_json, created_at, updated_at,
                confidence, validation_status, validation_message, tags_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.id,
                record.content,
                json.dumps(record.evidence.to_dict()),
                record.created_at.isoformat(),
                record.updated_at.isoformat(),
                record.confidence,
                record.validation_status.value,
                record.validation_message,
                json.dumps(record.tags),
            ),
        )
        self._conn.commit()

    def get(self, memory_id: str) -> MemoryRecord | None:
        """Get a memory by ID."""
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def search(self, query: str, limit: int = 10) -> list[MemoryRecord]:
        """Full-text search for memories."""
        # Quote each token to prevent FTS5 interpreting special chars as operators
        safe_query = " ".join(f'"{token}"' for token in query.split())
        rows = self._conn.execute(
            """SELECT m.* FROM memories m
               JOIN memories_fts fts ON m.rowid = fts.rowid
               WHERE memories_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (safe_query, limit),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_all(self, limit: int = 100) -> list[MemoryRecord]:
        """List all memories."""
        rows = self._conn.execute(
            "SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        cursor = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def update_validation(
        self, memory_id: str, status: ValidationStatus, message: str, confidence: float
    ) -> None:
        """Update validation status for a memory."""
        self._conn.execute(
            """UPDATE memories
               SET validation_status = ?, validation_message = ?,
                   confidence = ?, updated_at = ?
               WHERE id = ?""",
            (status.value, message, confidence, datetime.now().isoformat(), memory_id),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        evidence_data = json.loads(row["evidence_json"])
        return MemoryRecord(
            id=row["id"],
            content=row["content"],
            evidence=evidence_from_dict(evidence_data),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            confidence=row["confidence"],
            validation_status=ValidationStatus(row["validation_status"]),
            validation_message=row["validation_message"],
            tags=json.loads(row["tags_json"]),
        )
