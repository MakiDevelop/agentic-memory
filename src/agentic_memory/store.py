"""SQLite storage backend for agentic-memory."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from agentic_memory.evidence import evidence_from_dict
from agentic_memory.models import MemoryRecord, ValidationStatus
from agentic_memory.tokenizer import has_cjk, is_jieba_available, tokenize_for_fts


@dataclass(frozen=True)
class VectorSearchHit:
    """A memory record with its vector similarity score."""

    record: MemoryRecord
    score: float


def _serialize_vector(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def _deserialize_vector(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


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

            CREATE TABLE IF NOT EXISTS memory_embeddings (
                memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                model_id TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vector_blob BLOB NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS embedding_provider_state (
                provider_key TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                dim INTEGER NOT NULL,
                state_blob BLOB NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)

        version = self._get_schema_version()
        if version < 2:
            self._upgrade_fts_v2()

        self._conn.commit()

    def _get_schema_version(self) -> int:
        try:
            row = self._conn.execute("SELECT value FROM schema_meta WHERE key='version'").fetchone()
            return int(row[0]) if row else 0
        except sqlite3.OperationalError:
            return 0

    def _upgrade_fts_v2(self) -> None:
        """Upgrade FTS5 to standalone table with CJK tokenization support."""
        self._conn.executescript("""
            DROP TRIGGER IF EXISTS memories_ai;
            DROP TRIGGER IF EXISTS memories_ad;
            DROP TRIGGER IF EXISTS memories_au;
            DROP TABLE IF EXISTS memories_fts;
            CREATE VIRTUAL TABLE memories_fts USING fts5(content);
        """)

        rows = self._conn.execute("SELECT rowid, content FROM memories").fetchall()
        for row in rows:
            tokenized = tokenize_for_fts(row[1])
            self._conn.execute(
                "INSERT INTO memories_fts(rowid, content) VALUES (?, ?)",
                (row[0], tokenized),
            )

        self._conn.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('version', '2')"
        )

    def save(self, record: MemoryRecord) -> None:
        """Insert or update a memory record."""
        # Remove old FTS5 entry if updating
        existing = self._conn.execute(
            "SELECT rowid FROM memories WHERE id = ?", (record.id,)
        ).fetchone()
        if existing:
            self._conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (existing[0],))

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

        # Insert tokenized content into FTS5
        row = self._conn.execute(
            "SELECT rowid FROM memories WHERE id = ?", (record.id,)
        ).fetchone()
        tokenized = tokenize_for_fts(record.content)
        self._conn.execute(
            "INSERT INTO memories_fts(rowid, content) VALUES (?, ?)",
            (row[0], tokenized),
        )

        self._conn.commit()

    def get(self, memory_id: str) -> MemoryRecord | None:
        """Get a memory by ID."""
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def search(self, query: str, limit: int = 10) -> list[MemoryRecord]:
        """Full-text search for memories. Tokenizes CJK text automatically."""
        tokenized = tokenize_for_fts(query)
        tokens = [f'"{token}"' for token in tokenized.split() if token.strip()]
        if not tokens:
            return []
        # CJK character-level fallback (no jieba): use OR to avoid requiring all chars
        # With jieba: use AND (word-level tokens are meaningful)
        if has_cjk(query) and not is_jieba_available():
            safe_query = " OR ".join(tokens)
        else:
            safe_query = " ".join(tokens)
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
        row = self._conn.execute(
            "SELECT rowid FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if not row:
            return False
        self._conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (row[0],))
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()
        return True

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

    def save_embedding(self, memory_id: str, model_id: str, vector: np.ndarray) -> None:
        """Store an embedding vector for a memory."""
        dim = int(vector.shape[0])
        self._conn.execute(
            """INSERT OR REPLACE INTO memory_embeddings
               (memory_id, model_id, dim, vector_blob, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (memory_id, model_id, dim, _serialize_vector(vector), datetime.now().isoformat()),
        )
        self._conn.commit()

    def vector_search(
        self, query_vector: np.ndarray, *, model_id: str, limit: int = 10
    ) -> list[VectorSearchHit]:
        """Brute-force cosine similarity search over stored embeddings."""
        rows = self._conn.execute(
            """SELECT m.*, e.dim, e.vector_blob
               FROM memories m
               JOIN memory_embeddings e ON m.id = e.memory_id
               WHERE e.model_id = ?""",
            (model_id,),
        ).fetchall()

        q = np.asarray(query_vector, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0:
            return []

        hits: list[VectorSearchHit] = []
        for row in rows:
            vec = _deserialize_vector(row["vector_blob"], row["dim"])
            denom = float(np.linalg.norm(vec)) * q_norm
            score = float(np.dot(q, vec) / denom) if denom > 0 else 0.0
            hits.append(VectorSearchHit(record=self._row_to_record(row), score=score))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]

    def save_provider_state(self, provider_key: str, model_id: str, dim: int, state: bytes) -> None:
        """Persist embedding provider state (e.g., TF-IDF vocabulary)."""
        self._conn.execute(
            """INSERT OR REPLACE INTO embedding_provider_state
               (provider_key, model_id, dim, state_blob, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (provider_key, model_id, dim, state, datetime.now().isoformat()),
        )
        self._conn.commit()

    def load_provider_state(self, provider_key: str) -> bytes | None:
        """Load persisted embedding provider state."""
        row = self._conn.execute(
            "SELECT state_blob FROM embedding_provider_state WHERE provider_key = ?",
            (provider_key,),
        ).fetchone()
        return bytes(row["state_blob"]) if row else None

    def has_embeddings(self, model_id: str) -> bool:
        """Check if any embeddings exist for the given model."""
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM memory_embeddings WHERE model_id = ?",
            (model_id,),
        ).fetchone()
        return row["cnt"] > 0

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
