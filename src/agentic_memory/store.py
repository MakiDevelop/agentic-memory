"""SQLite storage backend for agentic-memory."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from agentic_memory.evidence import evidence_from_dict
from agentic_memory.models import MemoryKind, MemoryRecord, RetrievalLog, ValidationStatus, _content_hash
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
        if version < 3:
            self._upgrade_v3()
        if version < 4:
            self._upgrade_v4()
        if version < 5:
            self._upgrade_v5()

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

    def _upgrade_v3(self) -> None:
        """Mark schema v3: multi-evidence support (no DDL changes, evidence_json handles both dict and list)."""
        self._conn.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('version', '3')"
        )

    def _upgrade_v4(self) -> None:
        """Schema v4: kind, importance, TTL, source_hash, retrieval_logs."""
        # Add new columns (ALTER TABLE ADD COLUMN is safe in SQLite)
        for col_def in [
            "kind TEXT DEFAULT 'fact'",
            "importance INTEGER DEFAULT 1",
            "ttl_seconds INTEGER DEFAULT NULL",
            "source_hash TEXT DEFAULT ''",
        ]:
            try:
                self._conn.execute(f"ALTER TABLE memories ADD COLUMN {col_def}")
            except sqlite3.OperationalError:
                pass  # Column already exists

        # Backfill source_hash for existing records
        rows = self._conn.execute("SELECT id, content FROM memories WHERE source_hash = '' OR source_hash IS NULL").fetchall()
        for row in rows:
            self._conn.execute(
                "UPDATE memories SET source_hash = ? WHERE id = ?",
                (_content_hash(row[1]), row[0]),
            )

        # Index for dedup lookups
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(source_hash)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance)")

        # Retrieval logs table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                returned_ids TEXT NOT NULL,
                result_count INTEGER NOT NULL,
                latency_ms REAL DEFAULT 0.0,
                created_at TEXT NOT NULL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_created_at ON retrieval_logs(created_at)")

        self._conn.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('version', '4')"
        )

    def _upgrade_v5(self) -> None:
        """Schema v5: adoption_logs table for tracking which memories agents actually use."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS adoption_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                query TEXT DEFAULT '',
                agent_name TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_adoption_memory ON adoption_logs(memory_id)")
        self._conn.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('version', '5')"
        )

    def _serialize_evidence(self, evidence) -> str:
        """Serialize evidence (single or list) to JSON string."""
        if isinstance(evidence, list):
            return json.dumps([e.to_dict() for e in evidence])
        return json.dumps(evidence.to_dict())

    def _deserialize_evidence(self, raw: str):
        """Deserialize evidence JSON (handles both dict and list formats)."""
        data = json.loads(raw)
        if isinstance(data, list):
            return [evidence_from_dict(e) for e in data]
        return evidence_from_dict(data)

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
                confidence, validation_status, validation_message, tags_json,
                kind, importance, ttl_seconds, source_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.id,
                record.content,
                self._serialize_evidence(record.evidence),
                record.created_at.isoformat(),
                record.updated_at.isoformat(),
                record.confidence,
                record.validation_status.value,
                record.validation_message,
                json.dumps(record.tags),
                record.kind.value,
                record.importance,
                record.ttl_seconds,
                record.source_hash or _content_hash(record.content),
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

    def search_any(self, query: str, limit: int = 10) -> list[MemoryRecord]:
        """Full-text search using OR logic (matches any token). Used for conflict detection."""
        tokenized = tokenize_for_fts(query)
        tokens = [f'"{token}"' for token in tokenized.split() if token.strip()]
        if not tokens:
            return []
        safe_query = " OR ".join(tokens)
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

    def find_by_hash(self, source_hash: str) -> MemoryRecord | None:
        """Find a memory by content hash (for deduplication)."""
        row = self._conn.execute(
            "SELECT * FROM memories WHERE source_hash = ?", (source_hash,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def log_retrieval(self, log: RetrievalLog) -> None:
        """Record a retrieval event."""
        self._conn.execute(
            """INSERT INTO retrieval_logs (query, returned_ids, result_count, latency_ms, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                log.query,
                json.dumps(log.returned_ids),
                log.result_count,
                log.latency_ms,
                log.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    def get_retrieval_stats(self, limit: int = 100) -> list[RetrievalLog]:
        """Get recent retrieval logs."""
        rows = self._conn.execute(
            "SELECT * FROM retrieval_logs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            RetrievalLog(
                query=row["query"],
                returned_ids=json.loads(row["returned_ids"]),
                result_count=row["result_count"],
                latency_ms=row["latency_ms"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def log_adoption(self, memory_id: str, query: str = "", agent_name: str = "") -> None:
        """Record that an agent adopted (used) a memory."""
        self._conn.execute(
            "INSERT INTO adoption_logs (memory_id, query, agent_name, created_at) VALUES (?, ?, ?, ?)",
            (memory_id, query, agent_name, datetime.now().isoformat()),
        )
        self._conn.commit()

    def get_adoption_counts(self) -> dict[str, int]:
        """Get adoption count per memory ID."""
        rows = self._conn.execute(
            "SELECT memory_id, COUNT(*) as cnt FROM adoption_logs GROUP BY memory_id"
        ).fetchall()
        return {row["memory_id"]: row["cnt"] for row in rows}

    def get_adoption_total(self) -> int:
        """Get total adoption count."""
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM adoption_logs").fetchone()
        return row["cnt"]

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        cols = row.keys()
        kind_val = row["kind"] if "kind" in cols else "fact"
        importance_val = row["importance"] if "importance" in cols else 1
        ttl_val = row["ttl_seconds"] if "ttl_seconds" in cols else None
        hash_val = row["source_hash"] if "source_hash" in cols else ""
        return MemoryRecord(
            id=row["id"],
            content=row["content"],
            evidence=self._deserialize_evidence(row["evidence_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            confidence=row["confidence"],
            validation_status=ValidationStatus(row["validation_status"]),
            validation_message=row["validation_message"],
            tags=json.loads(row["tags_json"]),
            kind=MemoryKind(kind_val),
            importance=importance_val,
            ttl_seconds=ttl_val,
            source_hash=hash_val,
        )
