"""Tests for the file-based schema migration engine."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agentic_memory.migration import backup_database, get_current_version, get_migration_files, run_migrations
from agentic_memory.store import SQLiteStore


def _create_v0_db(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE memories (
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

        CREATE TABLE memory_embeddings (
            memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
            model_id TEXT NOT NULL,
            dim INTEGER NOT NULL,
            vector_blob BLOB NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE embedding_provider_state (
            provider_key TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            dim INTEGER NOT NULL,
            state_blob BLOB NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    conn.execute(
        "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "legacy-1",
            "legacy searchable memory",
            json.dumps({"type": "manual", "note": "legacy"}),
            "2026-01-01T00:00:00",
            "2026-01-01T00:00:00",
            1.0,
            "valid",
            "",
            "[]",
        ),
    )
    conn.commit()
    conn.close()


def _create_v3_db(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            validation_status TEXT DEFAULT 'valid',
            validation_message TEXT DEFAULT '',
            tags_json TEXT DEFAULT '[]'
        );
        CREATE TABLE memory_embeddings (
            memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
            model_id TEXT NOT NULL,
            dim INTEGER NOT NULL,
            vector_blob BLOB NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE embedding_provider_state (
            provider_key TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            dim INTEGER NOT NULL,
            state_blob BLOB NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_meta VALUES ('version', '3');
        CREATE VIRTUAL TABLE memories_fts USING fts5(content);
    """)
    conn.execute(
        "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "v3-1",
            "pre migration record",
            json.dumps({"type": "manual", "note": "v3"}),
            "2026-01-01T00:00:00",
            "2026-01-01T00:00:00",
            1.0,
            "valid",
            "",
            "[]",
        ),
    )
    conn.execute("INSERT INTO memories_fts(rowid, content) VALUES (1, 'pre migration record')")
    conn.commit()
    conn.close()


def test_get_migration_files_discovers_bundled_versions():
    migrations = get_migration_files()
    assert {2, 3, 4, 5}.issubset(migrations)
    assert max(migrations) >= 5


def test_get_current_version_missing_table_returns_zero():
    conn = sqlite3.connect(":memory:")
    try:
        assert get_current_version(conn) == 0
    finally:
        conn.close()


def test_backup_database_returns_none_for_missing_path(tmp_path):
    missing = tmp_path / "missing.db"
    assert backup_database(missing, 0) is None


def test_fresh_db_gets_latest_version(tmp_path):
    latest_version = max(get_migration_files())
    db_path = tmp_path / "fresh.db"
    store = SQLiteStore(db_path)
    try:
        version = store._conn.execute("SELECT value FROM schema_meta WHERE key = 'version'").fetchone()[0]
        assert version == str(latest_version)

        tables = {
            row[0]
            for row in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')"
            ).fetchall()
        }
        assert "memories_fts" in tables
        assert "retrieval_logs" in tables
        assert "adoption_logs" in tables
        if latest_version >= 6:
            assert "memory_edges" in tables
    finally:
        store.close()


def test_upgrade_from_v0_preserves_data_and_rebuilds_fts(tmp_path):
    db_path = tmp_path / "legacy.db"
    _create_v0_db(db_path)

    store = SQLiteStore(db_path)
    try:
        record = store.get("legacy-1")
        assert record is not None
        assert record.content == "legacy searchable memory"
        assert record.source_hash != ""

        results = store.search("searchable")
        assert [hit.id for hit in results] == ["legacy-1"]
    finally:
        store.close()


def test_backup_created_before_upgrade(tmp_path):
    db_path = tmp_path / "legacy.db"
    _create_v0_db(db_path)

    store = SQLiteStore(db_path)
    store.close()

    backup_path = db_path.with_suffix(".db.v0-backup")
    assert backup_path.exists()


def test_no_migration_needed_is_noop(tmp_path):
    latest_version = max(get_migration_files())
    db_path = tmp_path / "noop.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO schema_meta VALUES ('version', ?)", (str(latest_version),))
    conn.commit()

    try:
        assert run_migrations(conn, db_path=db_path) == latest_version
    finally:
        conn.close()

    assert not db_path.with_suffix(f".db.v{latest_version}-backup").exists()


def test_v3_upgrade_preserves_existing_rows(tmp_path):
    db_path = tmp_path / "v3.db"
    _create_v3_db(db_path)

    store = SQLiteStore(db_path)
    try:
        record = store.get("v3-1")
        assert record is not None
        assert record.content == "pre migration record"
        assert record.importance == 1
        assert record.source_hash != ""
    finally:
        store.close()


def test_python_hooks_called(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    called: list[int] = []

    def hook(connection: sqlite3.Connection) -> None:
        called.append(2)
        connection.execute("CREATE TABLE hook_table (value INTEGER)")

    monkeypatch.setattr("agentic_memory.migration.get_migration_files", lambda: {2: []})
    try:
        assert run_migrations(conn, python_hooks={2: hook}) == 2
        assert called == [2]
        assert conn.execute("SELECT value FROM schema_meta WHERE key = 'version'").fetchone()[0] == "2"
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'hook_table'"
        ).fetchone()
    finally:
        conn.close()


def test_comment_only_migration_still_updates_version(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    monkeypatch.setattr("agentic_memory.migration.get_migration_files", lambda: {3: ["-- comment only"]})
    try:
        assert run_migrations(conn) == 3
        assert conn.execute("SELECT value FROM schema_meta WHERE key = 'version'").fetchone()[0] == "3"
    finally:
        conn.close()
