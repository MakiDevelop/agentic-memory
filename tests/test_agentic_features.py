"""Tests for agentic features: kind, importance, TTL, dedup, retrieval logging."""

from datetime import datetime, timedelta

from agentic_memory import ManualRef, Memory, MemoryKind, ValidationStatus


class TestKind:
    def test_add_with_kind(self, tmp_path):
        mem = Memory(str(tmp_path))
        record = mem.add("no semicolons", evidence=ManualRef("style guide"), kind=MemoryKind.RULE)
        assert record.kind == MemoryKind.RULE
        mem.close()

    def test_add_with_kind_string(self, tmp_path):
        mem = Memory(str(tmp_path))
        record = mem.add("no semicolons", evidence=ManualRef("style guide"), kind="rule")
        assert record.kind == MemoryKind.RULE
        mem.close()

    def test_default_kind_is_fact(self, tmp_path):
        mem = Memory(str(tmp_path))
        record = mem.add("uses pytest", evidence=ManualRef("README"))
        assert record.kind == MemoryKind.FACT
        mem.close()

    def test_kind_roundtrip(self, tmp_path):
        mem = Memory(str(tmp_path))
        record = mem.add("avoid dangerous patterns", evidence=ManualRef("security"), kind="antipattern")
        retrieved = mem.get(record.id)
        assert retrieved.kind == MemoryKind.ANTIPATTERN
        mem.close()

    def test_query_filter_by_kind(self, tmp_path):
        mem = Memory(str(tmp_path))
        mem.add("uses ruff", evidence=ManualRef("docs"), kind="fact")
        mem.add("no dangerous patterns allowed", evidence=ManualRef("security"), kind="rule")
        mem.add("always test edge cases", evidence=ManualRef("culture"), kind="rule")

        result = mem.query("ruff dangerous test", kind="rule")
        assert all(r.kind == MemoryKind.RULE for r in result.memories)
        mem.close()


class TestImportance:
    def test_add_with_importance(self, tmp_path):
        mem = Memory(str(tmp_path))
        record = mem.add("critical config", evidence=ManualRef("ops"), importance=3)
        assert record.importance == 3
        mem.close()

    def test_importance_clamped(self, tmp_path):
        mem = Memory(str(tmp_path))
        r1 = mem.add("too high", evidence=ManualRef("a"), importance=10)
        r2 = mem.add("too low", evidence=ManualRef("b"), importance=-5)
        assert r1.importance == 3
        assert r2.importance == 0
        mem.close()

    def test_query_filter_min_importance(self, tmp_path):
        mem = Memory(str(tmp_path))
        mem.add("low priority item", evidence=ManualRef("a"), importance=0)
        mem.add("high priority item", evidence=ManualRef("b"), importance=3)

        result = mem.query("priority item", min_importance=2)
        assert len(result.memories) == 1
        assert result.memories[0].importance == 3
        mem.close()

    def test_results_sorted_by_importance(self, tmp_path):
        mem = Memory(str(tmp_path))
        mem.add("low importance item", evidence=ManualRef("a"), importance=0)
        mem.add("high importance item", evidence=ManualRef("b"), importance=3)
        mem.add("medium importance item", evidence=ManualRef("c"), importance=1)

        result = mem.query("importance item")
        importances = [r.importance for r in result.memories]
        assert importances == sorted(importances, reverse=True)
        mem.close()


class TestTTL:
    def test_add_with_ttl(self, tmp_path):
        mem = Memory(str(tmp_path))
        record = mem.add("temp fact", evidence=ManualRef("chat"), ttl_seconds=3600)
        assert record.ttl_seconds == 3600
        assert not record.is_expired
        mem.close()

    def test_no_ttl_never_expires(self, tmp_path):
        mem = Memory(str(tmp_path))
        record = mem.add("permanent fact", evidence=ManualRef("docs"))
        assert record.ttl_seconds is None
        assert not record.is_expired
        mem.close()

    def test_expired_memory_filtered_from_query(self, tmp_path):
        mem = Memory(str(tmp_path))
        record = mem.add("old fact", evidence=ManualRef("chat"), ttl_seconds=1)
        # Force created_at to the past
        record.created_at = datetime.now() - timedelta(seconds=10)
        mem._store.save(record)

        result = mem.query("old fact")
        assert len(result.memories) == 0
        mem.close()

    def test_validate_marks_expired_as_stale(self, tmp_path):
        mem = Memory(str(tmp_path))
        record = mem.add("expiring fact", evidence=ManualRef("chat"), ttl_seconds=1)
        record.created_at = datetime.now() - timedelta(seconds=10)
        mem._store.save(record)

        problematic = mem.validate()
        assert len(problematic) == 1
        assert problematic[0].validation_status == ValidationStatus.STALE
        assert "expired" in problematic[0].validation_message.lower()
        mem.close()

    def test_status_counts_expired(self, tmp_path):
        mem = Memory(str(tmp_path))
        mem.add("permanent", evidence=ManualRef("a"))
        record = mem.add("expiring", evidence=ManualRef("b"), ttl_seconds=1)
        record.created_at = datetime.now() - timedelta(seconds=10)
        mem._store.save(record)

        s = mem.status()
        assert s["expired"] == 1
        mem.close()


class TestDedup:
    def test_duplicate_returns_existing(self, tmp_path):
        mem = Memory(str(tmp_path))
        r1 = mem.add("uses ruff for linting", evidence=ManualRef("docs"))
        r2 = mem.add("uses ruff for linting", evidence=ManualRef("other"))
        assert r1.id == r2.id  # same record returned
        mem.close()

    def test_duplicate_case_insensitive(self, tmp_path):
        mem = Memory(str(tmp_path))
        r1 = mem.add("Uses Ruff for Linting", evidence=ManualRef("docs"))
        r2 = mem.add("uses ruff for linting", evidence=ManualRef("other"))
        assert r1.id == r2.id
        mem.close()

    def test_dedup_can_be_disabled(self, tmp_path):
        mem = Memory(str(tmp_path))
        r1 = mem.add("same content here", evidence=ManualRef("a"))
        r2 = mem.add("same content here", evidence=ManualRef("b"), deduplicate=False)
        assert r1.id != r2.id  # different records
        mem.close()

    def test_different_content_not_deduped(self, tmp_path):
        mem = Memory(str(tmp_path))
        r1 = mem.add("uses ruff for linting", evidence=ManualRef("docs"))
        r2 = mem.add("uses black for formatting", evidence=ManualRef("docs"))
        assert r1.id != r2.id
        mem.close()


class TestRetrievalLog:
    def test_query_logs_retrieval(self, tmp_path):
        mem = Memory(str(tmp_path))
        mem.add("test memory", evidence=ManualRef("note"))
        mem.query("test")

        logs = mem.retrieval_stats()
        assert len(logs) == 1
        assert logs[0].query == "test"
        assert logs[0].result_count >= 0
        assert logs[0].latency_ms >= 0
        mem.close()

    def test_multiple_queries_logged(self, tmp_path):
        mem = Memory(str(tmp_path))
        mem.add("fact one", evidence=ManualRef("a"))
        mem.add("fact two", evidence=ManualRef("b"))

        mem.query("one")
        mem.query("two")
        mem.query("three")

        logs = mem.retrieval_stats()
        assert len(logs) == 3
        # Most recent first
        assert logs[0].query == "three"
        mem.close()

    def test_retrieval_log_contains_ids(self, tmp_path):
        mem = Memory(str(tmp_path))
        record = mem.add("findable memory", evidence=ManualRef("note"))
        mem.query("findable")

        logs = mem.retrieval_stats()
        assert record.id in logs[0].returned_ids
        mem.close()


class TestSchemaV4Migration:
    def test_v3_db_auto_upgrades(self, tmp_path):
        """A v3 database should auto-upgrade to v4 with new columns."""
        import json
        import sqlite3

        db_path = tmp_path / ".agentic-memory.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE memories (
                id TEXT PRIMARY KEY, content TEXT NOT NULL,
                evidence_json TEXT NOT NULL, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, confidence REAL DEFAULT 1.0,
                validation_status TEXT DEFAULT 'valid',
                validation_message TEXT DEFAULT '', tags_json TEXT DEFAULT '[]'
            );
            CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO schema_meta VALUES ('version', '3');
            CREATE VIRTUAL TABLE memories_fts USING fts5(content);
        """)
        evidence = json.dumps({"type": "manual", "note": "old"})
        conn.execute(
            "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("v3-1", "old memory", evidence,
             "2026-01-01T00:00:00", "2026-01-01T00:00:00", 1.0, "valid", "", "[]"),
        )
        conn.execute("INSERT INTO memories_fts(rowid, content) VALUES (1, 'old memory')")
        conn.commit()
        conn.close()

        mem = Memory(str(tmp_path))
        record = mem.get("v3-1")
        assert record is not None
        assert record.kind == MemoryKind.FACT  # default
        assert record.importance == 1  # default
        assert record.source_hash != ""  # backfilled
        mem.close()
