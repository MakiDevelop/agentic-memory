"""Tests for CJK (Chinese/Japanese/Korean) text support."""

from unittest import mock

from agentic_memory import Memory, ManualRef, TFIDFEmbedding
from agentic_memory.tokenizer import has_cjk, tokenize_for_fts, _CJK_STOP_CHARS


class TestTokenizer:
    def test_has_cjk_chinese(self):
        assert has_cjk("使用 ruff 做 linting")

    def test_has_cjk_english(self):
        assert not has_cjk("uses ruff for linting")

    def test_has_cjk_mixed(self):
        assert has_cjk("project 使用 pytest")

    def test_tokenize_english_passthrough(self):
        text = "uses ruff for linting"
        assert tokenize_for_fts(text) == text

    def test_tokenize_chinese_segments(self):
        result = tokenize_for_fts("使用pytest做測試")
        # Should contain segmented Chinese words separated by spaces
        assert " " in result
        # Should preserve English tokens
        assert "pytest" in result

    def test_tokenize_chinese_sentence(self):
        result = tokenize_for_fts("這個專案使用結構化日誌")
        tokens = result.split()
        # Should have multiple tokens (not one big string)
        assert len(tokens) > 1

    def test_stop_chars_filtered_in_fallback(self):
        """Without jieba, CJK stop characters should be filtered."""
        import agentic_memory.tokenizer as tok

        with mock.patch.object(tok, "_jieba_available", False):
            result = tok.tokenize_for_fts("評分用什麼模型")
            # "用", "什", "麼" are stop chars, should be filtered
            assert "什" not in result.split()
            assert "麼" not in result.split()
            assert "用" not in result.split()
            # Content words should remain
            assert "評" in result.split()
            assert "模" in result.split()

    def test_multiword_query_without_jieba(self, tmp_path):
        """Without jieba, CJK queries should still find results via OR."""
        import agentic_memory.tokenizer as tok
        import agentic_memory.store as store_mod

        # Both store and query without jieba — consistent char-level tokenization
        with mock.patch.object(tok, "_jieba_available", False), \
             mock.patch.object(store_mod, "is_jieba_available", return_value=False):
            mem = Memory(str(tmp_path))
            mem.add("9 維 LLM 評分系統 qwen3 模型", evidence=ManualRef("docs"))
            result = mem.query("評分用什麼模型")
            assert len(result.memories) > 0
        mem.close()


class TestFTS5Chinese:
    def test_query_chinese_finds_memory(self, tmp_path):
        mem = Memory(str(tmp_path))
        mem.add("這個專案使用 ruff 做程式碼檢查", evidence=ManualRef("docs"))
        result = mem.query("程式碼檢查")
        assert len(result.memories) > 0
        assert "ruff" in result.memories[0].content
        mem.close()

    def test_query_chinese_mixed(self, tmp_path):
        mem = Memory(str(tmp_path))
        mem.add("使用 pytest 做單元測試", evidence=ManualRef("docs"))
        mem.add("部署到 production 環境", evidence=ManualRef("docs"))
        result = mem.query("測試")
        assert len(result.memories) > 0
        assert "pytest" in result.memories[0].content
        mem.close()

    def test_query_english_still_works(self, tmp_path):
        mem = Memory(str(tmp_path))
        mem.add("Uses ruff for linting", evidence=ManualRef("docs"))
        result = mem.query("ruff linting")
        assert len(result.memories) > 0
        mem.close()

    def test_delete_with_cjk_content(self, tmp_path):
        mem = Memory(str(tmp_path))
        record = mem.add("中文記憶測試", evidence=ManualRef("test"))
        assert mem.delete(record.id)
        result = mem.query("中文")
        assert len(result.memories) == 0
        mem.close()

    def test_update_cjk_memory(self, tmp_path):
        """Ensure FTS5 stays consistent after updating a memory with CJK content."""
        mem = Memory(str(tmp_path))
        record = mem.add("舊的設定", evidence=ManualRef("v1"))
        # Overwrite via store
        from agentic_memory.models import MemoryRecord, ValidationStatus
        from datetime import datetime

        updated = MemoryRecord(
            id=record.id,
            content="新的設定：使用 vitest",
            evidence=ManualRef("v2"),
            created_at=record.created_at,
            updated_at=datetime.now(),
            confidence=1.0,
            validation_status=ValidationStatus.VALID,
        )
        mem._store.save(updated)
        result = mem.query("vitest")
        assert len(result.memories) > 0
        # Old content should not match
        result_old = mem.query("舊的")
        assert len(result_old.memories) == 0
        mem.close()


class TestTFIDFChinese:
    def test_tfidf_chinese_embedding(self, tmp_path):
        mem = Memory(str(tmp_path), embedding=TFIDFEmbedding())
        mem.add("這個專案使用 pytest 做測試", evidence=ManualRef("docs"))
        mem.add("部署流程使用 Docker", evidence=ManualRef("docs"))
        mem.add("資料庫用 PostgreSQL", evidence=ManualRef("docs"))
        result = mem.query("測試框架")
        assert len(result.memories) > 0
        mem.close()

    def test_tfidf_chinese_similarity(self):
        emb = TFIDFEmbedding()
        docs = [
            "使用 pytest 做單元測試",
            "部署到 Docker 容器",
            "資料庫使用 PostgreSQL",
        ]
        emb.fit(docs)
        vecs = emb.embed_documents(docs)
        q = emb.embed_query("測試")
        # Query "測試" should be closest to doc about testing
        import numpy as np

        scores = [float(np.dot(q, v)) for v in vecs]
        assert scores[0] > scores[1]  # testing doc > deployment doc


class TestSchemaUpgrade:
    def test_v1_to_v2_migration(self, tmp_path):
        """Simulate a v1 database and verify migration works."""
        import sqlite3

        db_path = tmp_path / ".agentic-memory.db"
        conn = sqlite3.connect(str(db_path))
        # Create v1 schema (content-sync FTS5 with triggers)
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
            CREATE VIRTUAL TABLE memories_fts USING fts5(
                content, content='memories', content_rowid='rowid'
            );
            CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
            END;
        """)
        # Insert a v1 record
        conn.execute(
            "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("test-1", "使用ruff做檢查", '{"type":"manual","note":"test"}',
             "2026-01-01T00:00:00", "2026-01-01T00:00:00", 1.0, "valid", "", "[]"),
        )
        conn.execute(
            "INSERT INTO memories_fts(rowid, content) VALUES (1, '使用ruff做檢查')"
        )
        conn.commit()
        conn.close()

        # Open with new Memory class — should auto-migrate
        mem = Memory(str(tmp_path))
        # Should be able to query CJK content after migration
        result = mem.query("ruff")
        assert len(result.memories) > 0
        assert result.memories[0].content == "使用ruff做檢查"
        mem.close()
