"""Tests for semantic embedding providers.

Tests that require sentence-transformers are skipped when the package is not installed.
Core tests (import errors, protocol conformance) always run.
"""

from __future__ import annotations

import numpy as np
import pytest

from agentic_memory import ManualRef, Memory


def _can_load_st_model() -> bool:
    """Check if sentence-transformers is installed AND model can actually load."""
    try:
        from agentic_memory.semantic import SentenceTransformerEmbedding

        SentenceTransformerEmbedding()
        return True
    except Exception:
        return False


requires_st = pytest.mark.skipif(not _can_load_st_model(), reason="sentence-transformers model not available")


class TestTFIDFStillDefault:
    def test_memory_without_embedding_works(self, tmp_path):
        """Memory() without explicit embedding still works (no regression)."""
        mem = Memory(str(tmp_path))
        mem.add("test content", evidence=ManualRef("note"))
        result = mem.query("test")
        assert len(result.memories) >= 1
        mem.close()


@requires_st
class TestSentenceTransformerEmbedding:
    def test_model_id_format(self):
        from agentic_memory.semantic import SentenceTransformerEmbedding

        emb = SentenceTransformerEmbedding()
        assert emb.model_id.startswith("st-")

    def test_dim_positive(self):
        from agentic_memory.semantic import SentenceTransformerEmbedding

        emb = SentenceTransformerEmbedding()
        assert emb.dim > 0

    def test_fit_is_noop(self):
        from agentic_memory.semantic import SentenceTransformerEmbedding

        emb = SentenceTransformerEmbedding()
        emb.fit(["hello", "world"])  # should not raise

    def test_embed_documents(self):
        from agentic_memory.semantic import SentenceTransformerEmbedding

        emb = SentenceTransformerEmbedding()
        vecs = emb.embed_documents(["hello world", "foo bar"])
        assert len(vecs) == 2
        assert all(isinstance(v, np.ndarray) for v in vecs)
        assert all(v.shape == (emb.dim,) for v in vecs)

    def test_embed_query(self):
        from agentic_memory.semantic import SentenceTransformerEmbedding

        emb = SentenceTransformerEmbedding()
        vec = emb.embed_query("hello world")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (emb.dim,)

    def test_vectors_are_normalized(self):
        from agentic_memory.semantic import SentenceTransformerEmbedding

        emb = SentenceTransformerEmbedding()
        vec = emb.embed_query("test normalization")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 0.01

    def test_dumps_loads_roundtrip(self):
        from agentic_memory.semantic import SentenceTransformerEmbedding

        emb = SentenceTransformerEmbedding()
        payload = emb.dumps()
        restored = SentenceTransformerEmbedding.loads(payload)
        assert restored.model_id == emb.model_id

    def test_semantic_similarity(self):
        """Semantically similar texts should have higher cosine similarity."""
        from agentic_memory.semantic import SentenceTransformerEmbedding

        emb = SentenceTransformerEmbedding()
        v_dog = emb.embed_query("a dog runs in the park")
        v_puppy = emb.embed_query("a puppy plays outside")
        v_code = emb.embed_query("Python SQLite database migration")

        sim_related = float(np.dot(v_dog, v_puppy))
        sim_unrelated = float(np.dot(v_dog, v_code))
        assert sim_related > sim_unrelated

    def test_with_memory_integration(self, tmp_path):
        """Full integration: Memory + SentenceTransformerEmbedding."""
        from agentic_memory.semantic import SentenceTransformerEmbedding

        emb = SentenceTransformerEmbedding()
        mem = Memory(str(tmp_path), embedding=emb)
        mem.add("Dogs are loyal pets", evidence=ManualRef("common knowledge"))
        mem.add("Python uses indentation for blocks", evidence=ManualRef("docs"))

        result = mem.query("canine companions")
        assert len(result.memories) >= 1
        # Semantic search should rank dog-related content higher
        assert "dog" in result.memories[0].content.lower() or "loyal" in result.memories[0].content.lower()
        mem.close()


class TestMultiModelMigration:
    def test_v7_migration_creates_composite_pk(self, tmp_path):
        """After migration, memory_embeddings should accept multiple models per memory."""
        from agentic_memory.store import SQLiteStore

        store = SQLiteStore(tmp_path / "test.db")
        # Insert two embeddings for same memory with different model_ids
        store._conn.execute(
            "INSERT INTO memories (id, content, evidence_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("m1", "test", '{"type":"manual","note":"t"}', "2026-01-01", "2026-01-01"),
        )
        store._conn.execute(
            "INSERT INTO memory_embeddings (memory_id, model_id, dim, vector_blob, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("m1", "tfidf-v1", 3, np.zeros(3, dtype=np.float32).tobytes(), "2026-01-01"),
        )
        store._conn.execute(
            "INSERT INTO memory_embeddings (memory_id, model_id, dim, vector_blob, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("m1", "st-all-MiniLM-L6-v2", 384, np.zeros(384, dtype=np.float32).tobytes(), "2026-01-01"),
        )
        store._conn.commit()

        # Both should be queryable
        rows = store._conn.execute("SELECT model_id FROM memory_embeddings WHERE memory_id = 'm1'").fetchall()
        assert len(rows) == 2
        model_ids = {row[0] for row in rows}
        assert "tfidf-v1" in model_ids
        assert "st-all-MiniLM-L6-v2" in model_ids
        store.close()
