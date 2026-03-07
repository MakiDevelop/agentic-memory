"""Tests for embedding providers and hybrid search."""

import numpy as np
import pytest

from agentic_memory import ManualRef, Memory, TFIDFEmbedding


class TestTFIDFEmbedding:
    def test_fit_and_embed(self):
        emb = TFIDFEmbedding()
        texts = [
            "This project uses ruff for linting",
            "Tests are written with pytest",
            "Deploy using docker compose",
        ]
        emb.fit(texts)
        assert emb.dim > 0

        vecs = emb.embed_documents(texts)
        assert len(vecs) == 3
        for v in vecs:
            assert v.shape == (emb.dim,)
            # Should be L2 normalized
            norm = np.linalg.norm(v)
            assert abs(norm - 1.0) < 0.01 or norm == 0.0

    def test_embed_query(self):
        emb = TFIDFEmbedding()
        emb.fit(["ruff linting tool", "pytest testing framework"])
        q = emb.embed_query("linting")
        assert q.shape == (emb.dim,)

    def test_similar_queries_have_high_cosine(self):
        emb = TFIDFEmbedding()
        emb.fit([
            "ruff is used for linting python code",
            "pytest is the testing framework",
            "docker compose for deployment",
        ])
        v1 = emb.embed_query("linting python")
        v2 = emb.embed_query("ruff linting")
        v3 = emb.embed_query("docker deployment")

        sim_12 = float(np.dot(v1, v2))
        sim_13 = float(np.dot(v1, v3))
        # linting queries should be more similar to each other than to docker
        assert sim_12 > sim_13

    def test_not_fitted_raises(self):
        emb = TFIDFEmbedding()
        with pytest.raises(ValueError, match="not fitted"):
            emb.embed_query("test")

    def test_serialization_roundtrip(self):
        emb = TFIDFEmbedding()
        emb.fit(["hello world", "foo bar baz"])

        data = emb.dumps()
        restored = TFIDFEmbedding.loads(data)

        assert restored.model_id == emb.model_id
        assert restored.dim == emb.dim

        # Vectors should be identical after roundtrip
        v1 = emb.embed_query("hello")
        v2 = restored.embed_query("hello")
        np.testing.assert_array_almost_equal(v1, v2)

    def test_empty_query_returns_zero_vector(self):
        emb = TFIDFEmbedding()
        emb.fit(["some text here"])
        v = emb.embed_query("xyznonexistent")
        assert np.allclose(v, 0.0)


class TestHybridSearch:
    def test_query_with_embedding(self, tmp_path):
        emb = TFIDFEmbedding()
        with Memory(tmp_path, embedding=emb) as mem:
            mem.add("Uses ruff for linting with line-length=120", evidence=ManualRef("pyproject.toml"))
            mem.add("Tests use pytest with coverage", evidence=ManualRef("README"))
            mem.add("Deploy with docker compose up", evidence=ManualRef("docs"))

            result = mem.query("linting tool")
            assert len(result.memories) > 0
            assert "ruff" in result.answer

    def test_vector_finds_semantic_match(self, tmp_path):
        emb = TFIDFEmbedding()
        with Memory(tmp_path, embedding=emb) as mem:
            mem.add("Uses ruff for python code formatting", evidence=ManualRef("config"))
            mem.add("Database uses PostgreSQL 15", evidence=ManualRef("infra"))
            mem.add("CI pipeline runs on GitHub Actions", evidence=ManualRef("ci"))

            # "formatting" should match "ruff" memory even via vector
            result = mem.query("code formatting")
            assert len(result.memories) > 0

    def test_query_without_embedding_still_works(self, tmp_path):
        """FTS-only search should work when no embedding provider is set."""
        with Memory(tmp_path) as mem:
            mem.add("Uses ruff for linting", evidence=ManualRef("note"))
            result = mem.query("ruff")
            assert len(result.memories) > 0

    def test_custom_weights(self, tmp_path):
        emb = TFIDFEmbedding()
        with Memory(tmp_path, embedding=emb) as mem:
            mem.add("ruff linting configuration", evidence=ManualRef("a"))
            mem.add("pytest testing setup", evidence=ManualRef("b"))

            # Should work with different weight configurations
            r1 = mem.query("linting", fts_weight=1.0, vector_weight=0.0)
            r2 = mem.query("linting", fts_weight=0.0, vector_weight=1.0)
            assert len(r1.memories) > 0
            assert len(r2.memories) > 0

    def test_embedding_persists_across_instances(self, tmp_path):
        emb = TFIDFEmbedding()
        with Memory(tmp_path, embedding=emb) as mem:
            mem.add("ruff linting tool", evidence=ManualRef("a"))
            mem.add("pytest framework", evidence=ManualRef("b"))

        # New instance should restore embedding state
        emb2 = TFIDFEmbedding()
        with Memory(tmp_path, embedding=emb2) as mem2:
            result = mem2.query("linting", fts_weight=0.0, vector_weight=1.0)
            assert len(result.memories) > 0
