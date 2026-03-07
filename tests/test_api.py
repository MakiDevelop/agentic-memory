"""Tests for REST API server."""

import pytest
from fastapi.testclient import TestClient

from agentic_memory import api_server


@pytest.fixture(autouse=True)
def setup_repo(tmp_path, monkeypatch):
    """Set up a temp repo and reset the global memory for each test."""
    monkeypatch.setenv("AGENTIC_MEMORY_REPO", str(tmp_path))
    api_server._memory = None

    # Create a sample file for FileRef tests
    (tmp_path / "config.toml").write_text("[tool.ruff]\nline-length = 120\n")

    yield


@pytest.fixture
def client():
    return TestClient(api_server.app)


class TestAddEndpoint:
    def test_add_with_manual_evidence(self, client):
        resp = client.post("/memories", json={
            "content": "Uses ruff for linting",
            "evidence": {"type": "manual", "note": "from README"},
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["content"] == "Uses ruff for linting"
        assert data["id"]
        assert data["validation_status"] == "valid"

    def test_add_with_file_evidence(self, client):
        resp = client.post("/memories", json={
            "content": "Line length is 120",
            "evidence": {"type": "file", "file_path": "config.toml", "lines_start": 1, "lines_end": 2},
        })
        assert resp.status_code == 201
        assert "config.toml" in resp.json()["evidence_label"]

    def test_add_with_tags(self, client):
        resp = client.post("/memories", json={
            "content": "Uses pytest",
            "evidence": {"type": "manual", "note": "docs"},
            "tags": ["testing", "tooling"],
        })
        assert resp.status_code == 201
        assert resp.json()["tags"] == ["testing", "tooling"]

    def test_add_missing_file_path(self, client):
        resp = client.post("/memories", json={
            "content": "test",
            "evidence": {"type": "file"},
        })
        assert resp.status_code == 400

    def test_add_unknown_evidence_type(self, client):
        resp = client.post("/memories", json={
            "content": "test",
            "evidence": {"type": "unknown"},
        })
        assert resp.status_code == 400


class TestQueryEndpoint:
    def test_query_finds_memory(self, client):
        client.post("/memories", json={
            "content": "Uses ruff for linting",
            "evidence": {"type": "manual", "note": "README"},
        })
        resp = client.post("/memories/query", json={"query": "ruff linting"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["memories"]) > 0
        assert "ruff" in data["answer"]

    def test_query_no_results(self, client):
        resp = client.post("/memories/query", json={"query": "nonexistent xyz"})
        assert resp.status_code == 200
        assert resp.json()["memories"] == []

    def test_query_with_limit(self, client):
        for i in range(5):
            client.post("/memories", json={
                "content": f"Memory number {i} about testing",
                "evidence": {"type": "manual", "note": f"source {i}"},
            })
        resp = client.post("/memories/query", json={"query": "testing", "limit": 2})
        assert len(resp.json()["memories"]) <= 2


class TestListEndpoint:
    def test_list_empty(self, client):
        resp = client.get("/memories")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_with_memories(self, client):
        client.post("/memories", json={
            "content": "first",
            "evidence": {"type": "manual", "note": "a"},
        })
        client.post("/memories", json={
            "content": "second",
            "evidence": {"type": "manual", "note": "b"},
        })
        resp = client.get("/memories")
        assert len(resp.json()) == 2


class TestGetEndpoint:
    def test_get_existing(self, client):
        add_resp = client.post("/memories", json={
            "content": "test memory",
            "evidence": {"type": "manual", "note": "note"},
        })
        memory_id = add_resp.json()["id"]
        resp = client.get(f"/memories/{memory_id}")
        assert resp.status_code == 200
        assert resp.json()["content"] == "test memory"

    def test_get_not_found(self, client):
        resp = client.get("/memories/nonexistent")
        assert resp.status_code == 404


class TestDeleteEndpoint:
    def test_delete_existing(self, client):
        add_resp = client.post("/memories", json={
            "content": "to delete",
            "evidence": {"type": "manual", "note": "note"},
        })
        memory_id = add_resp.json()["id"]
        resp = client.delete(f"/memories/{memory_id}")
        assert resp.status_code == 200

        # Should be gone
        resp2 = client.get(f"/memories/{memory_id}")
        assert resp2.status_code == 404

    def test_delete_not_found(self, client):
        resp = client.delete("/memories/nonexistent")
        assert resp.status_code == 404


class TestValidateEndpoint:
    def test_validate_all_valid(self, client):
        client.post("/memories", json={
            "content": "valid memory",
            "evidence": {"type": "manual", "note": "note"},
        })
        resp = client.post("/memories/validate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"]["valid"] >= 1
        assert data["problematic"] == []


class TestStatusEndpoint:
    def test_status_empty(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_status_with_memories(self, client):
        client.post("/memories", json={
            "content": "test",
            "evidence": {"type": "manual", "note": "n"},
        })
        resp = client.get("/status")
        assert resp.json()["total"] == 1
