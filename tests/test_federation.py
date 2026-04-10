"""Tests for Cross-repo Memory Federation."""

from __future__ import annotations

import sqlite3

import pytest

from agentic_memory import ManualRef, Memory
from agentic_memory.federation import FederatedMemory, FederatedQueryResult, FederatedResult


def _create_repo_with_memories(path, memories: list[tuple[str, str]]):
    """Helper: create a repo with memories and return the path."""
    repo = path
    repo.mkdir(exist_ok=True)
    mem = Memory(str(repo))
    for content, note in memories:
        mem.add(content, evidence=ManualRef(note))
    mem.close()
    return repo


@pytest.fixture()
def repo_a(tmp_path):
    return _create_repo_with_memories(
        tmp_path / "repo-a",
        [
            ("Project uses ruff for linting", "team decision"),
            ("Deploy target is Cloud Run", "infra doc"),
        ],
    )


@pytest.fixture()
def repo_b(tmp_path):
    return _create_repo_with_memories(
        tmp_path / "repo-b",
        [
            ("Database is PostgreSQL 15", "ops manual"),
            ("API uses FastAPI framework", "architecture doc"),
        ],
    )


@pytest.fixture()
def federation(tmp_path):
    fed = FederatedMemory(registry_path=tmp_path / "federation.db")
    yield fed
    fed.close()


class TestRegister:
    def test_register_repo(self, federation, repo_a):
        result = federation.register(repo_a)
        assert result.repo_path == str(repo_a)
        assert result.alias == "repo-a"

    def test_register_with_alias(self, federation, repo_a):
        result = federation.register(repo_a, alias="frontend")
        assert result.alias == "frontend"

    def test_register_nonexistent_path_raises(self, federation, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            federation.register(tmp_path / "nonexistent")

    def test_register_no_db_raises(self, federation, tmp_path):
        empty_repo = tmp_path / "empty"
        empty_repo.mkdir()
        with pytest.raises(ValueError, match="No memory database"):
            federation.register(empty_repo)

    def test_register_duplicate_raises(self, federation, repo_a):
        federation.register(repo_a)
        with pytest.raises(sqlite3.IntegrityError):
            federation.register(repo_a)

    def test_register_validates_db_schema(self, federation, tmp_path):
        """A random SQLite file should be rejected."""
        bad_repo = tmp_path / "bad-repo"
        bad_repo.mkdir()
        db_path = bad_repo / ".agentic-memory.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE random_table (x TEXT)")
        conn.close()
        with pytest.raises(ValueError, match="Not a valid"):
            federation.register(bad_repo)


class TestUnregister:
    def test_unregister_existing(self, federation, repo_a):
        federation.register(repo_a)
        assert federation.unregister(repo_a) is True

    def test_unregister_nonexistent(self, federation, tmp_path):
        assert federation.unregister(tmp_path / "nope") is False


class TestListRepos:
    def test_list_empty(self, federation):
        assert federation.list_repos() == []

    def test_list_registered(self, federation, repo_a, repo_b):
        federation.register(repo_a)
        federation.register(repo_b)
        repos = federation.list_repos()
        assert len(repos) == 2
        aliases = {r.alias for r in repos}
        assert aliases == {"repo-a", "repo-b"}


class TestFederatedQuery:
    def test_query_single_repo(self, federation, repo_a):
        federation.register(repo_a)
        result = federation.query("linting")
        assert isinstance(result, FederatedQueryResult)
        assert len(result.results) >= 1
        assert result.repos_queried == 1
        assert result.repos_failed == 0

    def test_query_result_has_attribution(self, federation, repo_a):
        federation.register(repo_a, alias="frontend")
        result = federation.query("ruff linting")
        assert len(result.results) >= 1
        hit = result.results[0]
        assert isinstance(hit, FederatedResult)
        assert hit.repo_alias == "frontend"
        assert hit.repo_path == str(repo_a)

    def test_query_across_repos(self, federation, repo_a, repo_b):
        federation.register(repo_a)
        federation.register(repo_b)
        result = federation.query("project uses")
        assert result.repos_queried == 2

    def test_query_no_repos(self, federation):
        result = federation.query("anything")
        assert result.results == []
        assert result.repos_queried == 0

    def test_query_with_missing_db_counts_as_failed(self, federation, repo_a, tmp_path):
        federation.register(repo_a)
        # Manually delete the DB file
        import os

        os.remove(repo_a / ".agentic-memory.db")
        result = federation.query("linting")
        assert result.repos_failed == 1

    def test_query_respects_limit(self, federation, repo_a, repo_b):
        federation.register(repo_a)
        federation.register(repo_b)
        result = federation.query("project", limit=1)
        assert len(result.results) <= 1


class TestContextManager:
    def test_context_manager(self, tmp_path):
        with FederatedMemory(registry_path=tmp_path / "fed.db") as fed:
            assert fed.list_repos() == []
