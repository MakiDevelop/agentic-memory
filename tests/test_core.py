"""Core tests for agentic-memory."""

import os
import tempfile

import pytest

from agentic_memory import FileRef, GitCommitRef, ManualRef, Memory, ValidationStatus


@pytest.fixture
def repo(tmp_path):
    """Create a temporary repo with some files."""
    # Create a sample file
    sample = tmp_path / "config.toml"
    sample.write_text("[tool.ruff]\nline-length = 120\ntarget = 'py310'\n")

    # Create a nested file
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("# Main module\nimport os\nprint('hello')\n")

    return tmp_path


@pytest.fixture
def mem(repo):
    """Create a Memory instance for the test repo."""
    m = Memory(repo)
    yield m
    m.close()


class TestAdd:
    def test_add_with_file_evidence(self, mem, repo):
        record = mem.add(
            "Uses ruff with line-length=120",
            evidence=FileRef("config.toml", lines=(1, 2)),
        )
        assert record.content == "Uses ruff with line-length=120"
        assert record.confidence == 1.0
        assert record.validation_status == ValidationStatus.VALID

    def test_add_without_evidence_raises(self, mem):
        with pytest.raises(TypeError, match="evidence must be an Evidence instance"):
            mem.add("some memory", evidence="not an evidence")

    def test_add_with_manual_evidence(self, mem):
        record = mem.add(
            "Team prefers short functions",
            evidence=ManualRef("discussed in standup"),
        )
        assert record.validation_status == ValidationStatus.VALID

    def test_add_with_missing_file(self, mem):
        record = mem.add(
            "Config in missing file",
            evidence=FileRef("nonexistent.toml"),
        )
        assert record.validation_status == ValidationStatus.INVALID

    def test_add_with_tags(self, mem):
        record = mem.add(
            "Uses pytest",
            evidence=ManualRef("from README"),
            tags=["testing", "tooling"],
        )
        assert record.tags == ["testing", "tooling"]


class TestQuery:
    def test_query_finds_memory(self, mem):
        mem.add("Uses ruff for linting", evidence=ManualRef("README"))
        result = mem.query("ruff linting")
        assert len(result.memories) > 0
        assert "ruff" in result.answer

    def test_query_no_results(self, mem):
        result = mem.query("nonexistent topic xyz")
        assert len(result.memories) == 0
        assert result.answer == ""

    def test_query_validates_by_default(self, mem, repo):
        mem.add(
            "Config uses line-length=120",
            evidence=FileRef("config.toml", lines=(1, 2)),
        )
        # Modify the file to make evidence stale
        (repo / "config.toml").write_text("[tool.black]\nline-length = 88\n")

        result = mem.query("line-length")
        assert len(result.citations) > 0
        assert result.citations[0].status == ValidationStatus.STALE

    def test_query_skip_validation(self, mem):
        mem.add("test memory", evidence=ManualRef("note"))
        result = mem.query("test", validate=False)
        assert len(result.memories) > 0

    def test_query_exclude_stale(self, mem, repo):
        mem.add("old config", evidence=FileRef("config.toml", lines=(1, 2)))
        (repo / "config.toml").write_text("completely different content\n")

        result = mem.query("config", include_stale=False)
        assert len(result.memories) == 0


class TestValidate:
    def test_validate_all_valid(self, mem):
        mem.add("fact 1", evidence=ManualRef("source 1"))
        mem.add("fact 2", evidence=ManualRef("source 2"))
        problematic = mem.validate()
        assert len(problematic) == 0

    def test_validate_detects_stale(self, mem, repo):
        mem.add("config rule", evidence=FileRef("config.toml", lines=(1, 2)))
        (repo / "config.toml").write_text("changed content\n")
        problematic = mem.validate()
        assert len(problematic) == 1
        assert problematic[0].validation_status == ValidationStatus.STALE

    def test_validate_detects_invalid(self, mem, repo):
        mem.add("source code rule", evidence=FileRef("src/main.py"))
        os.remove(repo / "src" / "main.py")
        problematic = mem.validate()
        assert len(problematic) == 1
        assert problematic[0].validation_status == ValidationStatus.INVALID


class TestStatus:
    def test_status_empty(self, mem):
        s = mem.status()
        assert s["total"] == 0

    def test_status_counts(self, mem, repo):
        mem.add("valid memory", evidence=ManualRef("note"))
        mem.add("file memory", evidence=FileRef("config.toml"))
        s = mem.status()
        assert s["total"] == 2


class TestEvidence:
    def test_file_ref_short_label(self):
        ref = FileRef("src/main.py", lines=(10, 20))
        assert ref.short_label() == "src/main.py L10-20"

    def test_file_ref_no_lines(self):
        ref = FileRef("README.md")
        assert ref.short_label() == "README.md"

    def test_manual_ref_short_label(self):
        ref = ManualRef("team decision from Monday standup")
        assert "team decision" in ref.short_label()

    def test_git_commit_ref_short_label(self):
        ref = GitCommitRef(sha="abc123def456", file_path="src/main.py")
        assert "abc123de" in ref.short_label()

    def test_file_ref_serialization(self):
        ref = FileRef("config.toml", lines=(1, 10), content_hash="abc123")
        data = ref.to_dict()
        restored = FileRef.from_dict(data)
        assert restored.path == "config.toml"
        assert restored.lines == (1, 10)
        assert restored.content_hash == "abc123"

    def test_manual_ref_serialization(self):
        ref = ManualRef("standup note")
        data = ref.to_dict()
        restored = ManualRef.from_dict(data)
        assert restored.note == "standup note"


class TestMemoryLifecycle:
    def test_get_by_id(self, mem):
        record = mem.add("test", evidence=ManualRef("note"))
        retrieved = mem.get(record.id)
        assert retrieved is not None
        assert retrieved.content == "test"

    def test_delete(self, mem):
        record = mem.add("to delete", evidence=ManualRef("note"))
        assert mem.delete(record.id)
        assert mem.get(record.id) is None

    def test_list_all(self, mem):
        mem.add("first", evidence=ManualRef("a"))
        mem.add("second", evidence=ManualRef("b"))
        records = mem.list_all()
        assert len(records) == 2

    def test_context_manager(self, repo):
        with Memory(repo) as mem:
            mem.add("test", evidence=ManualRef("note"))
            assert mem.status()["total"] == 1
