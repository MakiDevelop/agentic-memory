"""Tests for memory lifecycle automation."""

from __future__ import annotations

import os
import subprocess
import time

import pytest

from agentic_memory import (
    FileRef,
    LifecycleManager,
    LifecycleResult,
    ManualRef,
    Memory,
    install_precommit_hook,
    is_installed,
    uninstall_precommit_hook,
)


@pytest.fixture()
def mem(tmp_path):
    m = Memory(str(tmp_path))
    yield m
    m.close()


class TestAutoExpire:
    def test_removes_ttl_expired(self, mem):
        mem.add("transient data", evidence=ManualRef("ephemeral"), ttl_seconds=1)
        time.sleep(1.1)
        removed = mem.auto_expire()
        assert removed == 1
        assert mem._store.count() == 0

    def test_keeps_non_expired(self, mem):
        mem.add("permanent data", evidence=ManualRef("stable"))
        removed = mem.auto_expire()
        assert removed == 0
        assert mem._store.count() == 1

    def test_mixed(self, mem):
        mem.add("kept", evidence=ManualRef("k"))
        mem.add("expired", evidence=ManualRef("e"), ttl_seconds=1)
        time.sleep(1.1)
        removed = mem.auto_expire()
        assert removed == 1
        assert mem._store.count() == 1


class TestAutoDowngradeStale:
    def test_downgrades_stale_file_memory(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        target = repo / "config.py"
        target.write_text("ORIGINAL_VALUE = 1\nANOTHER_LINE = 2\nYET_MORE_CONTENT = 3\n")

        mem = Memory(str(repo))
        r = mem.add(
            "config has ORIGINAL_VALUE",
            evidence=FileRef("config.py", lines=(1, 1)),
            importance=2,
        )

        # Modify file so hash no longer matches
        target.write_text("CHANGED_VALUE = 99\nDIFFERENT_CONTENT = 42\nUNRELATED_STUFF = 7\n")

        downgraded = mem.auto_downgrade_stale()
        assert downgraded == 1

        # Verify importance dropped and auto_downgraded flag set
        row = mem._store._conn.execute(
            "SELECT importance, auto_downgraded FROM memories WHERE id = ?", (r.id,)
        ).fetchone()
        assert row[0] == 1  # 2 - 1
        assert row[1] == 1
        mem.close()

    def test_skips_valid_memories(self, mem):
        mem.add("valid memory", evidence=ManualRef("trusted"), importance=2)
        downgraded = mem.auto_downgrade_stale()
        assert downgraded == 0

    def test_skips_already_downgraded(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        target = repo / "f.py"
        target.write_text("ORIGINAL_CONTENT_HERE = 1\nSECOND_LINE_FOR_SAFETY = 2\n")

        mem = Memory(str(repo))
        mem.add(
            "reference to original",
            evidence=FileRef("f.py", lines=(1, 1)),
            importance=2,
        )
        target.write_text("TOTALLY_DIFFERENT_STUFF = 99\nMORE_CHANGES_HERE = 42\n")

        first_pass = mem.auto_downgrade_stale()
        assert first_pass == 1

        # Running again should be a no-op
        second_pass = mem.auto_downgrade_stale()
        assert second_pass == 0
        mem.close()

    def test_importance_floor_at_zero(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        target = repo / "f.py"
        target.write_text("ORIGINAL_IMPORTANT_CONTENT = 1\nSECOND_IMPORTANT_LINE = 2\n")

        mem = Memory(str(repo))
        r = mem.add(
            "low importance ref",
            evidence=FileRef("f.py", lines=(1, 1)),
            importance=0,  # already at floor
        )
        target.write_text("CHANGED_CONTENT_COMPLETELY = 99\nTOTALLY_NEW_LINE = 42\n")
        mem.auto_downgrade_stale(importance_penalty=1)

        row = mem._store._conn.execute(
            "SELECT importance FROM memories WHERE id = ?", (r.id,)
        ).fetchone()
        assert row[0] == 0  # doesn't go negative
        mem.close()


class TestAutoCompactByAdoption:
    def test_skips_critical_memories(self, mem):
        r = mem.add("critical rule", evidence=ManualRef("law"), importance=3)
        # Force created_at to 60 days ago via direct DB update
        mem._store._conn.execute(
            "UPDATE memories SET created_at = datetime('now', '-60 days') WHERE id = ?",
            (r.id,),
        )
        mem._store._conn.commit()
        removed = mem.auto_compact_by_adoption(min_adoption_count=0, min_age_days=30)
        assert removed == 0

    def test_removes_old_unused(self, mem):
        r = mem.add("unused memory", evidence=ManualRef("note"), importance=1)
        mem._store._conn.execute(
            "UPDATE memories SET created_at = datetime('now', '-60 days') WHERE id = ?",
            (r.id,),
        )
        mem._store._conn.commit()
        removed = mem.auto_compact_by_adoption(min_adoption_count=0, min_age_days=30)
        assert removed == 1

    def test_keeps_recently_created(self, mem):
        mem.add("new memory", evidence=ManualRef("note"), importance=1)
        removed = mem.auto_compact_by_adoption(min_adoption_count=0, min_age_days=30)
        assert removed == 0

    def test_keeps_adopted_memories(self, mem):
        r = mem.add("used memory", evidence=ManualRef("note"), importance=1)
        mem._store._conn.execute(
            "UPDATE memories SET created_at = datetime('now', '-60 days') WHERE id = ?",
            (r.id,),
        )
        mem._store._conn.commit()
        mem.mark_adopted(r.id, agent_name="claude")
        mem.mark_adopted(r.id, agent_name="codex")
        removed = mem.auto_compact_by_adoption(min_adoption_count=1, min_age_days=30)
        assert removed == 0


class TestRunLifecycle:
    def test_full_pipeline(self, mem):
        mem.add("kept", evidence=ManualRef("k"), importance=3)
        mem.add("expired", evidence=ManualRef("e"), ttl_seconds=1)
        time.sleep(1.1)
        result = mem.run_lifecycle()
        assert isinstance(result, LifecycleResult)
        assert result.expired_removed == 1
        assert result.total_before == 2
        assert result.total_after == 1


class TestLifecycleManager:
    def test_manager_wraps_memory(self, mem):
        manager = LifecycleManager(mem)
        assert manager._memory is mem


class TestGitHookInstall:
    def test_install_in_non_git_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Not a git repository"):
            install_precommit_hook(tmp_path)

    def test_install_creates_hook(self, tmp_path):
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        hook_path = install_precommit_hook(tmp_path)
        assert hook_path.exists()
        assert os.access(hook_path, os.X_OK)
        content = hook_path.read_text()
        assert "agentic-memory" in content
        assert "am validate" in content

    def test_is_installed_detects(self, tmp_path):
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        assert is_installed(tmp_path) is False
        install_precommit_hook(tmp_path)
        assert is_installed(tmp_path) is True

    def test_install_refreshes_existing(self, tmp_path):
        """Installing over an existing agentic-memory hook should succeed."""
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        install_precommit_hook(tmp_path)
        # Second install should not raise
        install_precommit_hook(tmp_path)
        assert is_installed(tmp_path)

    def test_install_refuses_to_overwrite_other_hook(self, tmp_path):
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        hook_path = tmp_path / ".git" / "hooks" / "pre-commit"
        hook_path.parent.mkdir(parents=True, exist_ok=True)
        hook_path.write_text("#!/bin/sh\necho 'my custom hook'\n")

        with pytest.raises(FileExistsError):
            install_precommit_hook(tmp_path)

    def test_install_force_overwrites(self, tmp_path):
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        hook_path = tmp_path / ".git" / "hooks" / "pre-commit"
        hook_path.parent.mkdir(parents=True, exist_ok=True)
        hook_path.write_text("#!/bin/sh\necho 'old'\n")

        install_precommit_hook(tmp_path, force=True)
        assert is_installed(tmp_path)

    def test_uninstall_removes_our_hook(self, tmp_path):
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        install_precommit_hook(tmp_path)
        assert uninstall_precommit_hook(tmp_path) is True
        assert is_installed(tmp_path) is False

    def test_uninstall_leaves_foreign_hook(self, tmp_path):
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        hook_path = tmp_path / ".git" / "hooks" / "pre-commit"
        hook_path.parent.mkdir(parents=True, exist_ok=True)
        hook_path.write_text("#!/bin/sh\necho 'mine'\n")
        assert uninstall_precommit_hook(tmp_path) is False
        assert hook_path.exists()

    def test_uninstall_nonexistent_returns_false(self, tmp_path):
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        assert uninstall_precommit_hook(tmp_path) is False
