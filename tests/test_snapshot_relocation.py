"""Tests for P0 (content snapshot + fuzzy relocation), P1 (validate_detail), P2 (multi-evidence)."""

import pytest

from agentic_memory import FileRef, ManualRef, Memory, ValidationResult, ValidationStatus
from agentic_memory.evidence import _find_snippet_in_file, _read_lines


# --- P0: Content Snapshot + Fuzzy Relocation ---


class TestContentSnapshot:
    def test_capture_snapshot(self, tmp_path):
        """FileRef.capture_hash should also capture content_snapshot."""
        f = tmp_path / "config.py"
        f.write_text("APP_NAME = 'test'\nDEBUG = True\nPORT = 8080\n")
        ref = FileRef("config.py", lines=(2, 3))
        ref.capture_hash(str(tmp_path))
        assert ref.content_snapshot == "DEBUG = True\nPORT = 8080\n"
        assert ref.content_hash != ""

    def test_snapshot_serialization_roundtrip(self, tmp_path):
        """content_snapshot survives to_dict/from_dict."""
        f = tmp_path / "test.py"
        f.write_text("line1\nline2\nline3\n")
        ref = FileRef("test.py", lines=(1, 2))
        ref.capture_hash(str(tmp_path))
        data = ref.to_dict()
        restored = FileRef.from_dict(data)
        assert restored.content_snapshot == ref.content_snapshot
        assert restored.content_hash == ref.content_hash

    def test_legacy_no_snapshot(self, tmp_path):
        """Old records without snapshot still work (no relocation attempted)."""
        f = tmp_path / "config.py"
        f.write_text("DEBUG = True\n")
        ref = FileRef("config.py", lines=(1, 1), content_hash="old_hash")
        # No content_snapshot — should fall back to STALE
        status, msg = ref.validate(str(tmp_path))
        assert status == ValidationStatus.STALE


class TestFuzzyRelocation:
    def test_relocate_on_line_shift(self, tmp_path):
        """Insert a line at the top — content should be relocated, not marked stale."""
        f = tmp_path / "config.py"
        f.write_text("DEBUG = True\nPORT = 8080\nHOST = 'localhost'\n")
        ref = FileRef("config.py", lines=(1, 2))
        ref.capture_hash(str(tmp_path))
        assert ref.content_snapshot == "DEBUG = True\nPORT = 8080\n"

        # Insert a line at the top
        f.write_text("# New comment added\nDEBUG = True\nPORT = 8080\nHOST = 'localhost'\n")

        status, msg = ref.validate(str(tmp_path))
        assert status == ValidationStatus.VALID
        assert "Relocated" in msg or "relocated" in msg.lower()
        assert ref.lines == (2, 3)  # shifted down by 1

    def test_relocate_multiple_insertions(self, tmp_path):
        """Multiple lines inserted before the referenced range."""
        f = tmp_path / "config.py"
        f.write_text("AAA = 'alpha'\nBBB = 'bravo'\nCCC = 'charlie'\nDATABASE_HOST = 'localhost'\nDATABASE_PORT = 5432\n")
        ref = FileRef("config.py", lines=(4, 5))
        ref.capture_hash(str(tmp_path))

        # Insert 3 lines before
        f.write_text("X = 0\nY = 0\nZ = 0\nAAA = 'alpha'\nBBB = 'bravo'\nCCC = 'charlie'\nDATABASE_HOST = 'localhost'\nDATABASE_PORT = 5432\n")

        status, msg = ref.validate(str(tmp_path))
        assert status == ValidationStatus.VALID
        assert ref.lines == (7, 8)

    def test_content_actually_changed(self, tmp_path):
        """Content really changed — should still be STALE."""
        f = tmp_path / "config.py"
        f.write_text("DEBUG = True\nPORT = 8080\n")
        ref = FileRef("config.py", lines=(1, 2))
        ref.capture_hash(str(tmp_path))

        # Actually change the content
        f.write_text("DEBUG = False\nPORT = 3000\n")

        status, msg = ref.validate(str(tmp_path))
        assert status == ValidationStatus.STALE

    def test_short_snippet_no_relocation(self, tmp_path):
        """Very short content (<20 chars) should not attempt fuzzy relocation."""
        f = tmp_path / "config.py"
        f.write_text("A = 1\n")
        ref = FileRef("config.py", lines=(1, 1))
        ref.capture_hash(str(tmp_path))

        f.write_text("B = 2\nA = 1\n")

        # Snapshot is "A = 1\n" — only 6 chars, below threshold
        status, msg = ref.validate(str(tmp_path))
        assert status == ValidationStatus.STALE

    def test_momo_scenario_4_refs_1_insertion(self, tmp_path):
        """MOMO real scenario: 4 refs to config.py, insert 1 line at top."""
        f = tmp_path / "config.py"
        lines = [f"CONFIG_{i} = {i}\n" for i in range(80)]
        f.write_text("".join(lines))

        refs = [
            FileRef("config.py", lines=(11, 19)),
            FileRef("config.py", lines=(29, 38)),
            FileRef("config.py", lines=(52, 57)),
            FileRef("config.py", lines=(68, 78)),
        ]
        for ref in refs:
            ref.capture_hash(str(tmp_path))

        # Insert 1 line at the beginning
        f.write_text("APP_NAME = 'MOMO'\n" + "".join(lines))

        # All 4 should relocate, not stale
        for i, ref in enumerate(refs):
            status, msg = ref.validate(str(tmp_path))
            assert status == ValidationStatus.VALID, f"ref[{i}] was {status}: {msg}"


class TestHelpers:
    def test_read_lines(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")
        assert _read_lines(str(f), 2, 3) == "line2\nline3\n"

    def test_read_lines_full_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\n")
        assert _read_lines(str(f)) == "line1\nline2\n"

    def test_find_snippet(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("aaa\nbbb\nccc\nddd target content here\neee\nfff\n")
        result = _find_snippet_in_file(str(f), "ddd target content here\neee\n")
        assert result == (4, 5)

    def test_find_snippet_not_found(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("aaa\nbbb\nccc\n")
        result = _find_snippet_in_file(str(f), "this text does not exist anywhere in file")
        assert result is None

    def test_find_snippet_too_short(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("short\n")
        result = _find_snippet_in_file(str(f), "short")
        assert result is None  # < 20 chars, skip


# --- P1: validate_detail ---


class TestValidateDetail:
    def test_valid_returns_no_diff(self, tmp_path):
        f = tmp_path / "config.py"
        f.write_text("DEBUG = True\nPORT = 8080\n")
        ref = FileRef("config.py", lines=(1, 2))
        ref.capture_hash(str(tmp_path))

        result = ref.validate_detail(str(tmp_path))
        assert isinstance(result, ValidationResult)
        assert result.status == ValidationStatus.VALID
        assert result.old_content is None
        assert result.new_content is None

    def test_stale_returns_diff(self, tmp_path):
        f = tmp_path / "config.py"
        f.write_text("DEBUG = True\nPORT = 8080\n")
        ref = FileRef("config.py", lines=(1, 2))
        ref.capture_hash(str(tmp_path))

        f.write_text("DEBUG = False\nPORT = 3000\n")

        result = ref.validate_detail(str(tmp_path))
        assert result.status == ValidationStatus.STALE
        assert result.old_content == "DEBUG = True\nPORT = 8080\n"
        assert result.new_content == "DEBUG = False\nPORT = 3000\n"

    def test_manual_ref_validate_detail(self, tmp_path):
        ref = ManualRef("some note")
        result = ref.validate_detail(str(tmp_path))
        assert result.status == ValidationStatus.VALID
        assert result.old_content is None

    def test_as_tuple(self):
        result = ValidationResult(
            status=ValidationStatus.STALE, message="changed"
        )
        t = result.as_tuple()
        assert t == (ValidationStatus.STALE, "changed")


# --- P2: Multi-Evidence ---


class TestMultiEvidence:
    def test_add_multi_evidence(self, tmp_path):
        f = tmp_path / "config.py"
        f.write_text("DEBUG = True\n")
        mem = Memory(str(tmp_path))
        record = mem.add(
            "Config and manual note",
            evidence=[FileRef("config.py", lines=(1, 1)), ManualRef("team decision")],
        )
        assert isinstance(record.evidence, list)
        assert len(record.evidence) == 2
        mem.close()

    def test_multi_evidence_roundtrip(self, tmp_path):
        f = tmp_path / "config.py"
        f.write_text("DEBUG = True\n")
        mem = Memory(str(tmp_path))
        record = mem.add(
            "Multi-ref test",
            evidence=[FileRef("config.py"), ManualRef("note")],
        )
        # Retrieve from DB
        retrieved = mem.get(record.id)
        assert isinstance(retrieved.evidence, list)
        assert len(retrieved.evidence) == 2
        mem.close()

    def test_multi_evidence_worst_wins(self, tmp_path):
        f = tmp_path / "config.py"
        f.write_text("DEBUG = True\n")
        mem = Memory(str(tmp_path))
        record = mem.add(
            "File + missing file",
            evidence=[FileRef("config.py"), FileRef("nonexistent.py")],
        )
        # nonexistent.py is INVALID → worst wins
        assert record.validation_status == ValidationStatus.INVALID
        mem.close()

    def test_single_evidence_unchanged(self, tmp_path):
        """Existing single-evidence API still works identically."""
        mem = Memory(str(tmp_path))
        record = mem.add("test", evidence=ManualRef("note"))
        assert not isinstance(record.evidence, list)
        assert record.evidence_list == [record.evidence]
        mem.close()

    def test_evidence_list_property(self, tmp_path):
        mem = Memory(str(tmp_path))
        single = mem.add("single", evidence=ManualRef("a"))
        assert len(single.evidence_list) == 1

        f = tmp_path / "f.txt"
        f.write_text("content\n")
        multi = mem.add("multi", evidence=[FileRef("f.txt"), ManualRef("b")])
        assert len(multi.evidence_list) == 2
        mem.close()

    def test_evidence_label(self, tmp_path):
        mem = Memory(str(tmp_path))
        f = tmp_path / "f.txt"
        f.write_text("content\n")
        record = mem.add("test", evidence=[FileRef("f.txt", lines=(1, 1)), ManualRef("note")])
        label = record.evidence_label
        assert "f.txt" in label
        assert "manual" in label
        assert "|" in label
        mem.close()

    def test_multi_evidence_validate_stale(self, tmp_path):
        """If one evidence becomes stale, the whole memory is stale."""
        f = tmp_path / "config.py"
        f.write_text("DEBUG = True\nPORT = 8080\n")
        mem = Memory(str(tmp_path))
        mem.add(
            "Config settings",
            evidence=[FileRef("config.py", lines=(1, 2)), ManualRef("docs")],
        )
        # Change the file
        f.write_text("DEBUG = False\nPORT = 3000\n")
        problematic = mem.validate()
        assert len(problematic) == 1
        assert problematic[0].validation_status == ValidationStatus.STALE
        mem.close()

    def test_legacy_single_evidence_db(self, tmp_path):
        """Old DB with single evidence JSON should still load."""
        import sqlite3
        import json

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
            INSERT INTO schema_meta VALUES ('version', '2');
        """)
        evidence = json.dumps({"type": "manual", "note": "old record"})
        conn.execute(
            "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("old-1", "legacy memory", evidence,
             "2026-01-01T00:00:00", "2026-01-01T00:00:00", 1.0, "valid", "", "[]"),
        )
        conn.commit()
        conn.close()

        mem = Memory(str(tmp_path))
        record = mem.get("old-1")
        assert record is not None
        assert record.content == "legacy memory"
        assert not isinstance(record.evidence, list)
        mem.close()
