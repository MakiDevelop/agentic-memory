"""Tests for content-level validation."""

from agentic_memory import (
    Memory,
    FileRef,
    ManualRef,
    KeywordOverlapValidator,
    LLMContentValidator,
)
from agentic_memory.content_validator import read_evidence_content, _extract_keywords


class TestExtractKeywords:
    def test_english(self):
        kw = _extract_keywords("Uses ruff for linting with line-length=120")
        assert "ruff" in kw
        assert "linting" in kw
        assert "120" in kw

    def test_chinese(self):
        kw = _extract_keywords("這個專案使用 pytest 做測試")
        assert "pytest" in kw

    def test_filters_stop_words(self):
        kw = _extract_keywords("the project is a good one")
        assert "the" not in kw
        assert "is" not in kw
        assert "project" in kw


class TestReadEvidenceContent:
    def test_file_ref_full(self, tmp_path):
        (tmp_path / "config.toml").write_text("[tool.ruff]\nline-length = 120\n")
        evidence = FileRef("config.toml")
        content = read_evidence_content(evidence, str(tmp_path))
        assert content is not None
        assert "ruff" in content
        assert "120" in content

    def test_file_ref_lines(self, tmp_path):
        (tmp_path / "config.toml").write_text("line1\nline2\nline3\nline4\n")
        evidence = FileRef("config.toml", lines=(2, 3))
        content = read_evidence_content(evidence, str(tmp_path))
        assert content == "line2\nline3\n"

    def test_file_not_found(self, tmp_path):
        evidence = FileRef("nonexistent.txt")
        content = read_evidence_content(evidence, str(tmp_path))
        assert content is None

    def test_manual_ref_returns_none(self, tmp_path):
        evidence = ManualRef("some note")
        content = read_evidence_content(evidence, str(tmp_path))
        assert content is None


class TestKeywordOverlapValidator:
    def test_consistent(self):
        validator = KeywordOverlapValidator()
        result = validator.check(
            "Uses ruff for linting with line-length=120",
            "[tool.ruff]\nline-length = 120\ntarget-version = 'py310'\n",
        )
        assert result.consistent
        assert result.score > 0.3

    def test_inconsistent(self):
        validator = KeywordOverlapValidator()
        result = validator.check(
            "Uses black for formatting with maxwidth=88",
            "[tool.ruff]\nline-length = 120\n",
        )
        # "black", "formatting", "maxwidth", "88" not in evidence
        assert not result.consistent
        assert result.score < 0.3

    def test_empty_memory(self):
        validator = KeywordOverlapValidator()
        result = validator.check("x", "some content here")
        assert result.consistent  # No meaningful keywords to check (single char filtered)

    def test_custom_threshold(self):
        validator = KeywordOverlapValidator(min_overlap=0.8)
        result = validator.check(
            "Uses ruff for linting",
            "[tool.ruff]\nline-length = 120\n",
        )
        # "ruff" matches but "linting" doesn't appear in evidence
        # With high threshold, should fail
        assert result.score < 0.8


class TestLLMContentValidator:
    def test_fallback_to_heuristic(self):
        validator = LLMContentValidator()
        result = validator.check(
            "Uses ruff for linting",
            "[tool.ruff]\nline-length = 120\n",
        )
        assert result.score > 0

    def test_with_llm(self):
        def mock_llm(system: str, user: str) -> str:
            return '{"consistent": true, "score": 0.9, "reason": "Memory matches source"}'

        validator = LLMContentValidator(llm_callable=mock_llm)
        result = validator.check("Uses ruff", "[tool.ruff]\n")
        assert result.consistent
        assert result.score == 0.9

    def test_llm_says_inconsistent(self):
        def mock_llm(system: str, user: str) -> str:
            return '{"consistent": false, "score": 0.2, "reason": "Memory says black but source uses ruff"}'

        validator = LLMContentValidator(llm_callable=mock_llm)
        result = validator.check("Uses black", "[tool.ruff]\n")
        assert not result.consistent
        assert result.score == 0.2

    def test_llm_failure_fallback(self):
        def broken_llm(system: str, user: str) -> str:
            raise RuntimeError("LLM down")

        validator = LLMContentValidator(llm_callable=broken_llm)
        result = validator.check("Uses ruff", "[tool.ruff]\n")
        # Should fallback to heuristic, not crash
        assert isinstance(result.score, float)


class TestMemoryWithContentValidator:
    def test_validate_detects_content_mismatch(self, tmp_path):
        (tmp_path / "config.toml").write_text("[tool.ruff]\nline-length = 120\n")
        mem = Memory(str(tmp_path), content_validator=KeywordOverlapValidator())

        # Add a memory that matches the file
        mem.add("Uses ruff with line-length=120", evidence=FileRef("config.toml", lines=(1, 2)))

        # Now change the file content WITHOUT changing the hash
        # (simulate: memory says "black" but file says "ruff")
        record = mem.add(
            "Uses black for formatting with max-line=88",
            evidence=FileRef("config.toml", lines=(1, 2)),
        )

        # Validate should detect the content mismatch
        problematic = mem.validate()
        # The "black" memory should be flagged
        flagged_ids = [r.id for r in problematic]
        assert record.id in flagged_ids
        mem.close()

    def test_validate_passes_consistent_memory(self, tmp_path):
        (tmp_path / "config.toml").write_text("[tool.ruff]\nline-length = 120\n")
        mem = Memory(str(tmp_path), content_validator=KeywordOverlapValidator())
        mem.add("Uses ruff with line-length=120", evidence=FileRef("config.toml", lines=(1, 2)))
        problematic = mem.validate()
        assert len(problematic) == 0
        mem.close()

    def test_query_with_content_validation(self, tmp_path):
        (tmp_path / "config.toml").write_text("[tool.ruff]\nline-length = 120\n")
        mem = Memory(str(tmp_path), content_validator=KeywordOverlapValidator())
        mem.add(
            "Uses black for formatting with max-line=88",
            evidence=FileRef("config.toml", lines=(1, 2)),
        )
        result = mem.query("formatting")
        # Should be flagged as stale due to content mismatch
        if result.memories:
            assert result.memories[0].validation_status.value == "stale"
        mem.close()

    def test_no_validator_skips_content_check(self, tmp_path):
        (tmp_path / "config.toml").write_text("[tool.ruff]\nline-length = 120\n")
        mem = Memory(str(tmp_path))  # No content_validator
        mem.add(
            "Uses black for formatting",
            evidence=FileRef("config.toml", lines=(1, 2)),
        )
        # Without content_validator, this should pass validation (file exists + hash matches)
        problematic = mem.validate()
        assert len(problematic) == 0
        mem.close()

    def test_manual_ref_skips_content_check(self, tmp_path):
        mem = Memory(str(tmp_path), content_validator=KeywordOverlapValidator())
        mem.add("Some random claim", evidence=ManualRef("I said so"))
        problematic = mem.validate()
        assert len(problematic) == 0
        mem.close()
