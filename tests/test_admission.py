"""Tests for admission control."""

import json

import pytest

from agentic_memory import (
    AlwaysAdmit,
    HeuristicAdmissionController,
    LLMAdmissionController,
    ManualRef,
    Memory,
)


class TestAlwaysAdmit:
    def test_admits_everything(self):
        ctrl = AlwaysAdmit()
        result = ctrl.check("anything at all")
        assert result.admitted is True
        assert result.score == 1.0

    def test_admits_empty_string(self):
        ctrl = AlwaysAdmit()
        result = ctrl.check("")
        assert result.admitted is True


class TestHeuristicAdmission:
    def test_rejects_too_short(self):
        ctrl = HeuristicAdmissionController()
        result = ctrl.check("hi")
        assert result.admitted is False
        assert result.reason == "Too short"

    def test_admits_actionable_content(self):
        ctrl = HeuristicAdmissionController()
        result = ctrl.check("This project uses ruff for linting with line-length=120")
        assert result.admitted is True
        assert result.score > 0.4

    def test_rejects_vague_content(self):
        ctrl = HeuristicAdmissionController()
        result = ctrl.check("ok")
        assert result.admitted is False

    def test_boosts_specific_content(self):
        ctrl = HeuristicAdmissionController()
        r1 = ctrl.check("something about the project configuration settings")
        r2 = ctrl.check("Uses `ruff` v0.4 at /src/pyproject.toml with MAX_LINE=120")
        assert r2.score > r1.score

    def test_custom_min_score(self):
        strict = HeuristicAdmissionController(min_score=0.8)
        result = strict.check("Uses pytest for testing")
        # Moderate content might not pass strict threshold
        assert result.score < 0.8 or result.admitted is True

    def test_custom_min_length(self):
        ctrl = HeuristicAdmissionController(min_length=5)
        result = ctrl.check("short")
        # "short" is 5 chars, should not be rejected for length
        assert result.reason != "Too short"


class TestLLMAdmission:
    def test_admits_high_score(self):
        def mock_llm(system: str, user: str) -> str:
            return json.dumps({"score": 0.9, "reason": "Actionable repo config"})

        ctrl = LLMAdmissionController(llm_callable=mock_llm)
        result = ctrl.check("Uses ruff with line-length=120")
        assert result.admitted is True
        assert result.score == 0.9
        assert result.reason == "Actionable repo config"

    def test_rejects_low_score(self):
        def mock_llm(system: str, user: str) -> str:
            return json.dumps({"score": 0.2, "reason": "Vague and ephemeral"})

        ctrl = LLMAdmissionController(llm_callable=mock_llm)
        result = ctrl.check("the code looks good")
        assert result.admitted is False
        assert result.score == 0.2

    def test_falls_back_on_invalid_json(self):
        def mock_llm(system: str, user: str) -> str:
            return "not valid json at all"

        ctrl = LLMAdmissionController(llm_callable=mock_llm)
        # Should fall back to heuristic, not crash
        result = ctrl.check("Uses ruff for linting configuration")
        assert isinstance(result.admitted, bool)

    def test_falls_back_on_exception(self):
        def mock_llm(system: str, user: str) -> str:
            raise ConnectionError("API down")

        ctrl = LLMAdmissionController(llm_callable=mock_llm)
        result = ctrl.check("Uses pytest version 7.0 for testing")
        assert isinstance(result.admitted, bool)

    def test_clamps_score(self):
        def mock_llm(system: str, user: str) -> str:
            return json.dumps({"score": 1.5, "reason": "over max"})

        ctrl = LLMAdmissionController(llm_callable=mock_llm)
        result = ctrl.check("test")
        assert result.score <= 1.0

    def test_custom_min_score(self):
        def mock_llm(system: str, user: str) -> str:
            return json.dumps({"score": 0.6, "reason": "moderate"})

        ctrl = LLMAdmissionController(llm_callable=mock_llm, min_score=0.7)
        result = ctrl.check("moderate memory")
        assert result.admitted is False


class TestMemoryWithAdmission:
    def test_default_admits_all(self, tmp_path):
        with Memory(tmp_path) as mem:
            record = mem.add("anything", evidence=ManualRef("note"))
            assert record.content == "anything"

    def test_heuristic_rejects_junk(self, tmp_path):
        ctrl = HeuristicAdmissionController()
        with Memory(tmp_path, admission=ctrl) as mem:
            with pytest.raises(ValueError, match="Memory rejected"):
                mem.add("ok", evidence=ManualRef("note"))

    def test_heuristic_admits_good_content(self, tmp_path):
        ctrl = HeuristicAdmissionController()
        with Memory(tmp_path, admission=ctrl) as mem:
            record = mem.add(
                "This project uses ruff for linting with line-length=120",
                evidence=ManualRef("from pyproject.toml"),
            )
            assert record.content.startswith("This project")

    def test_llm_rejects_low_score(self, tmp_path):
        def mock_llm(system: str, user: str) -> str:
            return json.dumps({"score": 0.1, "reason": "Not useful"})

        ctrl = LLMAdmissionController(llm_callable=mock_llm)
        with Memory(tmp_path, admission=ctrl) as mem:
            with pytest.raises(ValueError, match="Memory rejected"):
                mem.add("the weather is nice", evidence=ManualRef("casual"))
