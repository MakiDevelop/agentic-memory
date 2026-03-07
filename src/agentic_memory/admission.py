"""Admission control for filtering low-value memories."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol


@dataclass
class AdmissionResult:
    """Result of an admission check."""

    admitted: bool
    score: float  # 0.0 to 1.0
    reason: str


class AdmissionController(ABC):
    """Base class for admission controllers."""

    @abstractmethod
    def check(self, content: str, tags: list[str] | None = None) -> AdmissionResult:
        """Decide whether a memory should be admitted.

        Args:
            content: The memory content to evaluate.
            tags: Optional tags for context.

        Returns:
            AdmissionResult with decision, score, and reason.
        """


class AlwaysAdmit(AdmissionController):
    """Admits everything. Default when no LLM is configured."""

    def check(self, content: str, tags: list[str] | None = None) -> AdmissionResult:
        return AdmissionResult(admitted=True, score=1.0, reason="No admission control configured")


class HeuristicAdmissionController(AdmissionController):
    """Rule-based admission control without LLM dependency.

    Filters out memories that are too short, too vague, or lack actionable content.
    """

    def __init__(self, min_length: int = 10, min_score: float = 0.4):
        self.min_length = min_length
        self.min_score = min_score

    def check(self, content: str, tags: list[str] | None = None) -> AdmissionResult:
        score = 0.0
        reasons: list[str] = []

        # Length check
        if len(content.strip()) < self.min_length:
            return AdmissionResult(admitted=False, score=0.0, reason="Too short")

        # Actionable keywords boost
        actionable_patterns = [
            r"\b(use[sd]?|requires?|must|should|always|never|prefer|avoid)\b",
            r"\b(config|setting|convention|rule|pattern|standard)\b",
            r"\b(version|dependency|api|endpoint|schema|migration)\b",
            r"\b(deploy|build|test|lint|format|ci)\b",
        ]
        matches = sum(1 for p in actionable_patterns if re.search(p, content, re.IGNORECASE))
        score += min(matches * 0.2, 0.6)

        if matches > 0:
            reasons.append(f"{matches} actionable pattern(s)")

        # Specificity boost: has numbers, file paths, or code-like content
        specificity_patterns = [
            r"\d+",  # numbers
            r"[/\\]\w+\.\w+",  # file paths
            r"`[^`]+`",  # inline code
            r"[A-Z_]{2,}",  # constants/env vars
        ]
        specifics = sum(1 for p in specificity_patterns if re.search(p, content))
        score += min(specifics * 0.15, 0.3)

        if specifics > 0:
            reasons.append(f"{specifics} specific reference(s)")

        # Penalty: vague/filler content
        vague_patterns = [
            r"^(ok|okay|sure|yes|no|maybe|idk|lol|haha)$",
            r"^(the weather|today is|i feel|hello|hi|hey)\b",
            r"^.{0,5}$",  # very short after strip
        ]
        for p in vague_patterns:
            if re.search(p, content.strip(), re.IGNORECASE):
                score = max(0.0, score - 0.3)
                reasons.append("Vague content detected")
                break

        # Base score for reasonable length
        if len(content.strip()) >= 20:
            score += 0.1

        score = min(score, 1.0)
        admitted = score >= self.min_score
        reason = "; ".join(reasons) if reasons else "No actionable patterns found"

        return AdmissionResult(admitted=admitted, score=round(score, 2), reason=reason)


class LLMAdmissionController(AdmissionController):
    """LLM-based admission control using any OpenAI-compatible API.

    Asks the LLM to score whether a memory is worth keeping.
    Falls back to HeuristicAdmissionController if LLM call fails.
    """

    SYSTEM_PROMPT = """You are a memory admission controller for an AI coding agent.
Your job is to decide whether a piece of knowledge is worth storing as a long-term memory.

A good memory is:
- Actionable: it will influence future decisions (e.g., "this repo uses ruff with line-length=120")
- Specific: it contains concrete details (file paths, version numbers, API names)
- Durable: it won't become irrelevant in a few minutes

A bad memory is:
- Vague: "the code looks good" or "we discussed testing"
- Ephemeral: "the build is currently failing" (temporary state)
- Redundant: common knowledge that any developer would know

Respond with JSON only: {"score": 0.0-1.0, "reason": "brief explanation"}"""

    USER_PROMPT_TEMPLATE = """Should this be stored as a long-term repository memory?

Content: {content}
Tags: {tags}

Respond with JSON only: {{"score": 0.0-1.0, "reason": "brief explanation"}}"""

    def __init__(
        self,
        llm_callable: LLMCallable,
        min_score: float = 0.5,
        fallback: AdmissionController | None = None,
    ):
        """
        Args:
            llm_callable: A callable that takes (system_prompt, user_prompt) and returns text.
            min_score: Minimum score to admit a memory.
            fallback: Controller to use if LLM call fails.
        """
        self.llm_callable = llm_callable
        self.min_score = min_score
        self._fallback = fallback or HeuristicAdmissionController(min_score=min_score)

    def check(self, content: str, tags: list[str] | None = None) -> AdmissionResult:
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            content=content,
            tags=", ".join(tags) if tags else "none",
        )

        try:
            response = self.llm_callable(self.SYSTEM_PROMPT, user_prompt)
            parsed = json.loads(response)
            score = float(parsed["score"])
            reason = parsed.get("reason", "")
            score = max(0.0, min(1.0, score))
            return AdmissionResult(
                admitted=score >= self.min_score,
                score=round(score, 2),
                reason=reason,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return self._fallback.check(content, tags)
        except Exception:
            return self._fallback.check(content, tags)


class LLMCallable(Protocol):
    """Protocol for LLM callables used by LLMAdmissionController."""

    def __call__(self, system_prompt: str, user_prompt: str) -> str: ...
