"""Content-level validation: check if memory content matches evidence file content."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agentic_memory.evidence import Evidence, FileRef
from agentic_memory.tokenizer import has_cjk, tokenize_for_fts


@dataclass
class ContentValidationResult:
    """Result of content-level validation."""

    consistent: bool
    score: float  # 0.0 to 1.0 — how well memory matches evidence
    reason: str


def _read_single_evidence_content(evidence: Evidence, repo_path: str) -> str | None:
    """Read content from a single evidence source."""
    if isinstance(evidence, FileRef):
        full_path = os.path.join(repo_path, evidence.path)
        try:
            with open(full_path) as f:
                lines = f.readlines()
            if evidence.lines:
                start, end = evidence.lines
                lines = lines[start - 1 : end]  # 1-indexed
            return "".join(lines)
        except (FileNotFoundError, OSError):
            return None
    return None


def read_evidence_content(evidence: Evidence | list[Evidence], repo_path: str) -> str | None:
    """Read the actual text content that an evidence points to.

    For multi-evidence, returns the first readable content found.
    Returns None if no evidence supports content reading.
    """
    items = evidence if isinstance(evidence, list) else [evidence]
    for e in items:
        content = _read_single_evidence_content(e, repo_path)
        if content is not None:
            return content
    return None


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from text, handling CJK."""
    if has_cjk(text):
        text = tokenize_for_fts(text)
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    # Filter out very short tokens and common stop words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "has", "have", "had", "do", "does", "did", "will", "would",
        "can", "could", "should", "may", "might", "must", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after",
        "and", "but", "or", "not", "no", "if", "then", "else",
        "this", "that", "it", "its", "they", "them", "their",
        "的", "是", "在", "了", "和", "與", "及", "或",
    }
    return {t for t in tokens if len(t) > 1 and t not in stop_words}


@runtime_checkable
class ContentValidator(Protocol):
    """Protocol for content-level validation."""

    def check(self, memory_content: str, evidence_content: str) -> ContentValidationResult:
        """Check if memory content is consistent with evidence content."""
        ...


class KeywordOverlapValidator:
    """Heuristic content validator based on keyword overlap.

    Checks whether key terms from the memory appear in the evidence content.
    """

    def __init__(self, min_overlap: float = 0.3) -> None:
        self.min_overlap = min_overlap

    def check(self, memory_content: str, evidence_content: str) -> ContentValidationResult:
        memory_kw = _extract_keywords(memory_content)
        evidence_kw = _extract_keywords(evidence_content)

        if not memory_kw:
            return ContentValidationResult(
                consistent=True, score=1.0, reason="No keywords to check"
            )

        overlap = memory_kw & evidence_kw
        score = len(overlap) / len(memory_kw)

        if score >= self.min_overlap:
            return ContentValidationResult(
                consistent=True,
                score=score,
                reason=f"Keyword overlap {score:.0%} ({len(overlap)}/{len(memory_kw)} terms match)",
            )

        missing = memory_kw - evidence_kw
        return ContentValidationResult(
            consistent=False,
            score=score,
            reason=f"Low keyword overlap {score:.0%} — terms not found in evidence: {', '.join(sorted(missing)[:5])}",
        )


class LLMContentValidator:
    """LLM-based content validator.

    Asks an LLM whether the memory accurately describes the evidence content.
    Falls back to KeywordOverlapValidator on failure.
    """

    def __init__(
        self,
        llm_callable: LLMCallable | None = None,
        fallback: ContentValidator | None = None,
    ) -> None:
        self._llm = llm_callable
        self._fallback = fallback or KeywordOverlapValidator()

    def check(self, memory_content: str, evidence_content: str) -> ContentValidationResult:
        if self._llm is None:
            return self._fallback.check(memory_content, evidence_content)

        system_prompt = (
            "You are a fact-checker. Given a MEMORY statement and the SOURCE content it claims to be based on, "
            "determine if the memory accurately describes the source.\n"
            'Respond with JSON: {"consistent": true/false, "score": 0.0-1.0, "reason": "..."}'
        )
        user_prompt = f"MEMORY: {memory_content}\n\nSOURCE:\n{evidence_content[:2000]}"

        try:
            response = self._llm(system_prompt, user_prompt)
            import json

            data = json.loads(response)
            return ContentValidationResult(
                consistent=bool(data.get("consistent", True)),
                score=max(0.0, min(1.0, float(data.get("score", 1.0)))),
                reason=str(data.get("reason", "LLM check passed")),
            )
        except Exception:
            return self._fallback.check(memory_content, evidence_content)


class LLMCallable(Protocol):
    """Protocol for LLM function used by LLMContentValidator."""

    def __call__(self, system: str, user: str) -> str: ...
