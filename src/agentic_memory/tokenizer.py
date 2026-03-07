"""Text tokenization with optional CJK support via jieba."""

from __future__ import annotations

import re

_CJK_RANGE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")

try:
    import jieba

    jieba.setLogLevel(20)  # suppress loading messages
    _jieba_available = True
except ImportError:
    _jieba_available = False


def has_cjk(text: str) -> bool:
    """Check if text contains CJK characters."""
    return bool(_CJK_RANGE.search(text))


def tokenize_for_fts(text: str) -> str:
    """Tokenize text for FTS5 indexing and querying.

    CJK text: uses jieba word segmentation if available,
    falls back to character-level splitting.
    Non-CJK text: returned as-is (FTS5 unicode61 handles it).
    """
    if not has_cjk(text):
        return text

    if _jieba_available:
        tokens = jieba.lcut(text)
        return " ".join(t.strip() for t in tokens if t.strip())

    # Fallback: insert spaces around each CJK character
    parts: list[str] = []
    for char in text:
        if _CJK_RANGE.match(char):
            parts.append(f" {char} ")
        else:
            parts.append(char)
    return " ".join("".join(parts).split())
