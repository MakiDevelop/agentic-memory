"""Text tokenization with optional CJK support via jieba."""

from __future__ import annotations

import re

_CJK_RANGE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")

# Common CJK function words / stop characters (filtered in char-level fallback)
_CJK_STOP_CHARS = set(
    "的是在了有不和就都而也要這那個人我你他她它們"
    "嗎吧呢啊哦呀吶麼什為從到與及或如被把讓給用"
    "上下中大小多少可會能想已還很更最怎"
)

try:
    import jieba

    jieba.setLogLevel(20)  # suppress loading messages
    _jieba_available = True
except ImportError:
    _jieba_available = False


def is_jieba_available() -> bool:
    """Check if jieba is available for word segmentation."""
    return _jieba_available


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

    # Fallback: insert spaces around each CJK character, filtering stop chars
    parts: list[str] = []
    for char in text:
        if _CJK_RANGE.match(char):
            if char not in _CJK_STOP_CHARS:
                parts.append(f" {char} ")
        else:
            parts.append(char)
    return " ".join("".join(parts).split())
