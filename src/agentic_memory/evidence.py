"""Evidence types for citation-backed memories."""

from __future__ import annotations

import hashlib
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agentic_memory.models import ValidationStatus


class Evidence(ABC):
    """Base class for all evidence types."""

    @abstractmethod
    def validate(self, repo_path: str) -> tuple[ValidationStatus, str]:
        """Validate this evidence against the current state.

        Returns (status, message) tuple.
        """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> Evidence:
        """Deserialize from dictionary."""

    @abstractmethod
    def short_label(self) -> str:
        """Human-readable short label for display."""


def _file_content_hash(path: str, start: int | None = None, end: int | None = None) -> str:
    """Hash file content, optionally only specific lines."""
    try:
        with open(path) as f:
            lines = f.readlines()
        if start is not None and end is not None:
            lines = lines[start - 1 : end]  # 1-indexed
        return hashlib.sha256("".join(lines).encode()).hexdigest()[:16]
    except (FileNotFoundError, OSError):
        return ""


@dataclass
class FileRef(Evidence):
    """Evidence from a file at specific line range."""

    path: str
    lines: tuple[int, int] | None = None
    content_hash: str = field(default="", repr=False)

    def __post_init__(self):
        if not self.content_hash:
            # Will be set when added to a Memory with a repo_path
            pass

    def capture_hash(self, repo_path: str) -> None:
        """Capture current content hash for future validation."""
        full_path = os.path.join(repo_path, self.path)
        start, end = self.lines if self.lines else (None, None)
        self.content_hash = _file_content_hash(full_path, start, end)

    def validate(self, repo_path: str) -> tuple[ValidationStatus, str]:
        full_path = os.path.join(repo_path, self.path)
        if not os.path.exists(full_path):
            return ValidationStatus.INVALID, f"File not found: {self.path}"

        if not self.content_hash:
            return ValidationStatus.VALID, "No hash to compare (legacy memory)"

        start, end = self.lines if self.lines else (None, None)
        current_hash = _file_content_hash(full_path, start, end)
        if current_hash != self.content_hash:
            return ValidationStatus.STALE, f"Content changed at {self.path}" + (
                f" L{start}-{end}" if start else ""
            )
        return ValidationStatus.VALID, "Content matches"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "file",
            "path": self.path,
            "lines": list(self.lines) if self.lines else None,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileRef:
        return cls(
            path=data["path"],
            lines=tuple(data["lines"]) if data.get("lines") else None,
            content_hash=data.get("content_hash", ""),
        )

    def short_label(self) -> str:
        label = self.path
        if self.lines:
            label += f" L{self.lines[0]}-{self.lines[1]}"
        return label


@dataclass
class GitCommitRef(Evidence):
    """Evidence from a specific git commit."""

    sha: str
    file_path: str | None = None
    message: str = ""

    def validate(self, repo_path: str) -> tuple[ValidationStatus, str]:
        try:
            result = subprocess.run(
                ["git", "cat-file", "-t", self.sha],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return ValidationStatus.INVALID, f"Commit {self.sha[:8]} not found"
            return ValidationStatus.VALID, f"Commit {self.sha[:8]} exists"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ValidationStatus.UNCHECKED, "Git not available"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "git_commit",
            "sha": self.sha,
            "file_path": self.file_path,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GitCommitRef:
        return cls(
            sha=data["sha"],
            file_path=data.get("file_path"),
            message=data.get("message", ""),
        )

    def short_label(self) -> str:
        label = f"commit:{self.sha[:8]}"
        if self.file_path:
            label += f" ({self.file_path})"
        return label


@dataclass
class URLRef(Evidence):
    """Evidence from a web URL."""

    url: str
    content_hash: str = field(default="", repr=False)

    def validate(self, repo_path: str) -> tuple[ValidationStatus, str]:
        # URL validation is best-effort: just check reachability
        try:
            import urllib.request

            req = urllib.request.Request(self.url, method="HEAD")
            req.add_header("User-Agent", "agentic-memory/0.1")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status < 400:
                    return ValidationStatus.VALID, f"URL reachable (HTTP {resp.status})"
                return ValidationStatus.STALE, f"URL returned HTTP {resp.status}"
        except Exception as e:
            return ValidationStatus.STALE, f"URL check failed: {e}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "url",
            "url": self.url,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> URLRef:
        return cls(url=data["url"], content_hash=data.get("content_hash", ""))

    def short_label(self) -> str:
        return self.url


@dataclass
class ManualRef(Evidence):
    """Human-provided evidence note. Always trusted."""

    note: str

    def validate(self, repo_path: str) -> tuple[ValidationStatus, str]:
        return ValidationStatus.VALID, "Manual evidence (always trusted)"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "manual", "note": self.note}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManualRef:
        return cls(note=data["note"])

    def short_label(self) -> str:
        return f"manual: {self.note[:50]}"


def evidence_from_dict(data: dict[str, Any]) -> Evidence:
    """Deserialize evidence from a dictionary based on type field."""
    type_map: dict[str, type[Evidence]] = {
        "file": FileRef,
        "git_commit": GitCommitRef,
        "url": URLRef,
        "manual": ManualRef,
    }
    evidence_type = data.get("type", "")
    cls = type_map.get(evidence_type)
    if cls is None:
        raise ValueError(f"Unknown evidence type: {evidence_type}")
    return cls.from_dict(data)
