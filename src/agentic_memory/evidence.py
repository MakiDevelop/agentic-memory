"""Evidence types for citation-backed memories."""

from __future__ import annotations

import hashlib
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agentic_memory.models import ValidationResult, ValidationStatus


def _check_path_within_repo(full_path: str, repo_path: str) -> None:
    """Ensure resolved path is within the repo directory. Raises ValueError on traversal."""
    resolved = os.path.realpath(full_path)
    repo_resolved = os.path.realpath(repo_path)
    if not resolved.startswith(repo_resolved + os.sep) and resolved != repo_resolved:
        raise ValueError(f"Path traversal denied: {full_path} is outside repo {repo_path}")


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

    def validate_detail(self, repo_path: str) -> ValidationResult:
        """Validate with rich result including diff content.

        Default implementation wraps validate(). Subclasses can override
        to provide old_content/new_content for diffs.
        """
        status, message = self.validate(repo_path)
        return ValidationResult(status=status, message=message)

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


def _read_lines(path: str, start: int | None = None, end: int | None = None) -> str:
    """Read file content, optionally only specific lines. Returns empty string on error."""
    try:
        with open(path) as f:
            lines = f.readlines()
        if start is not None and end is not None:
            lines = lines[start - 1 : end]  # 1-indexed
        return "".join(lines)
    except (FileNotFoundError, OSError):
        return ""


def _find_snippet_in_file(path: str, snippet: str) -> tuple[int, int] | None:
    """Search for a content snippet in a file and return its new line range.

    Returns (start_line, end_line) 1-indexed, or None if not found.
    Skips snippets shorter than 20 chars to avoid false positives.
    """
    if not snippet or len(snippet.strip()) < 20:
        return None
    try:
        with open(path) as f:
            full_text = f.read()
    except (FileNotFoundError, OSError):
        return None

    pos = full_text.find(snippet)
    if pos == -1:
        return None

    start_line = full_text[:pos].count("\n") + 1
    lines_in_snippet = snippet.count("\n") + (0 if snippet.endswith("\n") else 1)
    end_line = start_line + lines_in_snippet - 1
    return (start_line, end_line)


@dataclass
class FileRef(Evidence):
    """Evidence from a file at specific line range."""

    path: str
    lines: tuple[int, int] | None = None
    content_hash: str = field(default="", repr=False)
    content_snapshot: str = field(default="", repr=False)

    def __post_init__(self):
        if not self.content_hash:
            # Will be set when added to a Memory with a repo_path
            pass

    def capture_hash(self, repo_path: str) -> None:
        """Capture current content hash and content snapshot for future validation."""
        full_path = os.path.join(repo_path, self.path)
        _check_path_within_repo(full_path, repo_path)
        start, end = self.lines if self.lines else (None, None)
        self.content_hash = _file_content_hash(full_path, start, end)
        self.content_snapshot = _read_lines(full_path, start, end)

    def validate(self, repo_path: str) -> tuple[ValidationStatus, str]:
        full_path = os.path.join(repo_path, self.path)
        try:
            _check_path_within_repo(full_path, repo_path)
        except ValueError as e:
            return ValidationStatus.INVALID, str(e)
        if not os.path.exists(full_path):
            return ValidationStatus.INVALID, f"File not found: {self.path}"

        if not self.content_hash:
            return ValidationStatus.VALID, "No hash to compare (legacy memory)"

        start, end = self.lines if self.lines else (None, None)
        current_hash = _file_content_hash(full_path, start, end)
        if current_hash != self.content_hash:
            # Try fuzzy relocation using content snapshot
            if self.content_snapshot:
                new_pos = _find_snippet_in_file(full_path, self.content_snapshot)
                if new_pos is not None:
                    self.lines = new_pos
                    self.content_hash = _file_content_hash(full_path, new_pos[0], new_pos[1])
                    return ValidationStatus.VALID, (
                        f"Content relocated from L{start}-{end} to L{new_pos[0]}-{new_pos[1]}"
                    )
            return ValidationStatus.STALE, f"Content changed at {self.path}" + (
                f" L{start}-{end}" if start else ""
            )
        return ValidationStatus.VALID, "Content matches"

    def validate_detail(self, repo_path: str) -> ValidationResult:
        """Validate with diff: provides old and new content when stale."""
        full_path = os.path.join(repo_path, self.path)
        status, message = self.validate(repo_path)
        if status == ValidationStatus.STALE:
            start, end = self.lines if self.lines else (None, None)
            new_content = _read_lines(full_path, start, end)
            return ValidationResult(
                status=status,
                message=message,
                old_content=self.content_snapshot or None,
                new_content=new_content or None,
            )
        return ValidationResult(status=status, message=message)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": "file",
            "path": self.path,
            "lines": list(self.lines) if self.lines else None,
            "content_hash": self.content_hash,
        }
        if self.content_snapshot:
            d["content_snapshot"] = self.content_snapshot
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileRef:
        return cls(
            path=data["path"],
            lines=tuple(data["lines"]) if data.get("lines") else None,
            content_hash=data.get("content_hash", ""),
            content_snapshot=data.get("content_snapshot", ""),
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
        # Restrict to safe URL schemes (case-insensitive per RFC 3986)
        url_lower = self.url.lower()
        if not url_lower.startswith(("http://", "https://")):
            return ValidationStatus.INVALID, f"URL scheme not allowed (only http/https): {self.url}"

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
