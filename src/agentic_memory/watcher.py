"""Git watcher — analyze recent commits and suggest memories."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass


@dataclass
class SuggestedMemory:
    """A memory suggestion extracted from git diff analysis."""

    content: str
    file_path: str
    lines: tuple[int, int] | None = None
    kind: str = "fact"
    importance: int = 1
    reason: str = ""
    commit_sha: str = ""


# File patterns worth watching for memory-worthy changes
_CONFIG_PATTERNS = {
    "pyproject.toml", "package.json", "Cargo.toml", "go.mod",
    "Makefile", "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    ".env.example", "requirements.txt", "setup.py", "setup.cfg",
    "tsconfig.json", "jest.config.js", "vitest.config.ts",
    ".eslintrc", ".eslintrc.json", ".prettierrc",
    "ruff.toml", ".flake8", ".pylintrc",
}

_CONFIG_EXTENSIONS = {".toml", ".yaml", ".yml", ".json", ".cfg", ".ini", ".conf"}

# Patterns that indicate important changes
_IMPORTANT_PATTERNS = [
    (re.compile(r'^\+.*["\']?version["\']?\s*[:=]', re.MULTILINE | re.IGNORECASE), "version bump", 2),
    (re.compile(r'^\+.*(?:dependencies|requires|deps)\b', re.MULTILINE | re.IGNORECASE), "dependency change", 2),
    (re.compile(
        r'^\+.*["\']?(?:port|host|url|endpoint|base.?url)["\']?\s*[:=]', re.MULTILINE | re.IGNORECASE,
    ), "endpoint/config", 2),
    (re.compile(r'^\+\s*(?:FROM\s+\S+:\S+|image:\s+\S+)', re.MULTILINE), "container image change", 2),
    (re.compile(
        r'^\+.*(?:DROP|ALTER|CREATE)\s+(?:TABLE|INDEX|DATABASE)', re.MULTILINE | re.IGNORECASE,
    ), "schema change", 3),
]


def _is_config_file(path: str) -> bool:
    """Check if a file path looks like a config/build file."""
    basename = os.path.basename(path)
    if basename in _CONFIG_PATTERNS:
        return True
    _, ext = os.path.splitext(basename)
    return ext in _CONFIG_EXTENSIONS


def _run_git(repo_path: str, *args: str) -> str:
    """Run a git command and return stdout."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout if result.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _extract_added_lines(diff_text: str) -> list[tuple[str, int]]:
    """Extract added lines with their approximate line numbers from a diff chunk."""
    results = []
    current_line = 0
    for line in diff_text.split("\n"):
        if line.startswith("@@"):
            # Parse @@ -x,y +a,b @@
            match = re.search(r'\+(\d+)', line)
            if match:
                current_line = int(match.group(1))
            continue
        if line.startswith("+") and not line.startswith("+++"):
            content = line[1:].strip()
            if content:
                results.append((content, current_line))
            current_line += 1
        elif not line.startswith("-"):
            current_line += 1
    return results


def _analyze_diff_for_file(file_path: str, diff_text: str) -> list[SuggestedMemory]:
    """Analyze a single file's diff and suggest memories."""
    suggestions = []

    # Check important patterns
    for pattern, label, importance in _IMPORTANT_PATTERNS:
        matches = pattern.findall(diff_text)
        if matches:
            # Get the first added line matching the pattern
            added_lines = _extract_added_lines(diff_text)
            for line_content, line_num in added_lines:
                if pattern.search("+" + line_content):
                    suggestions.append(SuggestedMemory(
                        content=f"{label}: {line_content.strip()}",
                        file_path=file_path,
                        lines=(line_num, line_num),
                        kind="fact",
                        importance=importance,
                        reason=f"Detected {label} in {file_path}",
                    ))
                    break

    # For config files, summarize key-value additions
    if _is_config_file(file_path):
        added_lines = _extract_added_lines(diff_text)
        kv_lines = []
        for line_content, line_num in added_lines:
            # Skip comments and blank lines
            if line_content.startswith(("#", "//", ";")) or not line_content:
                continue
            # Look for key=value or key: value patterns
            if re.match(r'^[\w.-]+\s*[:=]', line_content):
                kv_lines.append((line_content, line_num))

        if kv_lines and not suggestions:
            # Summarize: take first few meaningful additions
            summary_lines = kv_lines[:3]
            content_parts = [entry[0].strip() for entry in summary_lines]
            start_line = summary_lines[0][1]
            end_line = summary_lines[-1][1]
            suggestions.append(SuggestedMemory(
                content=f"Config change in {file_path}: {'; '.join(content_parts)}",
                file_path=file_path,
                lines=(start_line, end_line),
                kind="fact",
                importance=1,
                reason=f"Config file modified: {file_path}",
            ))

    return suggestions


def watch(repo_path: str, commits: int = 5) -> list[SuggestedMemory]:
    """Analyze recent git commits and suggest memories.

    Args:
        repo_path: Path to the git repository.
        commits: Number of recent commits to analyze.

    Returns:
        List of suggested memories based on diff analysis.
    """
    # Get recent commits
    log_output = _run_git(repo_path, "log", f"-{commits}", "--format=%H", "--diff-filter=AM")
    if not log_output.strip():
        return []

    commit_shas = [sha.strip() for sha in log_output.strip().split("\n") if sha.strip()]
    all_suggestions: list[SuggestedMemory] = []
    seen_contents: set[str] = set()

    for sha in commit_shas:
        # Get diff for this commit
        diff_output = _run_git(repo_path, "diff", f"{sha}~1..{sha}", "--unified=3")
        if not diff_output:
            # First commit or error
            diff_output = _run_git(repo_path, "show", sha, "--format=", "--unified=3")
        if not diff_output:
            continue

        # Split diff by file
        file_diffs = re.split(r'^diff --git a/(\S+) b/(\S+)', diff_output, flags=re.MULTILINE)

        # file_diffs: ['', file_a, file_b, diff_content, file_a, file_b, diff_content, ...]
        i = 1
        while i + 2 < len(file_diffs):
            file_path = file_diffs[i + 1]  # b/ path
            diff_content = file_diffs[i + 2]
            i += 3

            suggestions = _analyze_diff_for_file(file_path, diff_content)
            for s in suggestions:
                s.commit_sha = sha
                if s.content not in seen_contents:
                    seen_contents.add(s.content)
                    all_suggestions.append(s)

    return all_suggestions
