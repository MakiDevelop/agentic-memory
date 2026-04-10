"""Git hook installation for automated memory validation.

Installs a pre-commit hook that runs `am validate --exit-code --offline`
before each commit, warning about stale memories.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

PRECOMMIT_HOOK_MARKER = "# agentic-memory precommit hook"

PRECOMMIT_HOOK_SCRIPT = """#!/bin/sh
# agentic-memory precommit hook
# Validates memory citations against current repo state.
# To skip: git commit --no-verify

if command -v am >/dev/null 2>&1; then
    am validate --exit-code || {
        echo ""
        echo "agentic-memory: stale or invalid memories detected."
        echo "Run 'am validate' to see details, or 'am status' for a summary."
        echo "To commit anyway: git commit --no-verify"
        exit 1
    }
fi
"""


def _hooks_dir(repo_path: str | Path) -> Path:
    """Return the .git/hooks directory for a repo."""
    repo = Path(repo_path).resolve()
    git_dir = repo / ".git"
    if not git_dir.exists():
        raise ValueError(f"Not a git repository: {repo}")

    # Handle git worktrees where .git is a file pointing to the real dir
    if git_dir.is_file():
        content = git_dir.read_text().strip()
        if content.startswith("gitdir:"):
            real_git_dir = Path(content.split(":", 1)[1].strip())
            if not real_git_dir.is_absolute():
                real_git_dir = repo / real_git_dir
            git_dir = real_git_dir

    hooks = git_dir / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    return hooks


def is_installed(repo_path: str | Path) -> bool:
    """Check whether the agentic-memory precommit hook is installed."""
    try:
        hook_path = _hooks_dir(repo_path) / "pre-commit"
    except ValueError:
        return False
    if not hook_path.exists():
        return False
    try:
        content = hook_path.read_text()
    except OSError:
        return False
    return PRECOMMIT_HOOK_MARKER in content


def install_precommit_hook(repo_path: str | Path, force: bool = False) -> Path:
    """Install the agentic-memory pre-commit hook in a git repo.

    Args:
        repo_path: Path to the git repository root.
        force: Overwrite an existing hook that's not ours.

    Returns:
        Path to the installed hook file.

    Raises:
        ValueError: If repo is not a git repository.
        FileExistsError: If a different pre-commit hook exists and force=False.
    """
    hook_path = _hooks_dir(repo_path) / "pre-commit"

    if hook_path.exists():
        existing = hook_path.read_text()
        if PRECOMMIT_HOOK_MARKER in existing:
            # Our hook already installed, just refresh the content
            hook_path.write_text(PRECOMMIT_HOOK_SCRIPT)
        elif force:
            hook_path.write_text(PRECOMMIT_HOOK_SCRIPT)
        else:
            raise FileExistsError(
                f"A pre-commit hook already exists at {hook_path}. "
                "Use force=True to overwrite."
            )
    else:
        hook_path.write_text(PRECOMMIT_HOOK_SCRIPT)

    # Make executable
    current = os.stat(hook_path).st_mode
    os.chmod(hook_path, current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return hook_path


def uninstall_precommit_hook(repo_path: str | Path) -> bool:
    """Remove the agentic-memory pre-commit hook.

    Only removes it if it contains our marker — leaves other hooks alone.

    Returns:
        True if the hook was removed, False if it wasn't installed.
    """
    try:
        hook_path = _hooks_dir(repo_path) / "pre-commit"
    except ValueError:
        return False

    if not hook_path.exists():
        return False

    content = hook_path.read_text()
    if PRECOMMIT_HOOK_MARKER not in content:
        return False  # not our hook

    hook_path.unlink()
    return True
