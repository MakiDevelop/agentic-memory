"""Claude Code Memory Bridge — auto-setup memcite for Claude Code sessions.

Generates MCP config and CLAUDE.md memory protocol instructions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

CLAUDE_MD_SECTION = """\

## Memory Protocol (memcite)

This project uses [memcite](https://github.com/MakiDevelop/agentic-memory) for persistent, citation-backed memory.

### Session Start
- Call `memory_query` with keywords relevant to your current task to load context from previous sessions.
- Example: `memory_query("test framework")`, `memory_query("deployment config")`

### During Work
- When you discover important project conventions, configs, or architectural decisions, store them:
  - Use `memory_add` with `evidence_type="file"` and the relevant file path + line range.
  - Use `memory_add` with `evidence_type="git_commit"` for decisions tied to specific commits.
  - Only store knowledge that would be useful across sessions — not transient task details.

### Session End
- Store any architectural decisions or project conventions discovered during this session.
- Run `memory_validate` periodically to detect stale memories.

### Rules
- Every memory MUST have a citation (no manual evidence unless truly no file source exists).
- Prefer `FileRef` evidence — it enables automatic staleness detection.
- Keep memories concise and actionable (e.g., "Uses ruff with line-length=120", not "the project has a linter").
"""

SECTION_MARKER_START = "## Memory Protocol (memcite)"
SECTION_MARKER_END = "### Rules"


def generate_mcp_config(repo_path: str | None = None) -> dict:
    """Generate MCP server config for .mcp.json."""
    args = ["--repo", os.path.abspath(repo_path)] if repo_path else []
    return {
        "agentic-memory": {
            "command": "am-mcp",
            "args": args,
        }
    }


def setup_mcp_config(repo_path: str) -> tuple[bool, str]:
    """Add or update agentic-memory entry in .mcp.json.

    Returns (changed, message).
    """
    mcp_path = Path(repo_path) / ".mcp.json"

    if mcp_path.exists():
        with open(mcp_path) as f:
            config = json.load(f)
    else:
        config = {}

    if "mcpServers" not in config:
        config["mcpServers"] = {}

    if "agentic-memory" in config["mcpServers"]:
        return False, f"agentic-memory already configured in {mcp_path}"

    config["mcpServers"].update(generate_mcp_config(repo_path))

    with open(mcp_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    return True, f"Added agentic-memory to {mcp_path}"


def setup_claude_md(repo_path: str) -> tuple[bool, str]:
    """Add memory protocol section to CLAUDE.md.

    Returns (changed, message).
    """
    claude_md_path = Path(repo_path) / "CLAUDE.md"

    if claude_md_path.exists():
        content = claude_md_path.read_text()
        if SECTION_MARKER_START in content:
            return False, f"Memory protocol already present in {claude_md_path}"
        new_content = content.rstrip() + "\n" + CLAUDE_MD_SECTION
    else:
        new_content = CLAUDE_MD_SECTION.lstrip()

    claude_md_path.write_text(new_content)
    return True, f"Added memory protocol to {claude_md_path}"


def setup(repo_path: str | None = None) -> list[str]:
    """Run full Claude Code setup. Returns list of status messages."""
    repo = repo_path or "."
    messages = []

    changed, msg = setup_mcp_config(repo)
    prefix = "✓" if changed else "·"
    messages.append(f"{prefix} {msg}")

    changed, msg = setup_claude_md(repo)
    prefix = "✓" if changed else "·"
    messages.append(f"{prefix} {msg}")

    return messages
