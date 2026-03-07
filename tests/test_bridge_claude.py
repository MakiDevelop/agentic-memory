"""Tests for Claude Code Memory Bridge."""

import json

from agentic_memory.bridges.claude import (
    SECTION_MARKER_START,
    generate_mcp_config,
    setup,
    setup_claude_md,
    setup_mcp_config,
)


class TestGenerateMcpConfig:
    def test_without_repo(self):
        config = generate_mcp_config()
        assert "agentic-memory" in config
        assert config["agentic-memory"]["command"] == "am-mcp"
        assert config["agentic-memory"]["args"] == []

    def test_with_repo(self, tmp_path):
        config = generate_mcp_config(str(tmp_path))
        assert config["agentic-memory"]["args"] == ["--repo", str(tmp_path)]


class TestSetupMcpConfig:
    def test_creates_new_file(self, tmp_path):
        changed, msg = setup_mcp_config(str(tmp_path))
        assert changed
        mcp_path = tmp_path / ".mcp.json"
        assert mcp_path.exists()
        config = json.loads(mcp_path.read_text())
        assert "agentic-memory" in config["mcpServers"]

    def test_adds_to_existing_file(self, tmp_path):
        mcp_path = tmp_path / ".mcp.json"
        mcp_path.write_text(json.dumps({"mcpServers": {"other-tool": {"command": "other"}}}))
        changed, msg = setup_mcp_config(str(tmp_path))
        assert changed
        config = json.loads(mcp_path.read_text())
        assert "other-tool" in config["mcpServers"]
        assert "agentic-memory" in config["mcpServers"]

    def test_skips_if_already_configured(self, tmp_path):
        mcp_path = tmp_path / ".mcp.json"
        mcp_path.write_text(json.dumps({
            "mcpServers": {"agentic-memory": {"command": "am-mcp", "args": []}}
        }))
        changed, msg = setup_mcp_config(str(tmp_path))
        assert not changed
        assert "already configured" in msg


class TestSetupClaudeMd:
    def test_creates_new_file(self, tmp_path):
        changed, msg = setup_claude_md(str(tmp_path))
        assert changed
        content = (tmp_path / "CLAUDE.md").read_text()
        assert SECTION_MARKER_START in content
        assert "memory_query" in content

    def test_appends_to_existing_file(self, tmp_path):
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("# My Project\n\nExisting content.\n")
        changed, msg = setup_claude_md(str(tmp_path))
        assert changed
        content = claude_md.read_text()
        assert "Existing content." in content
        assert SECTION_MARKER_START in content

    def test_skips_if_already_present(self, tmp_path):
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text(f"# Project\n\n{SECTION_MARKER_START}\n")
        changed, msg = setup_claude_md(str(tmp_path))
        assert not changed
        assert "already present" in msg


class TestSetup:
    def test_full_setup(self, tmp_path):
        messages = setup(str(tmp_path))
        assert len(messages) == 2
        assert all("✓" in m for m in messages)
        assert (tmp_path / ".mcp.json").exists()
        assert (tmp_path / "CLAUDE.md").exists()

    def test_idempotent(self, tmp_path):
        setup(str(tmp_path))
        messages = setup(str(tmp_path))
        assert all("·" in m for m in messages)
