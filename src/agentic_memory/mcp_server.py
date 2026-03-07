"""MCP Server for agentic-memory.

Exposes memory operations as MCP tools for Claude Code and other MCP clients.

Usage:
    python -m agentic_memory.mcp_server [--repo /path/to/repo]
"""

from __future__ import annotations

import argparse
import os

from mcp.server.fastmcp import FastMCP

from agentic_memory.evidence import FileRef, GitCommitRef, ManualRef, URLRef
from agentic_memory.memory import Memory

mcp = FastMCP(
    "agentic-memory",
    instructions="Repo memory for AI agents — every memory has a source, every source gets verified.",
)

_memory: Memory | None = None


def _get_memory() -> Memory:
    global _memory
    if _memory is None:
        repo_path = os.environ.get("AGENTIC_MEMORY_REPO", ".")
        _memory = Memory(repo_path)
    return _memory


def _build_evidence(
    evidence_type: str,
    file_path: str | None = None,
    lines_start: int | None = None,
    lines_end: int | None = None,
    url: str | None = None,
    commit_sha: str | None = None,
    commit_file: str | None = None,
    note: str | None = None,
) -> FileRef | GitCommitRef | URLRef | ManualRef:
    """Build an Evidence object from parameters."""
    if evidence_type == "file":
        if not file_path:
            raise ValueError("file_path is required for file evidence")
        lines = (lines_start, lines_end) if lines_start and lines_end else None
        return FileRef(path=file_path, lines=lines)
    elif evidence_type == "url":
        if not url:
            raise ValueError("url is required for url evidence")
        return URLRef(url=url)
    elif evidence_type == "git_commit":
        if not commit_sha:
            raise ValueError("commit_sha is required for git_commit evidence")
        return GitCommitRef(sha=commit_sha, file_path=commit_file)
    elif evidence_type == "manual":
        return ManualRef(note=note or "manually added")
    else:
        raise ValueError(f"Unknown evidence_type: {evidence_type}. Use: file, url, git_commit, manual")


@mcp.tool()
def memory_add(
    content: str,
    evidence_type: str,
    file_path: str | None = None,
    lines_start: int | None = None,
    lines_end: int | None = None,
    url: str | None = None,
    commit_sha: str | None = None,
    commit_file: str | None = None,
    note: str | None = None,
    tags: list[str] | None = None,
    kind: str = "fact",
    importance: int = 1,
    ttl_seconds: int | None = None,
) -> str:
    """Add a memory with required evidence citation.

    Every memory must have a verifiable source. Choose an evidence_type and provide
    the corresponding parameters.

    Args:
        content: The knowledge to remember (e.g., "This project uses ruff with line-length=120")
        evidence_type: One of "file", "url", "git_commit", "manual"
        file_path: (file) Relative path to the evidence file
        lines_start: (file) Start line number (1-indexed)
        lines_end: (file) End line number (1-indexed)
        url: (url) Web URL as evidence source
        commit_sha: (git_commit) Git commit SHA
        commit_file: (git_commit) File path within the commit
        note: (manual) Human-provided evidence note
        tags: Optional tags for categorization
        kind: Memory type: "fact", "rule", "antipattern", "preference", "decision" (default: "fact")
        importance: Priority 0-3: 0=low, 1=normal, 2=high, 3=critical (default: 1)
        ttl_seconds: Time-to-live in seconds. Omit for permanent memories.
    """
    mem = _get_memory()
    evidence = _build_evidence(
        evidence_type=evidence_type,
        file_path=file_path,
        lines_start=lines_start,
        lines_end=lines_end,
        url=url,
        commit_sha=commit_sha,
        commit_file=commit_file,
        note=note,
    )

    try:
        result = mem.add_with_result(
            content, evidence=evidence, tags=tags,
            kind=kind, importance=importance, ttl_seconds=ttl_seconds,
        )
        conflict_info = ""
        if result.conflicts:
            conflict_info = f"\nConflicts with: {', '.join(c.id for c in result.conflicts)}"
        dedup_info = " (duplicate — returned existing)" if result.was_duplicate else ""
        return (
            f"Memory added [{result.record.id}]{dedup_info}\n"
            f"Content: {result.record.content}\n"
            f"Evidence: {result.record.evidence_label}\n"
            f"Kind: {result.record.kind.value} | Importance: {result.record.importance}\n"
            f"Status: {result.record.validation_status.value}\n"
            f"Confidence: {result.record.confidence}"
            f"{conflict_info}"
        )
    except (TypeError, ValueError) as e:
        return f"Error: {e}"


@mcp.tool()
def memory_query(
    query: str,
    limit: int = 5,
    validate: bool = True,
    include_stale: bool = True,
    kind: str | None = None,
    min_importance: int = 0,
) -> str:
    """Query memories with automatic citation re-validation.

    Searches stored memories and re-validates their evidence citations before returning.
    Stale or invalid citations are flagged with reduced confidence scores.

    Args:
        query: Search query string
        limit: Maximum number of results (default: 5)
        validate: Re-validate citations before returning (default: true)
        include_stale: Include stale/invalid memories in results (default: true)
        kind: Filter by kind: "fact", "rule", "antipattern", "preference", "decision"
        min_importance: Minimum importance level 0-3 (default: 0)
    """
    mem = _get_memory()
    result = mem.query(
        query, limit=limit, validate=validate, include_stale=include_stale,
        kind=kind, min_importance=min_importance,
    )

    if not result.memories:
        return "No memories found."

    status_icons = {"valid": "VALID", "stale": "STALE", "invalid": "INVALID", "unchecked": "UNCHECKED"}
    lines = []
    for i, (memory, citation) in enumerate(zip(result.memories, result.citations), 1):
        icon = status_icons.get(citation.status.value, "?")
        lines.append(
            f"{i}. {memory.content}\n"
            f"   [{icon}] {citation.evidence.short_label()}\n"
            f"   Confidence: {memory.confidence:.1f}"
        )
        if citation.message:
            lines.append(f"   Note: {citation.message}")

    lines.append(f"\nOverall confidence: {result.confidence:.2f}")
    return "\n".join(lines)


@mcp.tool()
def memory_validate() -> str:
    """Validate all stored memories by re-checking their evidence citations.

    Checks whether referenced files still exist and their content hasn't changed.
    Returns a summary of valid, stale, and invalid memories.
    """
    mem = _get_memory()
    problematic = mem.validate()
    status = mem.status()

    lines = [
        "Validation complete:",
        f"  VALID:     {status['valid']}",
        f"  STALE:     {status['stale']}",
        f"  INVALID:   {status['invalid']}",
        f"  UNCHECKED: {status['unchecked']}",
        f"  TOTAL:     {status['total']}",
    ]

    if problematic:
        lines.append("\nProblematic memories:")
        for record in problematic:
            label = "STALE" if record.validation_status.value == "stale" else "INVALID"
            lines.append(f"  [{label}] {record.id}: {record.content}")
            lines.append(f"    Reason: {record.validation_message}")

    return "\n".join(lines)


@mcp.tool()
def memory_status() -> str:
    """Get a summary of all stored memories and their validation status."""
    mem = _get_memory()
    s = mem.status()

    return (
        f"Memory Status:\n"
        f"  Total:     {s['total']}\n"
        f"  Valid:     {s['valid']}\n"
        f"  Stale:     {s['stale']}\n"
        f"  Invalid:   {s['invalid']}\n"
        f"  Unchecked: {s['unchecked']}"
    )


@mcp.tool()
def memory_list(limit: int = 20) -> str:
    """List all stored memories with their evidence and validation status.

    Args:
        limit: Maximum number of memories to list (default: 20)
    """
    mem = _get_memory()
    records = mem.list_all(limit=limit)

    if not records:
        return "No memories stored."

    status_icons = {"valid": "VALID", "stale": "STALE", "invalid": "INVALID", "unchecked": "UNCHECKED"}
    lines = []
    for record in records:
        icon = status_icons.get(record.validation_status.value, "?")
        lines.append(
            f"[{record.id}] {record.content}\n"
            f"  [{icon}] {record.evidence_label} | confidence: {record.confidence:.1f}"
        )

    lines.append(f"\nShowing {len(records)} of {mem.status()['total']} memories")
    return "\n".join(lines)


@mcp.tool()
def memory_adopt(memory_id: str, query: str = "", agent_name: str = "") -> str:
    """Mark a memory as adopted — confirm that you actually used this memory.

    Call this after using a memory from query results to track adoption metrics.

    Args:
        memory_id: The memory ID that was used
        query: The query that retrieved this memory (optional)
        agent_name: Name of the agent that used it (optional)
    """
    mem = _get_memory()
    if mem.mark_adopted(memory_id, query=query, agent_name=agent_name):
        return f"Marked memory {memory_id} as adopted."
    return f"Memory {memory_id} not found."


@mcp.tool()
def memory_compact() -> str:
    """Remove expired memories (TTL exceeded) and return cleanup stats."""
    mem = _get_memory()
    result = mem.compact()
    return (
        f"Compact complete:\n"
        f"  Expired removed: {result.expired_removed}\n"
        f"  Before: {result.total_before}\n"
        f"  After: {result.total_after}"
    )


@mcp.tool()
def memory_search_context(
    query: str,
    limit: int = 5,
    kind: str | None = None,
    min_importance: int = 0,
) -> str:
    """Search memories and return a formatted context block ready for agent use.

    Returns a structured text with memories, citations, confidence, and conflict info.
    Use this when you need memory context for decision-making.

    Args:
        query: Search query string
        limit: Maximum results (default: 5)
        kind: Filter by kind: "fact", "rule", "antipattern", "preference", "decision"
        min_importance: Minimum importance 0-3 (default: 0)
    """
    mem = _get_memory()
    return mem.search_context(query, limit=limit, kind=kind, min_importance=min_importance)


@mcp.tool()
def memory_metrics() -> str:
    """Get health and usage metrics for the memory system.

    Returns query count, memory count, latency, adoption rate, and health indicators.
    """
    mem = _get_memory()
    m = mem.eval_metrics()
    return (
        f"Memory Metrics:\n"
        f"  Total memories: {m.total_memories}\n"
        f"  Total queries: {m.total_queries}\n"
        f"  Avg latency: {m.avg_latency_ms:.1f}ms\n"
        f"  Avg results/query: {m.avg_results_per_query:.1f}\n"
        f"  Total adoptions: {m.total_adoptions}\n"
        f"  Adoption rate: {m.adoption_rate:.1%}\n"
        f"  Expired: {m.expired_count}\n"
        f"  Stale: {m.stale_count}"
    )


@mcp.tool()
def memory_delete(memory_id: str) -> str:
    """Delete a specific memory by its ID.

    Args:
        memory_id: The memory ID to delete
    """
    mem = _get_memory()
    if mem.delete(memory_id):
        return f"Deleted memory {memory_id}"
    return f"Memory {memory_id} not found"


def main():
    parser = argparse.ArgumentParser(description="agentic-memory MCP server")
    parser.add_argument(
        "--repo", default=None, help="Repository path (default: cwd or AGENTIC_MEMORY_REPO env)"
    )
    args = parser.parse_args()

    if args.repo:
        os.environ["AGENTIC_MEMORY_REPO"] = args.repo

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
