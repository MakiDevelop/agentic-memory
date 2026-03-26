"""CLI for agentic-memory."""

from __future__ import annotations

import argparse
import sys

from agentic_memory.evidence import FileRef, GitCommitRef, ManualRef, URLRef
from agentic_memory.memory import Memory
from agentic_memory.models import ValidationStatus


def _get_memory(repo_path: str | None = None) -> Memory:
    return Memory(repo_path or ".")


def cmd_add(args: argparse.Namespace) -> None:
    """Add a memory with evidence."""
    evidence: FileRef | GitCommitRef | URLRef | ManualRef
    if args.file:
        lines = None
        if args.lines:
            parts = args.lines.split("-")
            lines = (int(parts[0]), int(parts[1]))
        evidence = FileRef(path=args.file, lines=lines)
    elif args.url:
        evidence = URLRef(url=args.url)
    elif args.commit:
        evidence = GitCommitRef(sha=args.commit, file_path=args.commit_file)
    else:
        evidence = ManualRef(note=args.note or "manually added")

    mem = _get_memory(args.repo)
    record = mem.add(
        args.content,
        evidence=evidence,
        tags=args.tag or [],
        kind=args.kind or "fact",
        importance=args.importance,
        ttl_seconds=args.ttl,
    )
    mem.close()
    print(f"Added memory {record.id}: {record.content}")
    print(f"  Evidence: {record.evidence_label}")
    print(f"  Kind: {record.kind.value} | Importance: {record.importance}")
    print(f"  Status: {record.validation_status.value}")


def cmd_query(args: argparse.Namespace) -> None:
    """Query memories."""
    mem = _get_memory(args.repo)
    result = mem.query(
        args.query,
        limit=args.limit,
        validate=not args.no_validate,
        kind=args.kind,
        min_importance=args.min_importance,
    )
    mem.close()

    if not result.memories:
        print("No memories found.")
        return

    status_icon = {"valid": "✓", "stale": "⚠", "invalid": "✗", "unchecked": "?"}
    for i, memory in enumerate(result.memories, 1):
        icon = status_icon.get(memory.validation_status.value, "?")
        print(f"{i}. {memory.content}")
        for evidence in memory.evidence_list:
            print(f"   {icon} {evidence.short_label()} [{memory.validation_status.value}]")
        print(f"   confidence: {memory.confidence:.1f} | kind: {memory.kind.value} | importance: {memory.importance}")
        if memory.validation_message:
            print(f"   {memory.validation_message}")
        print()


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate all memories."""
    mem = _get_memory(args.repo)
    problematic = mem.validate()
    status = mem.status()
    mem.close()

    print(f"✓ {status['valid']} valid")
    if status["stale"]:
        print(f"⚠ {status['stale']} stale")
    if status["invalid"]:
        print(f"✗ {status['invalid']} invalid")

    if problematic:
        print("\nProblematic memories:")
        for record in problematic:
            icon = "⚠" if record.validation_status == ValidationStatus.STALE else "✗"
            print(f"  {icon} [{record.id}] {record.content}")
            print(f"    {record.validation_message}")
        if args.exit_code:
            sys.exit(1)
    else:
        print("\nAll memories are valid.")


def cmd_status(args: argparse.Namespace) -> None:
    """Show memory status."""
    mem = _get_memory(args.repo)
    s = mem.status()
    mem.close()

    print(f"Total: {s['total']} memories")
    print(f"  ✓ Valid:     {s['valid']}")
    print(f"  ⚠ Stale:     {s['stale']}")
    print(f"  ✗ Invalid:   {s['invalid']}")
    print(f"  ? Unchecked: {s['unchecked']}")
    if s.get("expired", 0):
        print(f"  ⏰ Expired:   {s['expired']}")


def cmd_delete(args: argparse.Namespace) -> None:
    """Delete a memory by ID."""
    mem = _get_memory(args.repo)
    if mem.delete(args.memory_id):
        print(f"Deleted memory {args.memory_id}")
    else:
        print(f"Memory {args.memory_id} not found")
    mem.close()


def cmd_claude_setup(args: argparse.Namespace) -> None:
    """Set up memcite for Claude Code."""
    from agentic_memory.bridges.claude import setup

    messages = setup(args.repo)
    for msg in messages:
        print(msg)
    print("\nDone! Restart Claude Code to activate memory tools.")


def cmd_watch(args: argparse.Namespace) -> None:
    """Watch recent git commits and suggest memories."""
    from agentic_memory.watcher import watch

    suggestions = watch(args.repo or ".", commits=args.commits)
    if not suggestions:
        print("No memory-worthy changes found in recent commits.")
        return

    print(f"Found {len(suggestions)} suggested memories:\n")

    mem = None
    added = 0
    for i, s in enumerate(suggestions, 1):
        loc = f"{s.file_path}"
        if s.lines:
            loc += f" L{s.lines[0]}-{s.lines[1]}"
        print(f"  {i}. [{s.kind}] {s.content}")
        print(f"     {loc} | importance={s.importance}")
        print(f"     reason: {s.reason}")
        print()

        if args.auto:
            if mem is None:
                mem = _get_memory(args.repo)
            try:
                # Always use FileRef for stale detection (content snapshot + fuzzy relocation)
                evidence = FileRef(path=s.file_path, lines=s.lines)
                record = mem.add(
                    s.content, evidence=evidence,
                    kind=s.kind, importance=s.importance,
                )
                print(f"     → Added as {record.id}")
                added += 1
            except (TypeError, ValueError) as e:
                print(f"     → Skipped: {e}")

    if mem:
        mem.close()
    if args.auto and added > 0:
        print(f"\nAuto-added {added} memories.")
    elif not args.auto:
        print("Run with --auto to automatically add these memories.")


def cmd_list(args: argparse.Namespace) -> None:
    """List all memories."""
    mem = _get_memory(args.repo)
    records = mem.list_all(limit=args.limit)
    mem.close()

    if not records:
        print("No memories stored.")
        return

    for record in records:
        status_icon = {"valid": "✓", "stale": "⚠", "invalid": "✗", "unchecked": "?"}
        icon = status_icon.get(record.validation_status.value, "?")
        print(f"{icon} [{record.id}] {record.content}")
        print(f"  Evidence: {record.evidence_label} | {record.kind.value} | importance: {record.importance}")
        print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="am",
        description="agentic-memory: Repo memory for AI agents with citation verification.",
    )
    parser.add_argument("--repo", default=None, help="Repository path (default: current directory)")

    sub = parser.add_subparsers(dest="command")

    # add
    add_p = sub.add_parser("add", help="Add a memory with evidence")
    add_p.add_argument("content", help="Memory content")
    add_p.add_argument("--file", "-f", help="File path for evidence")
    add_p.add_argument("--lines", "-l", help="Line range (e.g., 1-10)")
    add_p.add_argument("--url", "-u", help="URL for evidence")
    add_p.add_argument("--commit", "-c", help="Git commit SHA for evidence")
    add_p.add_argument("--commit-file", help="File path within the commit")
    add_p.add_argument("--note", "-n", help="Manual evidence note")
    add_p.add_argument("--tag", "-t", action="append", help="Tags (repeatable)")
    add_p.add_argument("--kind", "-k", choices=["fact", "rule", "antipattern", "preference", "decision"], default="fact", help="Memory kind (default: fact)")
    add_p.add_argument("--importance", "-i", type=int, default=1, choices=[0, 1, 2, 3], help="Importance 0-3 (default: 1)")
    add_p.add_argument("--ttl", type=int, default=None, help="TTL in seconds (default: never expires)")

    # query
    query_p = sub.add_parser("query", help="Query memories")
    query_p.add_argument("query", help="Search query")
    query_p.add_argument("--limit", type=int, default=5, help="Max results")
    query_p.add_argument("--no-validate", action="store_true", help="Skip citation validation")
    query_p.add_argument("--kind", "-k", choices=["fact", "rule", "antipattern", "preference", "decision"], default=None, help="Filter by kind")
    query_p.add_argument("--min-importance", type=int, default=0, help="Minimum importance (0-3)")

    # validate
    validate_p = sub.add_parser("validate", help="Validate all memories")
    validate_p.add_argument("--exit-code", action="store_true", help="Exit with code 1 if any memory is stale/invalid (for CI)")

    # status
    sub.add_parser("status", help="Show memory status")

    # delete
    delete_p = sub.add_parser("delete", help="Delete a memory by ID")
    delete_p.add_argument("memory_id", help="Memory ID to delete")

    # claude-setup
    sub.add_parser("claude-setup", help="Set up memcite for Claude Code (MCP config + CLAUDE.md)")

    # watch
    watch_p = sub.add_parser("watch", help="Analyze recent git commits and suggest memories")
    watch_p.add_argument("--commits", type=int, default=5, help="Number of recent commits to analyze (default: 5)")
    watch_p.add_argument("--auto", action="store_true", help="Automatically add suggested memories")

    # list
    list_p = sub.add_parser("list", help="List all memories")
    list_p.add_argument("--limit", type=int, default=50, help="Max results")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "add": cmd_add,
        "query": cmd_query,
        "validate": cmd_validate,
        "status": cmd_status,
        "list": cmd_list,
        "delete": cmd_delete,
        "watch": cmd_watch,
        "claude-setup": cmd_claude_setup,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
