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
    record = mem.add(args.content, evidence=evidence, tags=args.tag or [])
    mem.close()
    print(f"Added memory {record.id}: {record.content}")
    print(f"  Evidence: {record.evidence.short_label()}")
    print(f"  Status: {record.validation_status.value}")


def cmd_query(args: argparse.Namespace) -> None:
    """Query memories."""
    mem = _get_memory(args.repo)
    result = mem.query(args.query, limit=args.limit, validate=not args.no_validate)
    mem.close()

    if not result.memories:
        print("No memories found.")
        return

    for i, (memory, citation) in enumerate(zip(result.memories, result.citations), 1):
        status_icon = {"valid": "✓", "stale": "⚠", "invalid": "✗", "unchecked": "?"}
        icon = status_icon.get(citation.status.value, "?")
        print(f"{i}. {memory.content}")
        print(f"   {icon} {citation.evidence.short_label()} [{citation.status.value}]")
        print(f"   confidence: {memory.confidence:.1f}")
        if citation.message:
            print(f"   {citation.message}")
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
        print(f"  Evidence: {record.evidence.short_label()}")
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

    # query
    query_p = sub.add_parser("query", help="Query memories")
    query_p.add_argument("query", help="Search query")
    query_p.add_argument("--limit", type=int, default=5, help="Max results")
    query_p.add_argument("--no-validate", action="store_true", help="Skip citation validation")

    # validate
    sub.add_parser("validate", help="Validate all memories")

    # status
    sub.add_parser("status", help="Show memory status")

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
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
