#!/usr/bin/env python3
"""
Interactive demo — see memcite's citation validation in action.

Usage:
    python examples/demo.py

What it does:
    1. Creates a temp project with a config file
    2. Stores memories with file-backed citations
    3. Queries and shows validated results
    4. Modifies the source file → detects stale memory
    5. Cleans up automatically
"""
import tempfile
import os
import shutil

from agentic_memory import Memory, FileRef, ManualRef


def banner(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def main() -> None:
    # Setup: create a temp project with a real config file
    tmpdir = tempfile.mkdtemp(prefix="memcite-demo-")
    config_path = os.path.join(tmpdir, "pyproject.toml")

    with open(config_path, "w") as f:
        f.write("""\
[tool.ruff]
line-length = 120
target-version = "py310"

[tool.pytest.ini_options]
testpaths = ["tests"]
""")

    try:
        mem = Memory(tmpdir)

        # ── Step 1: Add memories with evidence ──
        banner("Step 1: Store memories with citations")

        m1 = mem.add(
            "Uses ruff for linting with line-length=120",
            evidence=FileRef("pyproject.toml", lines=(1, 3)),
        )
        print(f"  ✓ Added: {m1.content}")
        print(f"    Evidence: pyproject.toml L1-3 (FileRef)")

        m2 = mem.add(
            "Test directory is tests/",
            evidence=FileRef("pyproject.toml", lines=(5, 6)),
        )
        print(f"  ✓ Added: {m2.content}")

        m3 = mem.add(
            "Never force-push to main",
            evidence=ManualRef("team convention"),
            kind="rule",
            importance=3,
        )
        print(f"  ✓ Added: {m3.content} (rule, importance=3)")

        # ── Step 2: Query with auto-validation ──
        banner("Step 2: Query — citations validated automatically")

        result = mem.query("linting")
        for m, c in zip(result.memories, result.citations):
            print(f"  → {m.content}")
            print(f"    Citation: {c.status.value}")

        # ── Step 3: Validate all ──
        banner("Step 3: Validate all memories")

        status = mem.status()
        print(f"  Valid: {status.get('valid', 0)}  Stale: {status.get('stale', 0)}  Invalid: {status.get('invalid', 0)}")

        # ── Step 4: Modify the source → detect staleness ──
        banner("Step 4: Change config → detect stale memory")

        print("  Changing line-length from 120 to 88...")
        with open(config_path, "w") as f:
            f.write("""\
[tool.ruff]
line-length = 88
target-version = "py312"

[tool.pytest.ini_options]
testpaths = ["tests", "integration"]
""")

        stale_memories = mem.validate()
        status = mem.status()
        print(f"\n  Valid: {status.get('valid', 0)}  Stale: {status.get('stale', 0)}  Invalid: {status.get('invalid', 0)}")

        if stale_memories:
            print(f"\n  ⚠ {len(stale_memories)} stale memories detected!")
            for sm in stale_memories:
                print(f"    → {sm.content}")
            print("    Your agent would be warned before using outdated info.")

        # ── Step 5: Show metrics ──
        banner("Step 5: Eval metrics")

        mem.mark_adopted(m1.id, agent_name="demo-agent")
        metrics = mem.eval_metrics()
        print(f"  Total memories: {metrics.total_memories}")
        print(f"  Adoption rate:  {metrics.adoption_rate:.0%}")
        print(f"  Query count:    {metrics.total_queries}")

        banner("Demo complete!")
        print("  memcite caught the config change before your agent")
        print("  could use stale information. That's the whole point.\n")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
