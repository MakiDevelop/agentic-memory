# agentic-memory

Open-source repo memory for AI agents — every memory has a source, every source gets verified.

## Why

AI agents forget everything between sessions. Existing memory layers (mem0, Zep, LangMem) store text in vector DBs but can't tell you *where* that knowledge came from or whether it's still true.

**agentic-memory** enforces a simple rule: **No evidence, no memory.**

Every memory must cite its source (file path + line number, git commit, URL). Before an agent uses a memory, the citation is automatically re-validated. Stale memories get flagged, not silently served.

## How it works

```python
from agentic_memory import Memory, FileRef

mem = Memory("./my-project")

# Store a memory — citation is required
mem.add(
    "This project uses ruff for linting with line-length=120",
    evidence=FileRef("pyproject.toml", lines=(15, 20)),
)

# Query — returns answer + citation status
result = mem.query("What linter does this project use?")
print(result.answer)     # "ruff with line-length=120"
print(result.citations)  # [FileRef("pyproject.toml", L15-20, status=VALID)]

# Validate all memories — find what's gone stale
stale = mem.validate()
# [StaleMemory("ruff config", reason="file content changed at L15")]
```

## Design Principles

1. **No Evidence, No Memory** — `add()` without a citation raises an error
2. **Validate Before Use** — `query()` re-checks citations by default
3. **Decay What's Stale** — confidence drops when evidence changes; invalid memories are deprioritized

## Evidence Types

| Type | What it tracks | Validation method |
|------|---------------|-------------------|
| `FileRef` | File path + line range | Check file exists, content matches |
| `GitCommitRef` | Commit SHA + file | Verify commit exists in history |
| `URLRef` | Web URL | HTTP HEAD check + content hash |
| `ManualRef` | Human-provided note | No auto-validation (always trusted) |

## Features

- **Repo-scoped** — each repository gets its own memory namespace
- **Local-first** — SQLite storage, no external services required
- **Citation-backed** — every memory traces back to a verifiable source
- **Auto-validation** — stale evidence is detected before it misleads your agent
- **Confidence scoring** — memories with invalid citations get deprioritized
- **CLI included** — `am add`, `am query`, `am validate`, `am status`

## Installation

```bash
pip install agentic-memory
```

## CLI Usage

```bash
# Add a memory with file evidence
am add "Uses pytest for testing" --file tests/conftest.py --lines 1-10

# Query memories
am query "What test framework?"

# Validate all memories
am validate

# Show memory status
am status
```

## Roadmap

- [x] Core SDK — add / query / validate with citation enforcement
- [ ] CLI tool
- [ ] MCP Server — use with Claude Code and other MCP clients
- [ ] Admission control — LLM-based scoring to filter low-value memories
- [ ] REST API server
- [ ] LangChain / LlamaIndex integration
- [ ] Web dashboard

## Compared to

| | mem0 | Zep | LangMem | **agentic-memory** |
|---|---|---|---|---|
| Vector search | Yes | Yes | Yes | Yes |
| Forced citations | No | No | No | **Yes** |
| Source validation | No | No | No | **Yes** |
| Staleness detection | No | No | No | **Yes** |
| Repo-scoped | No | No | No | **Yes** |
| Self-hosted | Yes | Yes | Yes | Yes |

## License

MIT
