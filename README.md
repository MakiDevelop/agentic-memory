# agentic-memory (memcite)

[![PyPI](https://img.shields.io/pypi/v/memcite)](https://pypi.org/project/memcite/)
[![CI](https://github.com/MakiDevelop/agentic-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/MakiDevelop/agentic-memory/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Open-source repo memory for AI agents — every memory has a source, every source gets verified.

> **Package name on PyPI: [`memcite`](https://pypi.org/project/memcite/)**

Designed for coding agents, code review agents, and CLI tools that work on a single repository at a time.

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
- **Copilot-inspired design** — repository-scoped memories with evidence and decay, inspired by GitHub's agentic memory architecture
- **CLI included** — `am add`, `am query`, `am validate`, `am status`

## Installation

> **Status: Alpha but usable** — core features are stable, API may evolve.

```bash
pip install memcite
```

With extras:
```bash
pip install memcite[mcp]     # MCP server for Claude Code
pip install memcite[api]     # REST API server (FastAPI)
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

## MCP Server

Add to your `.mcp.json` to use with Claude Code:

```json
{
  "mcpServers": {
    "agentic-memory": {
      "command": "am-mcp",
      "args": ["--repo", "/path/to/your/project"]
    }
  }
}
```

Tools: `memory_add`, `memory_query`, `memory_validate`, `memory_status`, `memory_list`, `memory_delete`

## REST API

```bash
am-server --repo /path/to/repo --port 8080
```

OpenAPI docs at `http://localhost:8080/docs`. Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| POST | `/memories` | Add a memory with evidence |
| POST | `/memories/query` | Hybrid search + citation validation |
| GET | `/memories` | List all memories |
| GET | `/memories/{id}` | Get a specific memory |
| DELETE | `/memories/{id}` | Delete a memory |
| POST | `/memories/validate` | Validate all citations |
| GET | `/status` | Memory status summary |

## Hybrid Search

When initialized with an embedding provider, queries combine FTS5 full-text search with vector similarity:

```python
from agentic_memory import Memory, TFIDFEmbedding, FileRef

mem = Memory("./my-project", embedding=TFIDFEmbedding())
mem.add("Uses ruff for code formatting", evidence=FileRef("pyproject.toml", lines=(1, 5)))

# Finds the memory even though "linting" != "formatting"
result = mem.query("What linter does this project use?")
```

Default weights: FTS5 (0.65) + Vector (0.35). Customize per query:

```python
result = mem.query("linting", fts_weight=0.5, vector_weight=0.5)
```

## Admission Control

Filter out low-value memories before they're stored:

```python
from agentic_memory import Memory, HeuristicAdmissionController

mem = Memory("./my-project", admission=HeuristicAdmissionController())
mem.add("ok", evidence=ManualRef("chat"))  # raises ValueError — too vague
```

Or use LLM-based scoring with any OpenAI-compatible API:

```python
from agentic_memory import LLMAdmissionController

def my_llm(system: str, user: str) -> str:
    # Call your LLM here, return JSON: {"score": 0.0-1.0, "reason": "..."}
    ...

mem = Memory("./my-project", admission=LLMAdmissionController(llm_callable=my_llm))
```

## Real-world Workflows

**PR reviewer agent** — remember repo conventions and enforce them automatically:
```python
mem.add(
    "Logging must use structlog, not stdlib logging",
    evidence=FileRef("docs/conventions.md", lines=(10, 15)),
)

# In your review pipeline
result = mem.query("What logging library should this project use?")
# → "structlog" with citation pointing to docs/conventions.md
```

**Coding agent** — look up project config with verifiable sources:
```python
result = mem.query("What env vars does this service need?")
# → Returns memories citing .env.example with current validation status
# If .env.example was deleted or changed, the memory is flagged as STALE
```

**CI pipeline** — catch drifted knowledge before it causes damage:
```bash
# Add to your CI workflow
am validate --exit-code  # exits non-zero if any memory is INVALID
```

## Roadmap

- [x] Core SDK — add / query / validate with citation enforcement
- [x] CLI tool
- [x] MCP Server — use with Claude Code and other MCP clients
- [x] Admission control — LLM-based scoring to filter low-value memories
- [x] Hybrid search — FTS5 + TF-IDF vector fusion, pluggable embedding providers
- [x] REST API server — FastAPI with OpenAPI docs
- [ ] GitHub App / GitLab integration (webhook + comment bot)
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
