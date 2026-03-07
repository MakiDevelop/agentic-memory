# agentic-memory (memcite)

[![PyPI](https://img.shields.io/pypi/v/memcite)](https://pypi.org/project/memcite/)
[![CI](https://github.com/MakiDevelop/agentic-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/MakiDevelop/agentic-memory/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **`pip install memcite`** → `from agentic_memory import Memory` → CLI: `am`

Open-source repo memory for AI agents. Every memory has a source, every source gets verified.

## The problem

Your AI agent remembers "this project uses Jest for testing." Two weeks later, someone switches to Vitest. The agent doesn't know. It keeps writing Jest tests and breaks your CI.

This isn't hallucination — the memory *was* correct. It's **stale memory**, and it's worse than hallucination because the agent is confident about it.

## The fix

memcite forces every memory to cite its source. Before using a memory, it checks: **is the source still the same?**

```bash
am add "Uses ruff for linting, line-length=120" --file pyproject.toml --lines 15-20
am query "linting"
# → ✓ Uses ruff for linting, line-length=120
#     pyproject.toml L15-20 [valid]

# Someone changes pyproject.toml → memcite detects it:
am validate
# → ⚠ 1 memory stale (evidence changed)
```

## Quickstart (5 minutes)

```bash
pip install memcite
cd your-project
```

```python
from agentic_memory import Memory, FileRef, ManualRef

mem = Memory(".")

# 1. Store a memory with evidence
mem.add(
    "This project uses ruff for linting with line-length=120",
    evidence=FileRef("pyproject.toml", lines=(15, 20)),
)

# 2. Store a rule with importance
mem.add(
    "Never force-push to main",
    evidence=ManualRef("team convention"),
    kind="rule",
    importance=3,
)

# 3. Query — citations are re-validated automatically
result = mem.query("What linter does this project use?")
print(result.memories[0].content)        # "ruff with line-length=120"
print(result.citations[0].status.value)  # "valid" or "stale"

# 4. Agent-ready context string (plug directly into prompts)
context = mem.search_context("coding standards", kind="rule", min_importance=2)

# 5. Track what your agent actually uses
mem.mark_adopted(result.memories[0].id, agent_name="claude")

# 6. System health
metrics = mem.eval_metrics()
print(f"Adoption rate: {metrics.adoption_rate:.0%}")
```

## Tested on real projects

We deployed memcite across 5 projects of different types to validate the design:

| Project | Type | Memories | Kind distribution | What memcite guards |
|---------|------|----------|-------------------|---------------------|
| mk-brain | AI knowledge pipeline | 6 | fact | Architecture — detect drift when code changes |
| momo-home-ai | Home AI assistant | 8 | fact | Config — found real bugs from stale settings |
| abd-ai-hub | Company monorepo | 6 | rule/antipattern/decision | **Governance rules** — CLAUDE.md as evidence |
| dl-pilot | Download manager | 5 | fact | Platform config and file paths |
| geo-checker | GEO tool | 4 | fact | Deployment settings |

**Key discovery:** abd-ai-hub uses `CLAUDE.md` as evidence source for deployment rules. When someone edits `CLAUDE.md`, rule memories are immediately flagged stale — the agent knows governance changed before acting on outdated rules. This is memcite acting as a **constitutional review mechanism** for AI agents.

### Benchmark numbers

| Metric | Result |
|--------|--------|
| Query latency | 0.077s (6 memories, SQLite FTS5) |
| Full validate | 0.073s (6 memories) |
| Storage overhead | ~8-10 KB per memory |
| Stale detection (v0.5+) | 1 true positive out of 4 flagged (75% false positive elimination vs v0.4) |
| CJK search accuracy | 100% (multi-word and single-char, with jieba tokenization) |

### False positive elimination

In v0.4, inserting a single line in a file caused **all 4 FileRef memories** pointing to that file to be flagged stale (line numbers shifted). In v0.5+, content snapshot + fuzzy relocation reduced this to **1/4 flagged** — and that 1 was a genuine content change.

### Known limitation

If a memory's **content is wrong but the evidence file hasn't changed**, memcite will report it as valid with full confidence. memcite validates that evidence hasn't drifted — it does not verify that the memory accurately describes the evidence. Content-level validation requires an optional `ContentValidator` (keyword overlap or LLM-based).

## Design Principles

1. **No Evidence, No Memory** — `add()` without a citation raises an error
2. **Validate Before Use** — `query()` re-checks citations by default
3. **Decay What's Stale** — confidence drops when evidence changes; invalid memories are deprioritized

## Evidence Types

| Type | What it tracks | Validation method |
|------|---------------|-------------------|
| `FileRef` | File path + line range + content snapshot | Content match + fuzzy relocation when lines shift |
| `GitCommitRef` | Commit SHA + file | Verify commit exists in history |
| `URLRef` | Web URL | HTTP HEAD check + content hash |
| `ManualRef` | Human-provided note | No auto-validation (always trusted) |

## Features

**Core**
- **Repo-scoped** — each repository gets its own memory namespace
- **Local-first** — SQLite storage, no external services required
- **Citation-backed** — every memory traces back to a verifiable source
- **Auto-validation** — stale evidence is detected before it misleads your agent
- **Confidence scoring** — memories with invalid citations get deprioritized
- **Content snapshot + fuzzy relocation** — when lines shift, memcite finds where the content moved

**Agentic**
- **Memory classification** — `fact`, `rule`, `antipattern`, `preference`, `decision`
- **Importance scoring** — 0-3 priority, query results sorted by importance
- **TTL / expiration** — ephemeral memories auto-expire
- **Deduplication** — identical content detected by hash
- **Conflict detection** — warns when new memories contradict existing ones
- **Adoption tracking** — `mark_adopted()` to measure which memories agents actually use

**Infrastructure**
- **Retrieval logging** — every query logged with IDs, count, latency
- **Eval metrics** — adoption rate, query stats, health indicators
- **Compact** — clean up expired memories in batch
- **CLI** — `am add`, `am query`, `am validate`, `am status`, `am list`
- **MCP Server** — 10 tools for Claude Code / Cursor
- **REST API** — FastAPI with OpenAPI docs

## Installation

```bash
pip install memcite
```

With extras:
```bash
pip install memcite[mcp]     # MCP server for Claude Code
pip install memcite[api]     # REST API server (FastAPI)
pip install memcite[cjk]     # Chinese/Japanese/Korean tokenization
```

## CLI

```bash
# Add memories with evidence
am add "Uses pytest for testing" --file tests/conftest.py --lines 1-10
am add "No force push to main" --note "team rule" --kind rule --importance 3
am add "Sprint ends Friday" --note "standup" --ttl 604800  # 1 week

# Query with filters
am query "test framework"
am query "coding rules" --kind rule --min-importance 2

# Validate + CI integration
am validate                # check all citations
am validate --exit-code    # exits non-zero if any INVALID (for CI)

# Housekeeping
am status
am list
am delete <memory-id>
```

## MCP Server (Claude Code / Cursor / etc.)

memcite includes a built-in MCP server that **runs locally on your machine** — no cloud service, no API key, no deployment needed.

**Setup:** add this to your project's `.mcp.json`:

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

Or use: `am claude-setup` (auto-generates `.mcp.json` + memory protocol in `CLAUDE.md`)

**10 MCP tools available:**

| Tool | Description |
|------|-------------|
| `memory_add` | Add a memory with evidence, kind, importance, TTL |
| `memory_query` | Search with filters + automatic citation validation |
| `memory_search_context` | Formatted context block for agent prompts |
| `memory_adopt` | Mark a memory as actually used by the agent |
| `memory_validate` | Re-check all evidence citations |
| `memory_compact` | Remove expired memories |
| `memory_metrics` | Query count, adoption rate, health stats |
| `memory_status` | Summary of valid/stale/invalid counts |
| `memory_list` | List all stored memories |
| `memory_delete` | Delete a specific memory |

## REST API

```bash
am-server --repo /path/to/repo --port 8080
```

OpenAPI docs at `http://localhost:8080/docs`.

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
from agentic_memory import Memory, HeuristicAdmissionController, ManualRef

mem = Memory("./my-project", admission=HeuristicAdmissionController())
mem.add("ok", evidence=ManualRef("chat"))  # raises ValueError — too vague
```

## Roadmap

- [x] Core SDK — add / query / validate with citation enforcement
- [x] CLI tool
- [x] MCP Server — 10 tools for Claude Code and other MCP clients
- [x] Admission control — heuristic + LLM-based scoring
- [x] Hybrid search — FTS5 + TF-IDF vector fusion
- [x] REST API server — FastAPI with OpenAPI docs
- [x] Agentic features — kind, importance, TTL, dedup, conflict detection
- [x] Adoption tracking — measure which memories agents actually use
- [ ] GitHub App / GitLab integration
- [ ] LangChain / LlamaIndex integration
- [ ] Web dashboard

## Compared to

| | mem0 | Zep | LangMem | **memcite** |
|---|---|---|---|---|
| Vector search | Yes | Yes | Yes | Yes |
| Forced citations | No | No | No | **Yes** |
| Source validation | No | No | No | **Yes** |
| Staleness detection | No | No | No | **Yes** |
| Repo-scoped | No | No | No | **Yes** |
| Memory classification | No | No | No | **Yes** |
| Conflict detection | No | No | No | **Yes** |
| Adoption tracking | No | No | No | **Yes** |
| Self-hosted | Yes | Yes | Yes | Yes |

## Built with

This project was built using a four-in-one AI collaboration model:

- **[Maki](https://github.com/MakiDevelop)** — Product direction, architecture decisions, testing, final review
- **Claude (Opus 4.6)** — Implementation, code review, documentation
- **Codex (o4-mini)** — Engineering review, bug detection, test validation
- **Gemini (2.5 Pro)** — Architecture analysis, strategy, gap analysis

## License

MIT
