# agentic-memory (memcite)

[![PyPI](https://img.shields.io/pypi/v/memcite)](https://pypi.org/project/memcite/)
[![CI](https://github.com/MakiDevelop/agentic-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/MakiDevelop/agentic-memory/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](#the-problem) | [中文](#問題)

> **`pip install memcite`** → `from agentic_memory import Memory` → CLI: `am`

**The memory governance layer for AI agents.** Every memory has a source, every source gets verified — like unit tests for agent context.

Other memory tools help agents *remember*. memcite helps agents *remember correctly* — with forced citations, automatic stale detection, and CI-ready validation.

開源的 AI agent 記憶治理層。不只是記住，而是記得正確 — 強制引用、過期偵測、CI 驗證一條龍。

---

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

We deployed memcite across 4 projects of different types to validate the design:

| Project | Type | Memories | Kind distribution | What memcite guards |
|---------|------|----------|-------------------|---------------------|
| mk-brain | AI knowledge pipeline | 6 | fact | Architecture — detect drift when code changes |
| momo-home-ai | Home AI assistant | 8 | fact | Config — found real bugs from stale settings |
| dl-pilot | Download manager | 5 | fact | Platform config and file paths |
| geo-checker | GEO tool | 4 | fact | Deployment settings |

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

## Use Cases

- **Automating issue triaging** with persistent contextual memory
- **Assisting pull request review** using historical code understanding
- **Maintaining developer workflows** through memory-aware agents
- **Coordinating multi-agent maintenance tasks** across repositories

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Your AI Agent                     │
│              (Claude / GPT / Cursor)                │
└──────────┬──────────────────────┬───────────────────┘
           │ query / add          │ MCP / REST / CLI
           ▼                      ▼
┌─────────────────────────────────────────────────────┐
│                    memcite Core                     │
│  ┌───────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  Memory    │  │ Evidence │  │ Hybrid Search    │ │
│  │  Manager   │→ │ Validator│  │ FTS5 + TF-IDF   │ │
│  └───────────┘  └────┬─────┘  └──────────────────┘ │
│                      │                              │
│  ┌───────────┐  ┌────▼─────┐  ┌──────────────────┐ │
│  │ Admission │  │ Citation │  │ Adoption         │ │
│  │ Control   │  │ Store    │  │ Tracker          │ │
│  └───────────┘  └──────────┘  └──────────────────┘ │
└──────────┬──────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────┐
│              SQLite + FTS5 (local file)             │
│                .agentic-memory.db                   │
└─────────────────────────────────────────────────────┘
           │
     Evidence Sources
     ├── FileRef    → local files (content snapshot + line tracking)
     ├── GitCommitRef → git history
     ├── URLRef     → web pages (HTTP HEAD + content hash)
     └── ManualRef  → human notes (always trusted)
```

## Example Workflow

```
1. Agent receives a new issue or task
2. Retrieves historical context from memory (with citation validation)
3. Analyzes related code and past decisions — stale memories are flagged
4. Suggests resolution or PR changes based on verified context
5. Updates memory with new findings, citing the source files
```

**Try it yourself** — the interactive demo walks through this flow in 5 seconds:

```bash
python examples/demo.py
```

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
pip install memcite[mcp]        # MCP server for Claude Code
pip install memcite[api]        # REST API server (FastAPI)
pip install memcite[langchain]  # LangChain retriever integration
pip install memcite[cjk]        # Chinese/Japanese/Korean tokenization
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

# Watch git commits for memory-worthy changes
am watch                   # analyze last 5 commits
am watch --commits 10      # analyze last 10 commits
am watch --auto            # auto-add suggested memories

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

## CI Integration (GitHub Action)

Add memory linting to your CI pipeline — catch stale agent context before it ships:

```yaml
# .github/workflows/memory-lint.yml
name: Memory Lint
on: [pull_request]
jobs:
  memory-lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: MakiDevelop/agentic-memory@main
        with:
          command: "validate --exit-code"
```

This fails the build if any memory has stale or invalid citations — like a linter, but for agent context.

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

## LangChain Integration

Use memcite as a LangChain retriever — every document comes with citation metadata:

```python
from agentic_memory.bridges.langchain import MemciteRetriever

retriever = MemciteRetriever(repo_path="./my-project")
docs = retriever.invoke("What linter does this project use?")

for doc in docs:
    print(doc.page_content)
    print(f"  status: {doc.metadata['validation_status']}")
    print(f"  evidence: {doc.metadata['evidence']}")
```

## Admission Control

Filter out low-value memories before they're stored:

```python
from agentic_memory import Memory, HeuristicAdmissionController, ManualRef

mem = Memory("./my-project", admission=HeuristicAdmissionController())
mem.add("ok", evidence=ManualRef("chat"))  # raises ValueError — too vague
```

## Scope

This project focuses on the **memory and context layer** for AI-assisted maintenance. It does not aim to replace existing CI/CD platforms, code hosting, or agent frameworks — it plugs into them as the memory substrate.

## Roadmap

- [x] Core SDK — add / query / validate with citation enforcement
- [x] CLI tool
- [x] MCP Server — 10 tools for Claude Code and other MCP clients
- [x] Admission control — heuristic + LLM-based scoring
- [x] Hybrid search — FTS5 + TF-IDF vector fusion
- [x] REST API server — FastAPI with OpenAPI docs
- [x] Agentic features — kind, importance, TTL, dedup, conflict detection
- [x] Adoption tracking — measure which memories agents actually use
- [x] GitHub Action — CI memory linting with `am validate --exit-code`
- [x] Path traversal protection — repo-scoped file access enforcement
- [x] Git Watch Mode — `am watch` analyzes commits and suggests memories
- [x] LangChain integration — `MemciteRetriever` with citation metadata
- [ ] LlamaIndex integration
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

---

# 中文

## 問題

你的 AI agent 記住了「這個專案用 Jest 測試」。兩週後有人換成 Vitest，agent 不知道，繼續寫 Jest 測試，CI 直接炸掉。

這不是幻覺 — 記憶*曾經*是對的。這是**過期記憶（stale memory）**，比幻覺更危險，因為 agent 對它深信不疑。

## 解法

memcite 強制每條記憶都要引用來源。使用記憶前會先檢查：**來源還是一樣的嗎？**

```bash
am add "使用 ruff 做 linting, line-length=120" --file pyproject.toml --lines 15-20
am query "linting"
# → ✓ 使用 ruff 做 linting, line-length=120
#     pyproject.toml L15-20 [valid]

# 有人改了 pyproject.toml → memcite 偵測到：
am validate
# → ⚠ 1 條記憶過期（證據已變更）
```

## 快速開始（5 分鐘）

```bash
pip install memcite
cd your-project
```

```python
from agentic_memory import Memory, FileRef, ManualRef

mem = Memory(".")

# 1. 儲存記憶並附上證據
mem.add(
    "本專案使用 ruff 做 linting, line-length=120",
    evidence=FileRef("pyproject.toml", lines=(15, 20)),
)

# 2. 儲存規則並設定重要度
mem.add(
    "禁止 force-push 到 main",
    evidence=ManualRef("團隊慣例"),
    kind="rule",
    importance=3,
)

# 3. 查詢 — 引用自動重新驗證
result = mem.query("這個專案用什麼 linter？")
print(result.memories[0].content)        # "ruff with line-length=120"
print(result.citations[0].status.value)  # "valid" 或 "stale"

# 4. Agent-ready 的上下文字串（直接塞進 prompt）
context = mem.search_context("coding standards", kind="rule", min_importance=2)

# 5. 追蹤 agent 實際使用了哪些記憶
mem.mark_adopted(result.memories[0].id, agent_name="claude")

# 6. 系統健康度
metrics = mem.eval_metrics()
print(f"採用率: {metrics.adoption_rate:.0%}")
```

## 使用場景

- **自動化 Issue 分流** — 透過持久化的上下文記憶
- **輔助 PR Review** — 利用歷史程式碼理解
- **維護開發流程** — 透過具備記憶的 agent
- **跨 Repo 多 Agent 協作** — 協調維護任務

## 架構

```
┌─────────────────────────────────────────────────────┐
│                   你的 AI Agent                     │
│              (Claude / GPT / Cursor)                │
└──────────┬──────────────────────┬───────────────────┘
           │ query / add          │ MCP / REST / CLI
           ▼                      ▼
┌─────────────────────────────────────────────────────┐
│                    memcite 核心                     │
│  ┌───────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  Memory    │  │ Evidence │  │ Hybrid Search    │ │
│  │  Manager   │→ │ Validator│  │ FTS5 + TF-IDF   │ │
│  └───────────┘  └────┬─────┘  └──────────────────┘ │
│                      │                              │
│  ┌───────────┐  ┌────▼─────┐  ┌──────────────────┐ │
│  │ Admission │  │ Citation │  │ Adoption         │ │
│  │ Control   │  │ Store    │  │ Tracker          │ │
│  └───────────┘  └──────────┘  └──────────────────┘ │
└──────────┬──────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────┐
│              SQLite + FTS5（本地檔案）                │
│                .agentic-memory.db                   │
└─────────────────────────────────────────────────────┘
           │
     證據來源
     ├── FileRef       → 本地檔案（內容快照 + 行號追蹤）
     ├── GitCommitRef  → git 歷史
     ├── URLRef        → 網頁（HTTP HEAD + 內容雜湊）
     └── ManualRef     → 人工備註（永遠信任）
```

## 範例流程

```
1. Agent 收到新的 issue 或任務
2. 從記憶中取得歷史上下文（引用同步驗證）
3. 分析相關程式碼與過去的決策 — 過期記憶會被標記
4. 基於已驗證的上下文，建議修復方案或 PR 變更
5. 將新發現寫入記憶，引用來源檔案
```

**自己試試看** — 互動 demo 5 秒跑完整個流程：

```bash
python examples/demo.py
```

## 設計原則

1. **沒有證據就沒有記憶** — `add()` 不附引用會直接報錯
2. **使用前先驗證** — `query()` 預設會重新檢查引用
3. **過期就衰減** — 證據變更時信心分數下降，無效記憶被降權

## 證據類型

| 類型 | 追蹤什麼 | 驗證方式 |
|------|---------|---------|
| `FileRef` | 檔案路徑 + 行範圍 + 內容快照 | 內容比對 + 行號偏移時模糊重定位 |
| `GitCommitRef` | Commit SHA + 檔案 | 驗證 commit 是否存在於歷史 |
| `URLRef` | 網頁 URL | HTTP HEAD 檢查 + 內容雜湊 |
| `ManualRef` | 人工備註 | 不自動驗證（永遠信任） |

## 功能

**核心**
- **Repo 範圍** — 每個 repo 有獨立的記憶命名空間
- **本地優先** — SQLite 儲存，不需要外部服務
- **引用驅動** — 每條記憶都可追溯到可驗證的來源
- **自動驗證** — 在 agent 被誤導前偵測過期證據
- **信心評分** — 引用失效的記憶會被降權
- **內容快照 + 模糊重定位** — 行號偏移時自動找到內容搬到哪裡

**Agentic**
- **記憶分類** — `fact`、`rule`、`antipattern`、`preference`、`decision`
- **重要度評分** — 0-3 優先級，查詢結果依重要度排序
- **TTL / 過期** — 臨時記憶自動過期
- **去重** — 靠雜湊偵測重複內容
- **衝突偵測** — 新記憶與既有記憶矛盾時發出警告
- **採用追蹤** — `mark_adopted()` 測量 agent 實際用了哪些記憶

**基礎設施**
- **查詢日誌** — 每次查詢記錄 ID、數量、延遲
- **評估指標** — 採用率、查詢統計、健康度指標
- **壓縮** — 批次清理過期記憶
- **CLI** — `am add`、`am query`、`am validate`、`am status`、`am list`
- **MCP Server** — 10 個工具，支援 Claude Code / Cursor
- **REST API** — FastAPI + OpenAPI 文件
- **GitHub Action** — CI 記憶品質檢查，過期引用直接擋 PR
- **路徑安全** — repo 邊界強制檢查，防止路徑穿越

## 與其他方案比較

| | mem0 | Zep | LangMem | **memcite** |
|---|---|---|---|---|
| 向量搜尋 | Yes | Yes | Yes | Yes |
| 強制引用 | No | No | No | **Yes** |
| 來源驗證 | No | No | No | **Yes** |
| 過期偵測 | No | No | No | **Yes** |
| Repo 範圍 | No | No | No | **Yes** |
| 記憶分類 | No | No | No | **Yes** |
| 衝突偵測 | No | No | No | **Yes** |
| 採用追蹤 | No | No | No | **Yes** |
| 自架部署 | Yes | Yes | Yes | Yes |

## 授權

MIT
