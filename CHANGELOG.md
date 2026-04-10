# Changelog

## [1.0.0] — 2026-04-10

Major release. Five new subsystems built on top of v0.8.0 — all backward-compatible, all opt-in.

### Added

**Plugin Architecture**
- `backends.py`: `StorageBackend` / `SearchBackend` Protocol (runtime-checkable)
- `plugins.py`: Three plugin registries (evidence / backends / search) discovered via `importlib.metadata.entry_points`
- `migration.py`: File-based schema migration engine with pre-migration backup and Python hooks
- `migrations/`: Bundled SQL files (v002–v007), replacing the inline `_upgrade_v*` cascade in `store.py`
- Third-party plugins can register custom evidence types via `[project.entry-points."agentic_memory.evidence"]`

**Memory Graph**
- `graph.py`: `MemoryGraph`, `RelationType`, `MemoryEdge`
- Four typed relationships: `contradicts`, `supports`, `supersedes`, `depends_on`
- BFS traversal with `max_depth` and direction control
- Cycle detection for `supersedes` edges
- `Memory.add_relation()`, `get_relations()`, `traverse()`, `supersede()`, `remove_relation()`
- `Memory.delete()` now auto-cleans graph edges
- Schema v006: `memory_edges` table + `memories.superseded_by` / `auto_downgraded` columns

**Semantic Search**
- `semantic.py`: `SentenceTransformerEmbedding` + `ONNXEmbedding` (both opt-in, implement existing `EmbeddingProvider` Protocol)
- TF-IDF remains the default — semantic providers are optional extras
- Schema v007: `memory_embeddings` PK migrated from `(memory_id)` to `(memory_id, model_id)` so TF-IDF and semantic embeddings can coexist per memory
- New extras: `pip install memcite[embedding]` (sentence-transformers) / `memcite[embedding-onnx]` (lighter ONNX path)

**Cross-repo Federation**
- `federation.py`: `FederatedMemory` — query memories across multiple registered repositories
- Federation registry at `~/.agentic-memory/federation.db`, separate from per-repo DBs
- Federated repos are opened **read-only** (`?mode=ro`) to prevent accidental writes
- Schema validation on register (rejects non-memcite SQLite files)
- Results include repo attribution (`repo_path` + `alias`)
- Offline-first — no central server required

**Lifecycle Automation**
- `lifecycle.py`: `LifecycleManager` with three operations:
  - `auto_expire()` — remove TTL-exceeded memories
  - `auto_downgrade_stale()` — lower `importance` of stale memories, mark `auto_downgraded`
  - `auto_compact_by_adoption()` — remove low-adoption memories older than N days (protects `importance=3`)
  - `run_all()` — full pipeline in order
- `Memory` class exposes each as a method (`auto_expire`, `auto_downgrade_stale`, `auto_compact_by_adoption`, `run_lifecycle`)
- `hooks.py`: Git pre-commit hook installer (`install_precommit_hook` / `uninstall_precommit_hook` / `is_installed`)
  - Only touches hooks with the agentic-memory marker — never overwrites foreign hooks unless `force=True`
  - Supports git worktrees

### Changed

- `evidence.py`: `evidence_from_dict` now uses the plugin registry instead of a hardcoded type map
- `store.py`: `_init_schema` delegates to `run_migrations()` with per-version Python hooks (-73 LOC of inline migration logic)
- `pyproject.toml`: new `[project.entry-points]` sections + bundled `migrations/*.sql` in wheel

### Compatibility

- All 188 pre-existing tests continue to pass with zero modifications
- Existing `.agentic-memory.db` files upgrade automatically on next open (v5 → v7), with a backup copied to `.db.v{N}-backup` before any migration
- All new features are opt-in — code that only uses `Memory`, `add`, `query`, `validate`, `compact` from v0.8.0 works unchanged

### Tests

- v0.8.0: 188 tests
- v1.0.0: **296 tests** (287 passing + 9 skipped when `sentence-transformers` model cache is unavailable)
- New test files: `test_plugins.py`, `test_backends.py`, `test_migration.py`, `test_graph.py`, `test_semantic.py`, `test_federation.py`, `test_lifecycle.py`

### Modules

- v0.8.0: 13 modules (~3,363 LOC)
- v1.0.0: **22 modules** (~5,500 LOC)
- New: `backends.py`, `plugins.py`, `migration.py`, `graph.py`, `semantic.py`, `federation.py`, `lifecycle.py`, `hooks.py`, `migrations/` package

---

## [0.8.0] — earlier

See git history for pre-1.0 releases.
