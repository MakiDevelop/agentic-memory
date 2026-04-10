"""Cross-repo Memory Federation — query memories across multiple repositories.

The federation registry lives at ~/.agentic-memory/federation.db (user-global),
separate from per-repo .agentic-memory.db files. Federated repos are opened
read-only to prevent accidental writes.

Usage:
    from agentic_memory.federation import FederatedMemory

    fed = FederatedMemory()
    fed.register("/path/to/repo-a")
    fed.register("/path/to/repo-b", alias="backend")
    results = fed.query("linting rules")
    fed.close()
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from agentic_memory.models import MemoryRecord, ValidationStatus
from agentic_memory.store import SQLiteStore


@dataclass
class FederatedResult:
    """A memory from a federated query, with repo attribution."""

    record: MemoryRecord
    repo_path: str
    repo_alias: str


@dataclass
class FederatedQueryResult:
    """Aggregated results from a cross-repo query."""

    results: list[FederatedResult] = field(default_factory=list)
    repos_queried: int = 0
    repos_failed: int = 0


@dataclass
class RegisteredRepo:
    """A repo registered for federation."""

    repo_path: str
    alias: str
    db_path: str
    registered_at: datetime
    active: bool = True


def _default_federation_db() -> Path:
    """Default federation registry path: ~/.agentic-memory/federation.db"""
    return Path.home() / ".agentic-memory" / "federation.db"


class FederatedMemory:
    """Query memories across multiple registered repositories.

    Each repo's .agentic-memory.db is opened read-only. The federation
    registry itself is a separate SQLite database.
    """

    def __init__(self, registry_path: str | Path | None = None):
        self._registry_path = Path(registry_path) if registry_path else _default_federation_db()
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._registry_path))
        self._conn.row_factory = sqlite3.Row
        self._init_registry()

    def _init_registry(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS federated_repos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_path TEXT NOT NULL UNIQUE,
                alias TEXT NOT NULL,
                db_path TEXT NOT NULL,
                registered_at TEXT NOT NULL,
                active INTEGER DEFAULT 1
            );
        """)
        self._conn.commit()

    def register(self, repo_path: str | Path, alias: str | None = None, db_name: str = ".agentic-memory.db") -> RegisteredRepo:
        """Register a repo for federated queries.

        Args:
            repo_path: Absolute path to the repo root.
            alias: Human-friendly name. Defaults to directory name.
            db_name: Name of the memory DB file within the repo.

        Raises:
            ValueError: If repo_path doesn't exist or DB is missing.
            sqlite3.IntegrityError: If repo is already registered.
        """
        repo = Path(repo_path).resolve()
        if not repo.is_dir():
            raise ValueError(f"Repo path does not exist: {repo}")

        db_file = repo / db_name
        if not db_file.exists():
            raise ValueError(f"No memory database found at {db_file}")

        # Validate it's actually a memcite DB
        try:
            test_conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
            test_conn.execute("SELECT value FROM schema_meta WHERE key='version'")
            test_conn.close()
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
            raise ValueError(f"Not a valid agentic-memory database: {db_file}") from exc

        if alias is None:
            alias = repo.name

        now = datetime.now().isoformat()
        self._conn.execute(
            "INSERT INTO federated_repos (repo_path, alias, db_path, registered_at, active) VALUES (?, ?, ?, ?, 1)",
            (str(repo), alias, str(db_file), now),
        )
        self._conn.commit()

        return RegisteredRepo(
            repo_path=str(repo),
            alias=alias,
            db_path=str(db_file),
            registered_at=datetime.fromisoformat(now),
        )

    def unregister(self, repo_path: str | Path) -> bool:
        """Remove a repo from the federation. Returns True if it existed."""
        repo = str(Path(repo_path).resolve())
        cursor = self._conn.execute("DELETE FROM federated_repos WHERE repo_path = ?", (repo,))
        self._conn.commit()
        return cursor.rowcount > 0

    def list_repos(self, active_only: bool = True) -> list[RegisteredRepo]:
        """List all registered repos."""
        if active_only:
            rows = self._conn.execute(
                "SELECT * FROM federated_repos WHERE active = 1 ORDER BY alias"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM federated_repos ORDER BY alias"
            ).fetchall()

        return [
            RegisteredRepo(
                repo_path=row["repo_path"],
                alias=row["alias"],
                db_path=row["db_path"],
                registered_at=datetime.fromisoformat(row["registered_at"]),
                active=bool(row["active"]),
            )
            for row in rows
        ]

    def query(self, query: str, limit: int = 10) -> FederatedQueryResult:
        """Search across all registered repos and merge results.

        Results are sorted by FTS rank across all repos, with repo attribution.
        Each repo's DB is opened read-only.
        """
        repos = self.list_repos(active_only=True)
        all_results: list[FederatedResult] = []
        repos_failed = 0

        for repo in repos:
            try:
                results = self._query_single_repo(repo, query, limit=limit)
                all_results.extend(results)
            except Exception:
                repos_failed += 1

        # Sort by importance (desc) then confidence (desc), take top N
        all_results.sort(key=lambda r: (r.record.importance, r.record.confidence), reverse=True)

        return FederatedQueryResult(
            results=all_results[:limit],
            repos_queried=len(repos),
            repos_failed=repos_failed,
        )

    def _query_single_repo(self, repo: RegisteredRepo, query: str, limit: int) -> list[FederatedResult]:
        """Query a single repo's DB read-only."""
        db_path = repo.db_path
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"DB not found: {db_path}")

        # Open read-only
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row

        try:
            store = _ReadOnlyStore(conn, repo.repo_path)
            records = store.search(query, limit=limit)
            return [
                FederatedResult(record=r, repo_path=repo.repo_path, repo_alias=repo.alias)
                for r in records
            ]
        finally:
            conn.close()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _ReadOnlyStore:
    """Minimal read-only wrapper to reuse SQLiteStore's search + deserialization logic."""

    def __init__(self, conn: sqlite3.Connection, repo_path: str):
        self._conn = conn
        self._repo_path = repo_path

    def search(self, query: str, limit: int = 10) -> list[MemoryRecord]:
        """FTS search on a read-only connection."""
        from agentic_memory.tokenizer import has_cjk, is_jieba_available, tokenize_for_fts

        tokenized = tokenize_for_fts(query)
        tokens = [f'"{token}"' for token in tokenized.split() if token.strip()]
        if not tokens:
            return []

        if has_cjk(query) and not is_jieba_available():
            safe_query = " OR ".join(tokens)
        else:
            safe_query = " ".join(tokens)

        rows = self._conn.execute(
            """SELECT m.* FROM memories m
               JOIN memories_fts fts ON m.rowid = fts.rowid
               WHERE memories_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (safe_query, limit),
        ).fetchall()

        return [self._row_to_record(row) for row in rows]

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        import json

        from agentic_memory.evidence import evidence_from_dict
        from agentic_memory.models import MemoryKind

        cols = row.keys()
        evidence_raw = json.loads(row["evidence_json"])
        if isinstance(evidence_raw, list):
            evidence = [evidence_from_dict(e) for e in evidence_raw]
        else:
            evidence = evidence_from_dict(evidence_raw)

        return MemoryRecord(
            id=row["id"],
            content=row["content"],
            evidence=evidence,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            confidence=row["confidence"],
            validation_status=ValidationStatus(row["validation_status"]),
            validation_message=row["validation_message"],
            tags=json.loads(row["tags_json"]),
            kind=MemoryKind(row["kind"] if "kind" in cols else "fact"),
            importance=row["importance"] if "importance" in cols else 1,
            ttl_seconds=row["ttl_seconds"] if "ttl_seconds" in cols else None,
            source_hash=row["source_hash"] if "source_hash" in cols else "",
        )
