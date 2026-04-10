"""Schema migration engine for agentic-memory SQLite databases."""

from __future__ import annotations

import importlib.resources
import re
import shutil
import sqlite3
from collections.abc import Callable
from pathlib import Path


def get_migration_files() -> dict[int, list[str]]:
    """Discover bundled migration SQL files grouped by schema version."""
    migrations: dict[int, list[tuple[str, str]]] = {}

    try:
        files = importlib.resources.files("agentic_memory.migrations")
    except (AttributeError, ModuleNotFoundError, TypeError):
        return {}

    for item in files.iterdir():
        if not item.name.endswith(".sql"):
            continue
        match = re.match(r"v(\d+)_", item.name)
        if match is None:
            continue
        version = int(match.group(1))
        migrations.setdefault(version, []).append((item.name, item.read_text(encoding="utf-8")))

    return {
        version: [sql for _, sql in sorted(files_for_version, key=lambda entry: entry[0])]
        for version, files_for_version in sorted(migrations.items())
    }


def get_current_version(conn: sqlite3.Connection) -> int:
    """Read the current schema version from schema_meta."""
    try:
        row = conn.execute("SELECT value FROM schema_meta WHERE key = 'version'").fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row[0]) if row else 0


def backup_database(db_path: Path, current_version: int) -> Path | None:
    """Create an on-disk backup before running migrations."""
    if not db_path.exists():
        return None
    backup_path = db_path.with_suffix(f".db.v{current_version}-backup")
    shutil.copy2(db_path, backup_path)
    return backup_path


def _iter_sql_statements(sql: str) -> list[str]:
    cleaned_sql = "\n".join(
        line
        for line in sql.splitlines()
        if line.strip() and not line.strip().startswith("--")
    )
    if not cleaned_sql:
        return []
    return [statement.strip() for statement in cleaned_sql.split(";") if statement.strip()]


def run_migrations(
    conn: sqlite3.Connection,
    db_path: Path | None = None,
    python_hooks: dict[int, Callable[[sqlite3.Connection], None]] | None = None,
) -> int:
    """Run all pending schema migrations and return the final version."""
    current = get_current_version(conn)
    all_migrations = get_migration_files()

    all_versions = set(all_migrations)
    if python_hooks:
        all_versions.update(python_hooks)

    pending = sorted(version for version in all_versions if version > current)
    if not pending:
        return current

    if db_path is not None:
        backup_database(db_path, current)

    for version in pending:
        with conn:
            if python_hooks and version in python_hooks:
                python_hooks[version](conn)

            for sql in all_migrations.get(version, []):
                for statement in _iter_sql_statements(sql):
                    conn.execute(statement)

            conn.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('version', ?)",
                (str(version),),
            )

    return pending[-1]
