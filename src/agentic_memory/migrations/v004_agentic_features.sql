-- Schema v4: kind, importance, TTL, source_hash, retrieval_logs
-- ALTER TABLE ADD COLUMN and backfill are handled by the Python hook.
CREATE TABLE IF NOT EXISTS retrieval_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    returned_ids TEXT NOT NULL,
    result_count INTEGER NOT NULL,
    latency_ms REAL DEFAULT 0.0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_logs_created_at ON retrieval_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(source_hash);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
