-- Schema v5: adoption tracking
CREATE TABLE IF NOT EXISTS adoption_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    query TEXT DEFAULT '',
    agent_name TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_adoption_memory ON adoption_logs(memory_id);
