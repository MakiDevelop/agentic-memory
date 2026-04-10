-- FTS5 standalone table with CJK tokenization support
-- Python hook seeds temp._v002_fts_seed with tokenized content before this runs.
CREATE TEMP TABLE IF NOT EXISTS _v002_fts_seed (
    rowid INTEGER PRIMARY KEY,
    content TEXT NOT NULL
);
DROP TRIGGER IF EXISTS memories_ai;
DROP TRIGGER IF EXISTS memories_ad;
DROP TRIGGER IF EXISTS memories_au;
DROP TABLE IF EXISTS memories_fts;
CREATE VIRTUAL TABLE memories_fts USING fts5(content);
INSERT INTO memories_fts(rowid, content)
SELECT rowid, content FROM temp._v002_fts_seed;
DROP TABLE IF EXISTS temp._v002_fts_seed;
