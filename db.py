"""
db.py — Database initialisation and shared connection helpers.

────────────────────────────────────────────────────────────────────────────
DESIGN CHOICES (plain English)
────────────────────────────────────────────────────────────────────────────

1. WAL (Write-Ahead Logging) mode
   SQLite's default journal mode locks the entire file during a write.
   WAL separates writes into a side-file, so readers can continue reading
   while a write is in progress. This matters once Phase 6's Streamlit
   dashboard is reading the DB at the same time as ingest.py is writing.

2. check_same_thread=False
   SQLite connections are not thread-safe by default; this flag lets the
   same connection object be shared across threads (needed for Streamlit's
   threaded request model in Phase 6). We compensate by using context
   managers (`with conn:`) which automatically commit or roll back.

3. row_factory = sqlite3.Row
   Makes query results accessible as both column-name dicts and index
   tuples. Downstream code can do `row["title"]` instead of `row[2]`.

4. foreign_keys = ON
   SQLite ignores FK constraints by default (for legacy reasons). This
   PRAGMA enforces them so orphaned rows in entities/classifications/
   scores/clusters are impossible.

5. IF NOT EXISTS everywhere
   Every CREATE TABLE / CREATE INDEX is safe to run repeatedly. Calling
   init_db() at the top of every script costs microseconds and means no
   script assumes another script already ran first.

6. classifications table — no PRIMARY KEY on article_id
   Phase 3 uses multi-label classification (an article can belong to
   multiple threat categories simultaneously). Using an AUTOINCREMENT id
   as PK allows multiple rows per article_id. A UNIQUE constraint on
   (article_id, category) prevents re-inserting the same label.

────────────────────────────────────────────────────────────────────────────
ASSUMPTIONS
────────────────────────────────────────────────────────────────────────────
- DB file lives next to this script (DB_PATH). Override by passing db_path
  to get_connection() / init_db() if you want to use an in-memory DB for
  tests (pass ":memory:").
- All timestamps stored as ISO-8601 TEXT for portability. SQLite has no
  native DATETIME type — TEXT is the recommended approach.
"""

import sqlite3
from pathlib import Path

# Resolve DB path relative to this file so the scripts work regardless of
# which directory they're invoked from.
DB_PATH = Path(__file__).parent / "osint.db"

# ─────────────────────────────────────────────────────────────────────────────
# DDL — one big executescript so the entire schema is applied atomically.
# ─────────────────────────────────────────────────────────────────────────────
_SCHEMA_SQL = """
-- ── Core article store ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS articles (
    id          TEXT PRIMARY KEY,   -- SHA-256(url) — fixed-length, index-friendly
    url         TEXT UNIQUE NOT NULL,
    title       TEXT,
    summary     TEXT,               -- RSS <description> or <summary> field
    source      TEXT,               -- Human-readable feed name, e.g. "Reuters"
    published   TEXT,               -- ISO-8601 pub date from feed (may be NULL)
    ingested_at TEXT NOT NULL       -- ISO-8601 wall-clock time of this run
);

-- ── Phase 2: spaCy NER entities ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS entities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id  TEXT NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    text        TEXT NOT NULL,      -- Raw entity surface form (e.g. "Ukraine")
    label       TEXT NOT NULL,      -- GPE | ORG | NORP
    canonical   TEXT,               -- ISO 3166-1 alpha-2 (e.g. "UA") or BLOC:xxx
    start_char  INTEGER,            -- Character offset in title+summary text
    end_char    INTEGER
);

-- ── Phase 3: Zero-shot multi-label classification ─────────────────────────
-- Multi-label: one row per (article, category) pair above confidence threshold.
-- UNIQUE constraint prevents re-inserting the same label on re-runs.
CREATE TABLE IF NOT EXISTS classifications (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id  TEXT NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    category    TEXT NOT NULL,      -- One of 6 threat types
    confidence  REAL NOT NULL,      -- Model confidence [0, 1]
    UNIQUE (article_id, category)
);

CREATE TABLE IF NOT EXISTS classified_articles (
    article_id  TEXT PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
    classified_at TEXT NOT NULL
);

-- ── Phase 4: Gemini escalation scores ────────────────────────────────────
CREATE TABLE IF NOT EXISTS scores (
    article_id  TEXT PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
    score       REAL NOT NULL,      -- 0.0 – 10.0
    reasoning   TEXT                -- One-sentence rationale from Gemini
);

-- ── Phase 5: DBSCAN event clusters ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS clusters (
    article_id  TEXT PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
    cluster_id  INTEGER NOT NULL    -- -1 = DBSCAN noise (no cluster assigned)
);

-- ── Indices for common query patterns ────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_entities_article   ON entities(article_id);
CREATE INDEX IF NOT EXISTS idx_entities_label     ON entities(label);
CREATE INDEX IF NOT EXISTS idx_class_article      ON classifications(article_id);
CREATE INDEX IF NOT EXISTS idx_articles_source    ON articles(source);
CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published);
CREATE INDEX IF NOT EXISTS idx_scores_score       ON scores(score);
CREATE INDEX IF NOT EXISTS idx_clusters_cluster   ON clusters(cluster_id);
"""


def get_connection(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    """
    Open (or create) the SQLite database and return a configured connection.

    Parameters
    ----------
    db_path : str or Path
        Path to the .db file.  Pass ":memory:" for an in-memory test DB.

    Returns
    -------
    sqlite3.Connection
        WAL-mode connection with FK enforcement and Row factory enabled.
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # These PRAGMAs must be set on every new connection — they are not
    # persisted in the file (except WAL, which is stored in the file header).
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    # Tune cache for read-heavy dashboard workload (Phase 6).
    conn.execute("PRAGMA cache_size=-16000;")  # 16 MB page cache

    return conn


def init_db(db_path: str | Path = DB_PATH) -> None:
    """
    Create all tables and indices if they do not already exist.

    Safe to call on every script startup — IF NOT EXISTS makes it a no-op
    when the schema is already in place.
    """
    conn = get_connection(db_path)
    with conn:
        conn.executescript(_SCHEMA_SQL)
    conn.close()


def migrate_db(db_path: str | Path = DB_PATH) -> None:
    """
    Apply forward-only schema migrations to an existing database.

    Each migration checks whether its target state already exists before
    applying the change, so this is safe to call on every startup.
    We use ALTER TABLE ... ADD COLUMN (SQLite supports this since 3.1.3)
    rather than drop-and-recreate so that existing data is preserved.
    """
    import logging as _logging
    _log = _logging.getLogger("db.migrate")
    conn = get_connection(db_path)
    with conn:
        # Migration 1 (Phase 2): add `canonical` column to entities
        existing_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(entities)")
        }
        if "canonical" not in existing_cols:
            conn.execute("ALTER TABLE entities ADD COLUMN canonical TEXT;")
            _log.info("Migration: added 'canonical' column to entities table.")

        # Migration 2 (Phase 3): tracking table for classified articles (ensures 100% idempotency)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS classified_articles (
                article_id TEXT PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
                classified_at TEXT NOT NULL
            );
        """)
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helpers used by multiple pipeline scripts
# ─────────────────────────────────────────────────────────────────────────────

def article_exists(conn: sqlite3.Connection, article_id: str) -> bool:
    """Return True if an article with this SHA-256 id is already in the DB."""
    row = conn.execute(
        "SELECT 1 FROM articles WHERE id = ?", (article_id,)
    ).fetchone()
    return row is not None


def get_unprocessed_articles(
    conn: sqlite3.Connection,
    table: str,
    limit: int | None = None,
) -> list[sqlite3.Row]:
    """
    Return articles that have no row in `table` yet.

    Used by NER, classification, scoring, and clustering phases to pick up
    only the articles they haven't processed yet — enabling incremental runs.

    Parameters
    ----------
    table : str
        One of: "entities", "classifications", "scores", "clusters"
    limit : int or None
        Cap the result set (useful for testing).
    """
    # Validate table name to prevent SQL injection (no user input, but good
    # practice for a project that might be extended).
    allowed = {"entities", "classifications", "scores", "clusters", "classified_articles"}
    if table not in allowed:
        raise ValueError(f"table must be one of {allowed}, got {table!r}")

    sql = f"""
        SELECT a.*
        FROM   articles a
        LEFT   JOIN {table} t ON t.article_id = a.id
        WHERE  t.article_id IS NULL
        ORDER  BY a.published DESC
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    return conn.execute(sql).fetchall()


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test — run `python db.py` to verify the DB initialises cleanly.
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    db_path = sys.argv[1] if len(sys.argv) > 1 else DB_PATH
    print(f"Initialising database at: {db_path}")
    init_db(db_path)

    conn = get_connection(db_path)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
    ).fetchall()
    print("Tables created:")
    for t in tables:
        print(f"  [OK]  {t['name']}")

    indices = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name;"
    ).fetchall()
    print(f"Indices created: {len(indices)}")
    conn.close()
    print("db.py self-test passed.")
