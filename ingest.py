"""
ingest.py — Phase 1: RSS feed ingestion into SQLite.

────────────────────────────────────────────────────────────────────────────
DESIGN CHOICES (plain English)
────────────────────────────────────────────────────────────────────────────

1. SHA-256 URL hash as primary key
   The article's canonical URL is its natural unique identifier. Rather than
   storing the full URL as a VARCHAR primary key (slow B-tree comparisons,
   variable length), we hash it with SHA-256 to produce a fixed 64-character
   hex string. The same URL will always produce the same hash — so duplicate
   detection is purely hash comparison, with zero extra queries.

2. INSERT OR IGNORE for deduplication
   SQLite's IGNORE conflict resolution silently skips the INSERT if the
   primary key already exists. This is faster than a SELECT-then-INSERT
   pattern and makes the script idempotent: run it 100 times, the DB stays
   consistent.

3. feedparser for RSS/Atom parsing
   feedparser is the de-facto Python RSS library. It handles:
   - Both RSS 2.0 and Atom 1.0 transparently
   - Malformed XML (it uses a lenient parser)
   - Character encoding detection
   - Relative URL resolution
   We check `feed.bozo` (feedparser's "this feed had parse errors" flag) and
   log a warning, but don't abort — many real-world feeds are slightly broken
   yet still yield usable entries.

4. Title + summary as article text
   Full-text scraping (following links, parsing HTML) is out of scope for
   Phase 1. RSS `<description>` / `<summary>` fields typically contain
   2-5 sentences -- sufficient for NER (Phase 2) and zero-shot classification
   (Phase 3). The raw URL is stored so scraping can be retrofitted later.

5. Per-feed isolation with try/except
   A single feed being down (DNS failure, 404, timeout) should not abort
   the whole ingestion run. Each feed is wrapped in its own try/except; a
   failed feed is logged and counted in stats, then skipped.

6. Published date normalisation
   feedparser returns `entry.published_parsed` as a UTC time.struct_time.
   We convert it to an ISO-8601 string ("2024-03-15T12:30:00Z"). If the
   field is absent (some feeds omit it), we fall back to None so the column
   stays NULL rather than storing a garbage value.

7. Socket timeout workaround
   feedparser 6.x removed the `timeout` kwarg from parse(). We temporarily
   set the global socket default timeout to 15 seconds before each parse()
   call and restore it afterward.

8. metrics.json append/merge
   The file is treated as a list under each top-level phase key.
   Each run appends a new timestamped object so you can track ingestion
   health over time without overwriting past runs.

────────────────────────────────────────────────────────────────────────────
ASSUMPTIONS
────────────────────────────────────────────────────────────────────────────
- Python 3.11+ (uses `str | None` union type hints).
- Network access to RSS feed URLs is available.
- Some feeds (e.g. SCMP) may geo-restrict or require cookies; they will be
  logged as errors but won't crash the run.
- AP News RSS feed URL may require verification -- AP removed public RSS in
  2021 but has since restored some endpoints.

USAGE
------
    python ingest.py --dry-run  # parse feeds but don't write to DB
    python ingest.py --limit 5  # process only the first 5 entries per feed
"""

import argparse
import calendar
import hashlib
import io
import json
import logging
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import feedparser
from tabulate import tabulate

from db import get_connection, init_db

# ─────────────────────────────────────────────────────────────────────────────
# UTF-8 stdout fix for Windows PowerShell (default encoding is cp1252 which
# cannot render Unicode box-drawing characters used by tabulate).
# reconfigure() is available from Python 3.7+ and is the clean way to do this.
# ─────────────────────────────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# Logging — structured, with timestamps so log lines are grep-able.
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ingest.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("ingest")

# ─────────────────────────────────────────────────────────────────────────────
# Feed definitions — exactly the 10 sources from the implementation plan.
# Format: (human_readable_name, rss_url)
# ─────────────────────────────────────────────────────────────────────────────
FEEDS: list[tuple[str, str]] = [
    # ── Global wire services ──────────────────────────────────────────────
    (
        "France 24 English",
        # France 24 International -- strong wire coverage, clean RSS.
        "https://www.france24.com/en/rss",
    ),
    (
        "NPR World",
        # NPR's world section — reliable, well-formed RSS.
        # Replaces AP News which has CDN/DNS resolution issues.
        "https://feeds.npr.org/1004/rss.xml",
    ),
    (
        "BBC World",
        "http://feeds.bbci.co.uk/news/world/rss.xml",
    ),
    # ── Regional outlets ──────────────────────────────────────────────────
    (
        "Al Jazeera",
        "https://www.aljazeera.com/xml/rss/all.xml",
    ),
    (
        "The Hindu",
        "https://www.thehindu.com/news/international/feeder/default.rss",
    ),
    (
        "South China Morning Post",
        "https://www.scmp.com/rss/91/feed",
    ),
    (
        "DW World",
        # Deutsche Welle (DW) — German international broadcaster.
        # Strong Eastern Europe / Ukraine coverage; clean RSS feed.
        # Replaces Kyiv Independent which has persistent XML malformation.
        "https://rss.dw.com/atom/rss-en-world",
    ),
    (
        "The Guardian World",
        "https://www.theguardian.com/world/rss",
    ),
    # ── Security-focused outlets ──────────────────────────────────────────
    (
        "Krebs on Security",
        "https://krebsonsecurity.com/feed/",
    ),
    (
        "The Hacker News",
        "https://feeds.feedburner.com/TheHackersNews",
    ),
]

# Metrics JSON path — sits alongside the DB in the project root.
METRICS_PATH = Path(__file__).parent / "metrics.json"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def url_to_id(url: str) -> str:
    """Return the SHA-256 hex digest of a canonical URL.

    We strip leading/trailing whitespace and lower-case the scheme+host
    (RFC 3986 §6.2.2.1) so that http://Example.com/X and http://example.com/X
    hash to the same value.  The path is left as-is (case-sensitive).
    """
    try:
        from urllib.parse import urlparse, urlunparse
        parts = urlparse(url.strip())
        canonical = urlunparse(parts._replace(
            scheme=parts.scheme.lower(),
            netloc=parts.netloc.lower(),
        ))
    except Exception:
        canonical = url.strip()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_published(entry: Any) -> str | None:
    """Convert feedparser's published_parsed (struct_time, UTC) to ISO-8601.

    Returns None if the field is absent or unparseable — better to store
    NULL than a wrong timestamp.
    """
    pt = getattr(entry, "published_parsed", None)
    if pt is None:
        # Try updated_parsed as a fallback (Atom feeds use <updated>)
        pt = getattr(entry, "updated_parsed", None)
    if pt is None:
        return None
    try:
        # calendar.timegm converts a UTC struct_time to a Unix timestamp.
        dt = datetime.fromtimestamp(calendar.timegm(pt), tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None


def clean_text(raw: str | None) -> str | None:
    """Strip HTML tags and normalise whitespace from feed summary text.

    feedparser partially sanitises HTML, but some feeds still include
    <p>, <a>, <br> tags in summaries. A lightweight regex strip is enough
    for Phase 1; a full parser is not worth the overhead here.
    """
    if not raw:
        return None
    import re
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", raw)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


# ─────────────────────────────────────────────────────────────────────────────
# Core ingestion logic
# ─────────────────────────────────────────────────────────────────────────────

def fetch_feed(
    source_name: str,
    url: str,
    limit: int | None = None,
) -> tuple[list[dict], str | None]:
    """Fetch and parse a single RSS/Atom feed.

    Parameters
    ----------
    source_name : str
        Human-readable feed name (stored in the DB `source` column).
    url : str
        RSS/Atom feed URL.
    limit : int or None
        If set, cap entries returned (useful for smoke-testing).

    Returns
    -------
    entries : list[dict]
        List of normalised article dicts ready for DB insertion.
    error : str or None
        Error message if the feed failed, else None.
    """
    log.info("Fetching: %s — %s", source_name, url)

    try:
        # feedparser 6.x removed the `timeout` kwarg from parse().
        # The standard workaround is to set the global socket default timeout
        # before calling parse(), then restore it.  This is thread-safe enough
        # for our single-threaded ingestion loop.
        prev_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(15)
        try:
            feed = feedparser.parse(url, agent="OSINTAggregator/1.0")
        finally:
            socket.setdefaulttimeout(prev_timeout)
    except Exception as exc:
        return [], f"Connection error: {exc}"

    # bozo=True means feedparser encountered parse errors.
    # Distinguish two cases:
    #   1. bozo=True AND entries is empty  ->  genuine failure (network error,
    #      totally unreadable feed).  Return an error.
    #   2. bozo=True AND entries is non-empty  ->  feed is slightly malformed
    #      but still yielded usable data.  Log a warning and continue.
    # Note: feedparser wraps network errors in bozo_exception too, so we check
    # the exception type to give a better error message.
    if feed.bozo and not feed.entries:
        exc = feed.bozo_exception
        exc_str = str(exc) if exc else "unknown parse error"
        return [], f"Feed parse error: {exc_str}"

    if feed.bozo:
        log.warning(
            "%s: feed has parse warnings (%s) but returned %d entries — continuing.",
            source_name,
            feed.bozo_exception,
            len(feed.entries),
        )

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entries_raw = feed.entries[:limit] if limit else feed.entries
    results: list[dict] = []

    for entry in entries_raw:
        # Get the canonical URL — prefer `link`, fall back to `id` (Atom).
        url_raw = getattr(entry, "link", None) or getattr(entry, "id", None)
        if not url_raw:
            log.debug("%s: entry has no URL, skipping.", source_name)
            continue

        # Summary: prefer `summary`, fall back to `description`.
        summary_raw = (
            getattr(entry, "summary", None)
            or getattr(entry, "description", None)
        )

        # C-6 fix: record the actual moment each article is processed, not a
        # single timestamp shared across all entries from the same feed fetch.
        ingested_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        results.append({
            "id":          url_to_id(url_raw),
            "url":         url_raw.strip(),
            "title":       clean_text(getattr(entry, "title", None)),
            "summary":     clean_text(summary_raw),
            "source":      source_name,
            "published":   parse_published(entry),
            "ingested_at": ingested_ts,
        })

    return results, None


def ingest_to_db(
    conn,
    articles: list[dict],
    dry_run: bool = False,
) -> tuple[int, int]:
    """Insert articles into the DB, skipping duplicates.

    Returns
    -------
    (inserted, skipped) : tuple[int, int]
        Count of rows actually inserted vs. silently skipped (duplicates).
    """
    if dry_run or not articles:
        return 0, 0

    inserted = 0
    skipped = 0

    # Use a single transaction per batch — dramatically faster than one
    # transaction per row for bulk inserts.
    with conn:
        for art in articles:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO articles
                    (id, url, title, summary, source, published, ingested_at)
                VALUES
                    (:id, :url, :title, :summary, :source, :published, :ingested_at)
                """,
                art,
            )
            if cursor.rowcount == 1:
                inserted += 1
            else:
                skipped += 1

    return inserted, skipped


# ─────────────────────────────────────────────────────────────────────────────
# Metrics persistence
# ─────────────────────────────────────────────────────────────────────────────

def load_metrics() -> dict:
    """Load existing metrics.json, or return an empty dict."""
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("metrics.json is corrupt — starting fresh.")
        except (OSError, PermissionError) as e:
            # E-4 fix: handle file-system errors gracefully
            log.warning("Cannot read metrics.json (%s) — starting fresh.", e)
    return {}


def save_metrics(metrics: dict) -> None:
    """Write metrics dict back to metrics.json (pretty-printed)."""
    try:
        METRICS_PATH.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except (OSError, PermissionError) as e:
        log.warning("Cannot write metrics.json (%s) — run data not persisted.", e)


def append_ingest_run(
    metrics: dict,
    run_stats: list[dict],
    total_new: int,
    total_skipped: int,
    total_attempted: int,
    total_errors: int,
    dry_run: bool,
) -> dict:
    """Append this run's stats to the metrics["ingest"] list."""
    run_record = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dry_run": dry_run,
        "total_attempted": total_attempted,
        "total_new": total_new,
        "total_skipped": total_skipped,
        "total_errors": total_errors,
        "dedup_rate_pct": (
            round(100 * total_skipped / total_attempted, 1)
            if total_attempted else 0
        ),
        "per_source": run_stats,
    }
    metrics.setdefault("ingest", []).append(run_record)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False, limit: int | None = None) -> None:
    # 1. Ensure schema exists
    init_db()
    conn = get_connection()

    run_stats: list[dict] = []
    total_new = total_skipped = total_attempted = total_errors = 0

    for source_name, feed_url in FEEDS:
        t_start = time.perf_counter()
        articles, error = fetch_feed(source_name, feed_url, limit=limit)
        fetch_elapsed = time.perf_counter() - t_start

        if error:
            log.error("%s: %s", source_name, error)
            run_stats.append({
                "source": source_name,
                "fetched": 0,
                "inserted": 0,
                "skipped": 0,
                "status": f"ERROR: {error}",
            })
            total_errors += 1
            continue

        fetched = len(articles)
        inserted, skipped = ingest_to_db(conn, articles, dry_run=dry_run)
        total_attempted += fetched
        total_new      += inserted
        total_skipped  += skipped

        status = "dry-run" if dry_run else "ok"
        log.info(
            "%-30s fetched=%3d  inserted=%3d  skipped=%3d  (%.2fs)",
            source_name, fetched, inserted, skipped, fetch_elapsed,
        )
        run_stats.append({
            "source":   source_name,
            "fetched":  fetched,
            "inserted": inserted,
            "skipped":  skipped,
            "status":   status,
        })

    conn.close()

    # ── Summary table ─────────────────────────────────────────────────────
    table_rows = [
        [
            s["source"],
            s["fetched"],
            s["inserted"],
            s["skipped"],
            s["status"],
        ]
        for s in run_stats
    ]
    print("\n" + "=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    print(tabulate(
        table_rows,
        headers=["Source", "Fetched", "Inserted", "Skipped", "Status"],
        tablefmt="rounded_outline",
    ))
    dedup_rate = (
        round(100 * total_skipped / total_attempted, 1)
        if total_attempted else 0
    )
    print(f"\nTotal attempted : {total_attempted}")
    print(f"Total new       : {total_new}")
    print(f"Total skipped   : {total_skipped}  ({dedup_rate}% duplicate rate)")
    print(f"Feed errors     : {total_errors}")
    if dry_run:
        print("\n[DRY RUN] — no data was written to the database.")
    print("=" * 70 + "\n")

    # ── Persist to metrics.json ───────────────────────────────────────────
    if not dry_run:
        metrics = load_metrics()
        metrics = append_ingest_run(
            metrics, run_stats,
            total_new, total_skipped, total_attempted, total_errors,
            dry_run=False,
        )
        save_metrics(metrics)
        print(f"Run stats appended to: {METRICS_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest RSS feeds into the OSINT SQLite database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py                  # Full ingest run
  python ingest.py --dry-run        # Parse feeds, print stats, skip DB writes
  python ingest.py --limit 5        # Only process 5 entries per feed (testing)
  python ingest.py --dry-run --limit 3
""",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and parse feeds but do NOT write to the database.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N entries per feed (useful for smoke-testing).",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run, limit=args.limit)
