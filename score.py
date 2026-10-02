"""
score.py — Gemini Threat Escalation Scoring Pipeline (Phase 4).

────────────────────────────────────────────────────────────────────────────
DESIGN CHOICES (plain English)
────────────────────────────────────────────────────────────────────────────

1. Model: gemini-1.5-flash ONLY — no auto-fallback to other models.
   Free tier limits: 1,500 req/day, 15 req/min (we self-limit to 14 req/min).
   gemini-2.5-flash is intentionally excluded: it shares a much tighter 20 req/day quota.
   Fast, cost-effective multimodal LLM tailored for low-latency scoring.
   API key is loaded securely from .env via python-dotenv. If the key is
   missing or fails, a mock scorer provides random fallback ratings (5-7)
   so the downstream pipeline never breaks.

2. Scoring Rubric & Anchoring:
   0-2: Routine/background news
   3-4: Elevated concern, no immediate violence
   5-6: Active conflict or significant disruption
   7-8: Major escalation, casualties reported
   9-10: Catastrophic/regional war-level event

3. Why Few-Shot Examples Matter:
   LLMs can exhibit arbitrary calibration on numeric scales. Without examples,
   one prompt might treat minor diplomatic friction as a 7, while another treats
   artillery shelling as a 6. By providing 3 diverse anchor examples in the prompt,
   the model consistently aligns its outputs to our exact severity thresholds.

4. Defensive JSON Extraction:
   The model is instructed to reply exclusively in JSON format:
   {"score": int, "reasoning": str}. We strip markdown code fences, parse the JSON,
   clamp scores to [0.0, 10.0], and fall back to score 5.0 if any parsing error occurs.

5. Rate Limiting & 429 Handling:
   A token-bucket enforces a hard ceiling of 14 requests/minute (buffer under the 15/min
   free-tier limit). On a 429 RESOURCE_EXHAUSTED response, the scorer waits 60 seconds and
   retries exactly once. If still 429, the article is logged as 'quota_exceeded' and skipped
   (not counted as failed). The run summary reports skipped articles separately.

6. Idempotence & Caching:
   Articles with existing records in the `scores` table are skipped by default.
   Pass `--rerun` to clear and re-score the entire database.
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Windows UTF-8 stdout fix ──────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from tabulate import tabulate

from db import (
    DB_PATH,
    get_connection,
    get_unprocessed_articles,
    init_db,
    migrate_db,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("score")

# ── Load environment variables ────────────────────────────────────────────────
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

VALIDATION_DIR = Path(__file__).parent / "validation"
VALIDATION_FILE = VALIDATION_DIR / "score_validation.jsonl"
METRICS_PATH = Path(__file__).parent / "metrics.json"

# ── Prompt Templates & Rubric ─────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Open Source Intelligence (OSINT) analyst evaluating global geopolitical and security threats.
Analyze the following news article and score its threat escalation level on a strict 0 to 10 scale based on this exact rubric:

SCORING RUBRIC:
0-2: Routine/background news (diplomatic courtesies, peaceful trade talks, cultural events, non-threat business or sports)
3-4: Elevated concern, no immediate violence (harsh diplomatic rhetoric, new sanctions, cyber espionage, low-level protests without injuries)
5-6: Active conflict or significant disruption (artillery exchanges, active cyberattacks disabling critical infrastructure, violent rioting, border skirmishes)
7-8: Major escalation, casualties reported (missile strikes causing deaths, assassinations, declaration of emergency, widespread civilian casualties)
9-10: Catastrophic/regional war-level event (full-scale multi-nation invasion, nuclear threat/use, collapse of government in major conflict, WW3-level escalation)

FEW-SHOT ANCHOR EXAMPLES:
Article: "South Korea delegation complains about bed sizes in athletes' village at Asian Games in Nagoya."
Output: {{"score": 1, "reasoning": "Routine sports logistical complaint with zero geopolitical or security escalation."}}

Article: "A suspected state-sponsored cyber group leveraged a zero-day flaw in Gitea to breach 13 organizations across six countries, stealing internal datasets."
Output: {{"score": 5, "reasoning": "Active, multi-national cyber campaign causing significant operational breach, but without kinetic casualties."}}

Article: "A Russian drone struck a passenger bus in southern Ukraine early Wednesday, killing five civilians and wounding seven others."
Output: {{"score": 8, "reasoning": "Direct lethal strike causing multiple civilian casualties in an active war zone, marking severe kinetic escalation."}}

INPUT ARTICLE TO EVALUATE:
Title: {title}
Text: {text}

Respond ONLY in valid, parseable JSON with this exact schema:
{{"score": <integer from 0 to 10>, "reasoning": "<one concise explanatory sentence>"}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Model Initialization Helper
# ─────────────────────────────────────────────────────────────────────────────

# Hard-pinned model — do NOT change to gemini-2.5-flash (only 20 req/day free tier)
GEMINI_MODEL = "gemini-1.5-flash"

# Rate-limit constants (gemini-1.5-flash free tier: 15 req/min, 1500 req/day)
MAX_REQUESTS_PER_MINUTE = 14   # 1 request buffer under the 15/min hard limit
QUOTA_429_WAIT_SECS     = 60   # How long to wait after a 429 before one retry


def get_gemini_client():
    """
    Initialize and return a (client, model_name) bundle for gemini-1.5-flash.

    Pinned to gemini-1.5-flash only — no auto-fallback to other model versions.
    Falls back to mock mode if API key is missing or the API rejects the key.
    """
    if not GEMINI_API_KEY:
        log.warning("No GEMINI_API_KEY found in .env; running in MOCK mode.")
        return None

    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        log.info("Gemini client initialised. Model pinned to: %s", GEMINI_MODEL)
        return (client, GEMINI_MODEL)
    except ImportError:
        log.warning("google.genai package not available; running in MOCK mode.")
        return None
    except Exception as exc:
        log.warning("Failed to configure Gemini API (%s); running in MOCK mode.", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Parsing & Scoring Logic
# ─────────────────────────────────────────────────────────────────────────────

def clean_json_response(raw_text: str) -> dict:
    """
    Defensively extract and parse JSON object from LLM response.
    """
    text = raw_text.strip()
    # Strip markdown fences if present
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


# Sentinel returned when the daily/per-minute quota is exhausted
_QUOTA_EXCEEDED = object()


def _is_429(err: Exception) -> bool:
    """Return True if the exception is a Gemini 429 RESOURCE_EXHAUSTED error."""
    msg = str(err).lower()
    return "429" in msg or "resource_exhausted" in msg or "quota" in msg


def score_article_gemini(
    client_bundle,
    title: str,
    text: str,
) -> tuple[float, str] | object:
    """
    Send one article to Gemini for escalation scoring.

    Returns
    -------
    (score: float, reasoning: str)  on success
    _QUOTA_EXCEEDED sentinel         when 429 persists after the 60-second retry
    (5.0, fallback_msg)             on any other API / parsing error
    """
    if client_bundle is None:
        # Mock mode
        score = float(random.randint(5, 7))
        return score, "Mock escalation score: elevated operational concern identified in region."

    client, model_name = client_bundle
    prompt = SYSTEM_PROMPT.format(title=title or "", text=text or "")

    for attempt in (1, 2):   # attempt 1 = first try; attempt 2 = after 60s 429 wait
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            if not response or not response.text:
                raise ValueError("Empty response from Gemini API")

            data  = clean_json_response(response.text)
            score = float(data.get("score", 5.0))
            score = max(0.0, min(10.0, score))          # clamp to [0, 10]
            reasoning = str(data.get("reasoning",
                                     "Escalation evaluated based on threat severity."))
            return score, reasoning

        except Exception as err:
            if _is_429(err):
                if attempt == 1:
                    log.warning(
                        "[model=%s] 429 RESOURCE_EXHAUSTED — waiting %ds then retrying once.",
                        model_name, QUOTA_429_WAIT_SECS,
                    )
                    time.sleep(QUOTA_429_WAIT_SECS)
                    continue          # go to attempt 2
                else:
                    # Still 429 after the wait — skip, do not waste more quota
                    log.warning(
                        "[model=%s] 429 persists after retry — marking article as quota_exceeded.",
                        model_name,
                    )
                    return _QUOTA_EXCEEDED
            else:
                # Non-429 error (JSON parse fail, network blip, etc.) — fallback immediately
                log.warning("[model=%s] Scoring error: %s — using fallback score 5.0.",
                            model_name, err)
                return 5.0, "Fallback score: API/parsing error."


# ─────────────────────────────────────────────────────────────────────────────
# DB Operations
# ─────────────────────────────────────────────────────────────────────────────

def article_to_text(article) -> str:
    """Combine title and summary into single input."""
    parts = []
    if article["title"]:
        parts.append(article["title"].strip())
    if article["summary"]:
        parts.append(article["summary"].strip())
    return " ".join(parts).strip()


def insert_scores(conn, records: list[tuple[str, float, str]]) -> int:
    """
    Insert scores into SQLite scores table.
    """
    if not records:
        return 0
    sql = """
        INSERT OR REPLACE INTO scores (article_id, score, reasoning)
        VALUES (?, ?, ?)
    """
    with conn:
        cursor = conn.executemany(sql, records)
        return cursor.rowcount


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Logger
# ─────────────────────────────────────────────────────────────────────────────

def append_metrics(run_data: dict) -> None:
    """Append scoring metrics to metrics.json."""
    metrics = {}
    if METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("Existing metrics.json corrupt; starting fresh.")
            metrics = {}
        except (OSError, PermissionError) as e:
            # E-4 fix: catch file-system errors, continue without metrics
            log.warning("Could not read metrics.json (%s); starting fresh.", e)
            metrics = {}

    if "score" not in metrics:
        metrics["score"] = []

    metrics["score"].append(run_data)
    try:
        METRICS_PATH.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("Updated metrics recorded at: %s", METRICS_PATH)
    except (OSError, PermissionError) as e:
        log.warning("Could not write metrics.json (%s) — run data not persisted.", e)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_scoring(
    limit: int | None = None,
    rerun: bool = False,
    sleep_between_calls: float = 1.0,
) -> dict:
    """
    Score all unscored articles using gemini-1.5-flash with:
      - Hard 14 req/min rate limit (token-bucket)
      - 429 -> 60s wait -> one retry -> skip as quota_exceeded
      - Idempotent: skips articles already in the scores table
    """
    init_db()
    migrate_db()
    conn = get_connection()

    if rerun:
        log.warning("--rerun specified: clearing all existing scores.")
        with conn:
            conn.execute("DELETE FROM scores")

    articles = get_unprocessed_articles(conn, "scores", limit=limit)
    total = len(articles)
    log.info("Articles to score: %d (model=%s, rate_limit=%d req/min)",
             total, GEMINI_MODEL, MAX_REQUESTS_PER_MINUTE)

    if total == 0:
        log.info("Nothing to do -- all articles already have escalation scores.")
        conn.close()
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "articles_processed": 0,
            "scores_stored": 0,
            "quota_exceeded_skipped": 0,
        }

    client_bundle = get_gemini_client()
    mode_name = f"Gemini API ({GEMINI_MODEL})" if client_bundle else "Mock Scorer"
    log.info("Scoring engine active: %s", mode_name)

    scores_collected  = []
    quota_skipped     = []   # article IDs skipped due to persistent 429
    bucket_counts = {
        "0-2 (Routine)": 0,
        "3-4 (Elevated)": 0,
        "5-6 (Active Conflict)": 0,
        "7-8 (Major Escalation)": 0,
        "9-10 (Catastrophic)": 0,
    }

    # ── Token-bucket rate limiter ──────────────────────────────────────────
    # Tracks timestamps of the last MAX_REQUESTS_PER_MINUTE calls.
    # Before each API call we ensure the oldest call in the window is
    # > 60 seconds ago; if not, we sleep the difference.
    _request_times: list[float] = []

    def _rate_limit_wait() -> None:
        """Block until it is safe to make another API request."""
        now = time.monotonic()
        window_start = now - 60.0
        # Drop timestamps outside the 60-second window
        while _request_times and _request_times[0] < window_start:
            _request_times.pop(0)
        if len(_request_times) >= MAX_REQUESTS_PER_MINUTE:
            # Oldest request in the window; sleep until it falls out
            sleep_for = 60.0 - (now - _request_times[0]) + 0.1  # 100ms buffer
            if sleep_for > 0:
                log.info("Rate limit: %d req in last 60s — waiting %.1fs",
                         len(_request_times), sleep_for)
                time.sleep(sleep_for)
        _request_times.append(time.monotonic())
    # ──────────────────────────────────────────────────────────────────────

    start_time = time.perf_counter()

    for idx, art in enumerate(articles, start=1):
        art_id = art["id"]
        title  = art["title"] or ""
        text   = article_to_text(art)

        # Enforce rate limit before every real API call
        if client_bundle is not None:
            _rate_limit_wait()

        result = score_article_gemini(client_bundle, title, text)

        if result is _QUOTA_EXCEEDED:
            log.warning("[quota_exceeded] Skipping article_id=%s", art_id[:16])
            quota_skipped.append(art_id)
            # Do NOT insert into DB — article stays unscored for next run
            continue

        score, reasoning = result
        insert_scores(conn, [(art_id, score, reasoning)])
        scores_collected.append(score)

        # Bucket categorisation
        if score <= 2.0:
            bucket_counts["0-2 (Routine)"] += 1
        elif score <= 4.0:
            bucket_counts["3-4 (Elevated)"] += 1
        elif score <= 6.0:
            bucket_counts["5-6 (Active Conflict)"] += 1
        elif score <= 8.0:
            bucket_counts["7-8 (Major Escalation)"] += 1
        else:
            bucket_counts["9-10 (Catastrophic)"] += 1

        if idx % 10 == 0 or idx == total:
            log.info("Progress: %d/%d scored | quota_skipped=%d | latest=%.1f",
                     idx, total, len(quota_skipped), score)

        # Additional inter-call sleep (on top of rate limiter) when requested
        if sleep_between_calls > 0 and client_bundle is not None and idx < total:
            time.sleep(sleep_between_calls)

    elapsed   = round(time.perf_counter() - start_time, 2)
    n_scored  = len(scores_collected)
    avg_score = round(sum(scores_collected) / n_scored, 2) if n_scored else 0.0
    conn.close()

    log.info("Scoring done in %.2fs. Scored=%d, quota_skipped=%d, avg=%.2f",
             elapsed, n_scored, len(quota_skipped), avg_score)

    summary_rows = [
        [bucket, count, f"{(count / n_scored * 100):.1f}%" if n_scored else "0%"]
        for bucket, count in bucket_counts.items()
    ]
    print("\n" + "=" * 65)
    print("GEMINI ESCALATION SCORING SUMMARY")
    print("=" * 65)
    print(
        tabulate(
            summary_rows,
            headers=["Escalation Level", "Articles", "% of Scored"],
            tablefmt="rounded_outline",
        )
    )
    print(f"Model:                  {GEMINI_MODEL}")
    print(f"Engine:                 {mode_name}")
    print(f"Total to score:         {total}")
    print(f"Successfully scored:    {n_scored}")
    print(f"Quota-exceeded skipped: {len(quota_skipped)}")
    print(f"Average score:          {avg_score} / 10.0")
    print(f"Elapsed time:           {elapsed:.2f}s")
    if quota_skipped:
        print(f"Skipped IDs (first 5):  {[s[:12] for s in quota_skipped[:5]]}")
    print("=" * 65 + "\n")

    run_stats = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": GEMINI_MODEL,
        "engine": mode_name,
        "articles_processed": total,
        "articles_scored": n_scored,
        "quota_exceeded_skipped": len(quota_skipped),
        "average_score": avg_score,
        "score_distribution": bucket_counts,
        "elapsed_seconds": elapsed,
    }
    append_metrics(run_stats)
    return run_stats


# ─────────────────────────────────────────────────────────────────────────────
# Validation Set Generator (25 articles)
# ─────────────────────────────────────────────────────────────────────────────

def generate_validation_file(val_size: int = 25) -> Path:
    """
    Generate pre-filled validation JSONL for Phase 4 human evaluation.
    Outputs 25 articles with Gemini scores and a blank 'manual_score'
    field for manual ground-truth annotation.
    """
    init_db()
    conn = get_connection()

    cursor = conn.execute("""
        SELECT id, source, title, summary, published
        FROM articles
        ORDER BY RANDOM()
        LIMIT ?
    """, (val_size,))
    articles = cursor.fetchall()
    conn.close()

    if not articles:
        log.error("No articles found in database. Run ingest.py first!")
        sys.exit(1)

    model = get_gemini_client()
    log.info("Generating pre-filled scoring validation set for %d articles ...", len(articles))

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for i, art in enumerate(articles, start=1):
        title = art["title"] or ""
        text = article_to_text(art)

        # C-3 fix: guard against _QUOTA_EXCEEDED sentinel before unpacking
        result = score_article_gemini(model, title, text)
        if result is _QUOTA_EXCEEDED:
            log.warning(
                "Validation sample %d/%d: quota exceeded — using fallback score 5.0.",
                i, len(articles),
            )
            score, reasoning = 5.0, "Quota exceeded during validation — fallback score assigned."
        else:
            score, reasoning = result

        records.append({
            "article_id": art["id"],
            "source": art["source"],
            "published": art["published"],
            "text": text,
            "gemini_score": round(score, 1),
            "gemini_reasoning": reasoning,
            "manual_score": None,  # User fills this in with their own 0-10 rating
        })
        log.info("Validation sample %d/%d scored: %.1f", i, len(articles), score)
        time.sleep(1.0)

    with open(VALIDATION_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info("Scoring validation template written to: %s", VALIDATION_FILE)
    print("\n" + "=" * 65)
    print("PRE-FILLED SCORING VALIDATION TEMPLATE GENERATED")
    print("=" * 65)
    print(f"File:     {VALIDATION_FILE}")
    print(f"Articles: {len(records)}")
    print("\nNext step: Open validation/score_validation.jsonl and add your manual")
    print("scores (0-10) in 'manual_score', then run:")
    print("  python evaluate.py --phase score")
    print("=" * 65 + "\n")
    return VALIDATION_FILE


# ─────────────────────────────────────────────────────────────────────────────
# Consistency Check (10 articles evaluated 3 times each)
# ─────────────────────────────────────────────────────────────────────────────

def run_consistency_check(sample_size: int = 10, runs: int = 3) -> dict:
    """
    Evaluate Gemini scoring consistency by scoring 10 articles 3 times each,
    measuring the standard deviation per article.
    """
    init_db()
    conn = get_connection()
    cursor = conn.execute("""
        SELECT id, source, title, summary
        FROM articles
        ORDER BY RANDOM()
        LIMIT ?
    """, (sample_size,))
    articles = cursor.fetchall()
    conn.close()

    model = get_gemini_client()
    log.info("Running consistency check: %d articles x %d runs ...", sample_size, runs)

    results = []
    table_rows = []

    for idx, art in enumerate(articles, start=1):
        title = art["title"] or ""
        text = article_to_text(art)
        article_scores = []

        for r in range(runs):
            # C-7 fix: guard against _QUOTA_EXCEEDED sentinel before unpacking
            result = score_article_gemini(model, title, text)
            if result is _QUOTA_EXCEEDED:
                log.warning("Consistency check run %d: quota exceeded — using fallback 5.0.", r + 1)
                score = 5.0
            else:
                score, _ = result
            article_scores.append(score)
            time.sleep(1.0)

        # Calculate mean and standard deviation
        mean_s = sum(article_scores) / len(article_scores)
        variance = sum((s - mean_s) ** 2 for s in article_scores) / (len(article_scores) - 1) if len(article_scores) > 1 else 0.0
        std_dev = variance ** 0.5

        results.append({
            "article_id": art["id"],
            "title": title[:50] + "...",
            "scores": article_scores,
            "mean": round(mean_s, 2),
            "std_dev": round(std_dev, 3),
        })

        table_rows.append([
            f"#{idx}",
            title[:40] + "...",
            ", ".join(str(s) for s in article_scores),
            f"{mean_s:.2f}",
            f"{std_dev:.3f}",
        ])

    avg_std = sum(r["std_dev"] for r in results) / len(results) if results else 0.0

    print("\n" + "=" * 65)
    print("GEMINI SCORING CONSISTENCY REPORT (3 Runs per Article)")
    print("=" * 65)
    print(
        tabulate(
            table_rows,
            headers=["#", "Article Title", "Runs (1, 2, 3)", "Mean", "Std Dev"],
            tablefmt="rounded_outline",
        )
    )
    print(f"Average Standard Deviation across articles: {avg_std:.3f}")
    print(f"(A lower Std Dev indicates higher consistency in LLM grading)")
    print("=" * 65 + "\n")

    return {
        "articles_tested": sample_size,
        "runs_per_article": runs,
        "average_std_dev": round(avg_std, 4),
        "details": results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 4: Gemini Escalation Scoring (0-10)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python score.py                         # Score all unscored articles
  python score.py --limit 10              # Test on 10 articles
  python score.py --rerun                 # Wipe and re-score all articles
  python score.py --generate-validation   # Generate 25-article validation JSONL
  python score.py --consistency-check     # Measure std dev across 10 articles x 3 runs
""",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum articles to score in this run",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Clear scores table and re-score all articles",
    )
    parser.add_argument(
        "--generate-validation",
        action="store_true",
        help="Generate validation/score_validation.jsonl for human evaluation",
    )
    parser.add_argument(
        "--consistency-check",
        action="store_true",
        help="Run 10 articles 3 times each to measure LLM scoring stability",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=25,
        help="Number of articles for validation template (default: 25)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to pause between API calls for rate limiting (default: 1.0)",
    )

    args = parser.parse_args()

    if args.generate_validation:
        generate_validation_file(val_size=args.val_size)
    elif args.consistency_check:
        run_consistency_check()
    else:
        run_scoring(
            limit=args.limit,
            rerun=args.rerun,
            sleep_between_calls=args.sleep,
        )
