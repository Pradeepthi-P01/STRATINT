"""
classify.py — Zero-Shot Threat Classification Pipeline (Phase 3).

────────────────────────────────────────────────────────────────────────────
DESIGN CHOICES (plain English)
────────────────────────────────────────────────────────────────────────────

1. Model: facebook/bart-large-mnli
   # Model: facebook/bart-large-mnli, 1.63GB, CPU-only, cached after first run at C:\\Users\\prade\\.cache\\huggingface
   BART-large trained on MultiNLI (MNLI) is the de-facto standard for zero-shot
   text classification. It operates without any task-specific fine-tuning by
   framing classification as Natural Language Inference (NLI).

2. Zero-Shot via Natural Language Inference (NLI)
   Under the hood, zero-shot classification does NOT compare text to a fixed
   list of keywords. Instead, it frames the problem as:
     - Premise:    The article text (title + summary)
     - Hypothesis: "This text is about {category}."
   The model evaluates whether the Premise logically ENTAILS the Hypothesis.
   Because NLI models were trained on hundreds of thousands of sentence pairs,
   they understand semantic nuance, context, and implied meaning.

3. Why multi_label=True is essential for news
   Standard single-label classification applies Softmax across candidate labels,
   forcing probabilities to sum to 1.0. This assumes categories are mutually
   exclusive. In real-world geopolitical intelligence, categories frequently
   overlap:
     - A state-sponsored cyberattack on a power grid during a war is BOTH
       "cyberattack" and "military conflict".
     - Mass protests in response to a diplomatic breakdown are BOTH
       "civil unrest" and "diplomatic tension".
   With multi_label=True, the model applies an independent Sigmoid function to
   each candidate label. Each category receives its own independent probability
   in [0, 1] without competing against the others.

4. Why a 0.45 confidence threshold?
   In multi-label NLI with BART-large, a score of ~0.50 represents the balance
   point between neutral/contradiction and entailment. Setting the threshold at
   0.45 provides high recall on concise news summaries (capturing secondary or
   co-occurring threats) while filtering out low-probability noise and non-threat
   articles.

5. Storage Schema & Idempotence
   Articles can belong to 0, 1, or multiple categories:
     - Stored in the `classifications` table with one row per (article_id, category, confidence).
     - UNIQUE(article_id, category) ensures re-runs will not duplicate rows.
     - By default, articles already present in `classifications` are skipped,
       making runs idempotent and safe to interrupt or resume.
     - Pass `--rerun` to wipe and recompute all classifications.

────────────────────────────────────────────────────────────────────────────
ASSUMPTIONS
────────────────────────────────────────────────────────────────────────────
- PyTorch and HuggingFace Transformers installed.
- Weights are cached at ~/.cache/huggingface/hub after first download (~1.63 GB).
- Target device is CPU.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Windows UTF-8 stdout fix ──────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import sqlite3
from tabulate import tabulate
from transformers import pipeline

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
log = logging.getLogger("classify")

# ── Constants ─────────────────────────────────────────────────────────────────
# Model: facebook/bart-large-mnli, 1.63GB, CPU-only, cached after first run at C:\Users\prade\.cache\huggingface
MODEL_NAME = "facebook/bart-large-mnli"
HYPOTHESIS_TEMPLATE = "This text is about {}."
DEFAULT_THRESHOLD = 0.45

# The 6 exact OSINT threat categories
THREAT_CATEGORIES = [
    "military conflict",
    "cyberattack",
    "civil unrest",
    "diplomatic tension",
    "terrorism or extremist violence",
    "natural disaster",
]

VALIDATION_DIR = Path(__file__).parent / "validation"
VALIDATION_FILE = VALIDATION_DIR / "classification_validation.jsonl"
METRICS_PATH = Path(__file__).parent / "metrics.json"


# ─────────────────────────────────────────────────────────────────────────────
# Model Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_classifier(model_name: str = MODEL_NAME):
    """
    Load the HuggingFace zero-shot classification pipeline on CPU.
    """
    log.info("Loading zero-shot classifier: %s ...", model_name)
    start = time.perf_counter()
    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device="cpu",
    )
    elapsed = time.perf_counter() - start
    log.info("Classifier loaded in %.2f seconds (CPU).", elapsed)
    return classifier


# ─────────────────────────────────────────────────────────────────────────────
# Text Helper
# ─────────────────────────────────────────────────────────────────────────────

def article_to_text(article: sqlite3.Row) -> str:
    """
    Combine title and summary for classification input.
    """
    parts = []
    if article["title"]:
        parts.append(article["title"].strip())
    if article["summary"]:
        parts.append(article["summary"].strip())
    return " ".join(parts).strip()


# ─────────────────────────────────────────────────────────────────────────────
# DB Operations
# ─────────────────────────────────────────────────────────────────────────────

def insert_classifications(
    conn: sqlite3.Connection,
    records: list[tuple[str, str, float]],
) -> int:
    """
    Insert classified threat tags into the classifications table.

    Parameters
    ----------
    conn : sqlite3.Connection
    records : list of (article_id, category, confidence)

    Returns
    -------
    int : number of rows inserted
    """
    if not records:
        return 0
    sql = """
        INSERT OR IGNORE INTO classifications (article_id, category, confidence)
        VALUES (?, ?, ?)
    """
    with conn:
        cursor = conn.executemany(sql, records)
        return cursor.rowcount


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Helpers
# ─────────────────────────────────────────────────────────────────────────────

def append_metrics(run_data: dict) -> None:
    """
    Append run statistics to metrics.json under the 'classify' key.
    """
    metrics = {}
    if METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("Existing metrics.json corrupt; starting fresh.")
            metrics = {}

    if "classify" not in metrics:
        metrics["classify"] = []

    metrics["classify"].append(run_data)
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Updated metrics recorded at: %s", METRICS_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_classification(
    batch_size: int = 8,
    limit: int | None = None,
    rerun: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
    model_name: str = MODEL_NAME,
) -> dict:
    """
    Run the zero-shot classification pipeline across unclassified articles.
    """
    init_db()
    migrate_db()
    conn = get_connection()

    if rerun:
        log.warning("--rerun specified: clearing all existing classifications.")
        with conn:
            conn.execute("DELETE FROM classifications")
            conn.execute("DELETE FROM classified_articles")

    articles = get_unprocessed_articles(conn, "classified_articles", limit=limit)
    total = len(articles)
    log.info("Articles to classify: %d", total)

    if total == 0:
        log.info("Nothing to do -- all articles already have classification rows.")
        conn.close()
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "articles_processed": 0,
            "classifications_stored": 0,
        }

    classifier = load_classifier(model_name)

    pairs = [(art["id"], article_to_text(art)) for art in articles]
    ids = [p[0] for p in pairs]
    texts = [p[1] for p in pairs]

    category_counts = {cat: 0 for cat in THREAT_CATEGORIES}
    total_stored = 0
    start_time = time.perf_counter()

    log.info("Starting zero-shot inference (threshold=%.2f, batch_size=%d) ...", threshold, batch_size)

    # Process in batches
    for i in range(0, total, batch_size):
        batch_ids = ids[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]

        # Call HuggingFace pipeline
        outputs = classifier(
            batch_texts,
            candidate_labels=THREAT_CATEGORIES,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=True,
        )

        if not isinstance(outputs, list):
            outputs = [outputs]

        batch_records: list[tuple[str, str, float]] = []
        for article_id, out in zip(batch_ids, outputs):
            for label, score in zip(out["labels"], out["scores"]):
                if score >= threshold:
                    batch_records.append((article_id, label, round(float(score), 4)))
                    category_counts[label] += 1

        inserted = insert_classifications(conn, batch_records)
        total_stored += inserted

        # Mark all articles in this batch as classified
        with conn:
            now_iso = datetime.now(timezone.utc).isoformat()
            conn.executemany(
                "INSERT OR IGNORE INTO classified_articles (article_id, classified_at) VALUES (?, ?)",
                [(art_id, now_iso) for art_id in batch_ids],
            )

        done = min(i + batch_size, total)
        log.info(
            "Progress: %d/%d articles processed (stored %d labels in this batch)",
            done,
            total,
            inserted,
        )

    elapsed = round(time.perf_counter() - start_time, 2)
    conn.close()

    log.info("Classification completed in %.2f seconds.", elapsed)
    log.info("Total threat classifications stored: %d across %d articles.", total_stored, total)

    # Pretty print summary table
    summary_rows = [
        [cat, count, f"{(count / total * 100):.1f}%" if total > 0 else "0%"]
        for cat, count in category_counts.items()
    ]
    print("\n" + "=" * 60)
    print("ZERO-SHOT THREAT CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(
        tabulate(
            summary_rows,
            headers=["Threat Category", "Tagged Articles", "% of Processed"],
            tablefmt="rounded_outline",
        )
    )
    print(f"Total articles classified: {total}")
    print(f"Total tags stored:         {total_stored}")
    print(f"Confidence threshold:      >= {threshold}")
    print(f"Elapsed time:              {elapsed:.2f}s")
    print("=" * 60 + "\n")

    run_stats = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "device": "cpu",
        "threshold": threshold,
        "articles_processed": total,
        "classifications_stored": total_stored,
        "category_distribution": category_counts,
        "elapsed_seconds": elapsed,
    }
    append_metrics(run_stats)
    return run_stats


# ─────────────────────────────────────────────────────────────────────────────
# Validation Set Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_validation_file(
    val_size: int = 25,
    threshold: float = DEFAULT_THRESHOLD,
    model_name: str = MODEL_NAME,
) -> Path:
    """
    Generate a pre-filled validation JSONL template for human review.

    Pulls articles across diverse sources from the articles table, runs the
    model to obtain candidate predictions, and writes them formatted for easy
    manual ground-truth correction.
    """
    init_db()
    conn = get_connection()

    # Pull balanced sample of articles across available sources
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

    log.info("Generating pre-filled validation template for %d articles ...", len(articles))
    classifier = load_classifier(model_name)

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for art in articles:
        text = article_to_text(art)
        out = classifier(
            text,
            candidate_labels=THREAT_CATEGORIES,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=True,
        )

        predicted_categories = [
            label for label, score in zip(out["labels"], out["scores"])
            if score >= threshold
        ]

        records.append({
            "article_id": art["id"],
            "source": art["source"],
            "published": art["published"],
            "text": text,
            "categories": predicted_categories,
        })

    with open(VALIDATION_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info("Validation template written to: %s", VALIDATION_FILE)
    print("\n" + "=" * 65)
    print("PRE-FILLED CLASSIFICATION VALIDATION TEMPLATE GENERATED")
    print("=" * 65)
    print(f"File: {VALIDATION_FILE}")
    print(f"Articles: {len(records)}")
    print("\nNext step: Inspect and adjust 'categories' in this file if needed,")
    print("then run:  python evaluate.py --phase classify")
    print("=" * 65 + "\n")
    return VALIDATION_FILE


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 3: Zero-Shot Threat Classification with BART-large-MNLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python classify.py                      # Classify all unprocessed articles
  python classify.py --limit 10           # Test on 10 articles
  python classify.py --batch-size 16      # Run with larger batch size
  python classify.py --rerun              # Clear and re-classify all
  python classify.py --generate-validation  # Generate validation JSONL
""",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for HuggingFace pipeline (default: 8)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of articles to classify",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Confidence threshold for multi-label assignment (default: 0.45)",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Wipe classifications table and re-run all articles",
    )
    parser.add_argument(
        "--generate-validation",
        action="store_true",
        help="Generate validation/classification_validation.jsonl for evaluation",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=25,
        help="Number of articles for validation template (default: 25)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="Model name (default: facebook/bart-large-mnli)",
    )

    args = parser.parse_args()

    if args.generate_validation:
        generate_validation_file(
            val_size=args.val_size,
            threshold=args.threshold,
            model_name=args.model,
        )
    else:
        run_classification(
            batch_size=args.batch_size,
            limit=args.limit,
            rerun=args.rerun,
            threshold=args.threshold,
            model_name=args.model,
        )
