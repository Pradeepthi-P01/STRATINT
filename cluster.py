"""
cluster.py — Event Clustering Pipeline via Sentence Embeddings & DBSCAN (Phase 5).

────────────────────────────────────────────────────────────────────────────
DESIGN CHOICES (plain English)
────────────────────────────────────────────────────────────────────────────

1. Model: all-MiniLM-L6-v2 (sentence-transformers)
   Produces high-quality 384-dimensional dense semantic embeddings.
   Lightweight (~80 MB) and runs fast on CPU.

2. Sentence Embeddings vs TF-IDF
   TF-IDF relies on exact keyword overlap and fails on news articles written
   by different journalists using synonyms (e.g., "US penalties on Tehran" vs
   "Washington sanctions on Iran"). Sentence embeddings encode latent semantic
   meaning, allowing articles with distinct vocabulary to cluster correctly.

3. Why DBSCAN over K-Means
   - DBSCAN does NOT require pre-specifying the number of clusters (k).
     A daily news cycle has an unknown number of events.
   - DBSCAN natively isolates non-clustered singleton articles as noise
     (cluster_id = -1), rather than forcing every article into a cluster.
   - min_samples = 2 ensures an event cluster represents at least 2 corroborating
     news reports.

4. Embedding Scope: Title + First 2 Sentences
   The inverted pyramid style of journalism puts the critical event facts
   (who, what, where, when) in the headline and opening sentences. Later
   paragraphs introduce historical background or unrelated commentary that
   adds vector noise.

5. Caching:
   Embeddings are saved to data/embeddings.npz so that re-runs do not
   re-encode already processed articles.
"""

import argparse
import json
import logging
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

import numpy as np
from tabulate import tabulate
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer

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
log = logging.getLogger("cluster")

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_EPS = 0.3
DEFAULT_MIN_SAMPLES = 2

DATA_DIR = Path(__file__).parent / "data"
CACHE_PATH = DATA_DIR / "embeddings.npz"
VALIDATION_DIR = Path(__file__).parent / "validation"
VALIDATION_FILE = VALIDATION_DIR / "cluster_validation.jsonl"
METRICS_PATH = Path(__file__).parent / "metrics.json"


# ─────────────────────────────────────────────────────────────────────────────
# Text Preprocessing Helper
# ─────────────────────────────────────────────────────────────────────────────

def extract_title_and_first_two_sentences(title: str | None, summary: str | None) -> str:
    """
    Extract the title plus only the first 2 sentences of the summary.
    """
    t = (title or "").strip()
    s = (summary or "").strip()

    if not s:
        return t

    # Split on sentence terminals followed by whitespace
    sentences = re.split(r"(?<=[.!?])\s+", s)
    lead_sentences = " ".join(sentences[:2]).strip()

    if t and lead_sentences:
        return f"{t}. {lead_sentences}"
    return t or lead_sentences


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Cache Manager
# ─────────────────────────────────────────────────────────────────────────────

def load_cached_embeddings() -> tuple[dict[str, np.ndarray], SentenceTransformer]:
    """
    Load cached embeddings from data/embeddings.npz if present.
    Returns a dict mapping article_id -> embedding (384,), and the loaded model.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache: dict[str, np.ndarray] = {}

    if CACHE_PATH.exists():
        try:
            data = np.load(CACHE_PATH, allow_pickle=True)
            ids = data["ids"]
            embeddings = data["embeddings"]
            for art_id, emb in zip(ids, embeddings):
                cache[str(art_id)] = emb
            log.info("Loaded %d cached embeddings from %s", len(cache), CACHE_PATH)
        except Exception as e:
            log.warning("Failed to load embedding cache (%s); starting fresh.", e)

    log.info("Loading sentence transformer model: %s ...", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    return cache, model


def save_cached_embeddings(cache: dict[str, np.ndarray]) -> None:
    """
    Save embedding dict to data/embeddings.npz.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ids = np.array(list(cache.keys()))
    embeddings = np.array(list(cache.values()))
    np.savez_compressed(CACHE_PATH, ids=ids, embeddings=embeddings)
    log.info("Saved %d embeddings to cache: %s", len(cache), CACHE_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# DB Operations
# ─────────────────────────────────────────────────────────────────────────────

def insert_clusters(conn, cluster_records: list[tuple[str, int]]) -> int:
    """
    Insert article cluster assignments into SQLite clusters table.
    """
    if not cluster_records:
        return 0
    sql = """
        INSERT OR REPLACE INTO clusters (article_id, cluster_id)
        VALUES (?, ?)
    """
    with conn:
        cursor = conn.executemany(sql, cluster_records)
        return cursor.rowcount


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Logger
# ─────────────────────────────────────────────────────────────────────────────

def append_metrics(run_data: dict) -> None:
    """Append clustering metrics to metrics.json."""
    metrics = {}
    if METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("Existing metrics.json corrupt; starting fresh.")
            metrics = {}

    if "cluster" not in metrics:
        metrics["cluster"] = []

    metrics["cluster"].append(run_data)
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Updated metrics recorded at: %s", METRICS_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_clustering(
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    rerun: bool = False,
) -> dict:
    """
    Cluster all articles in the database using sentence embeddings and DBSCAN.
    """
    init_db()
    migrate_db()
    conn = get_connection()

    if rerun:
        log.warning("--rerun specified: clearing all existing cluster records.")
        with conn:
            conn.execute("DELETE FROM clusters")

    # Check already-clustered articles for idempotency
    existing_count = conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
    total_articles = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]

    if existing_count >= total_articles and not rerun:
        log.info("Nothing to do -- all %d articles already have cluster assignments.", total_articles)
        conn.close()
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "already_clustered",
            "articles_processed": 0,
        }

    # Fetch all articles to cluster
    cursor = conn.execute("SELECT id, title, summary, source FROM articles ORDER BY published DESC, ingested_at DESC")
    articles = cursor.fetchall()
    total = len(articles)

    log.info("Preparing embeddings for %d articles ...", total)
    cache, model = load_cached_embeddings()

    # Identify articles that need embedding
    to_encode_ids = []
    to_encode_texts = []
    for art in articles:
        art_id = art["id"]
        if art_id not in cache:
            to_encode_ids.append(art_id)
            to_encode_texts.append(extract_title_and_first_two_sentences(art["title"], art["summary"]))

    if to_encode_texts:
        log.info("Computing embeddings for %d new articles ...", len(to_encode_texts))
        start_enc = time.perf_counter()
        new_embs = model.encode(to_encode_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        for art_id, emb in zip(to_encode_ids, new_embs):
            cache[art_id] = emb
        save_cached_embeddings(cache)
        log.info("Encoding finished in %.2f seconds.", time.perf_counter() - start_enc)
    else:
        log.info("All %d article embeddings found in cache!", total)

    # Build matrix ordered by articles
    embedding_matrix = np.array([cache[art["id"]] for art in articles])

    # Run DBSCAN
    log.info("Running DBSCAN (metric='cosine', eps=%.2f, min_samples=%d) ...", eps, min_samples)
    start_cluster = time.perf_counter()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = dbscan.fit_predict(embedding_matrix)
    cluster_time = time.perf_counter() - start_cluster

    # Store in DB
    cluster_records = [(art["id"], int(label)) for art, label in zip(articles, labels)]
    insert_clusters(conn, cluster_records)

    # Compute cluster statistics
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    noise_count = int(np.sum(labels == -1))
    clustered_count = total - noise_count
    noise_pct = (noise_count / total * 100) if total > 0 else 0.0

    # Group by cluster to find largest clusters
    from collections import defaultdict
    clusters_dict = defaultdict(list)
    for art, label in zip(articles, labels):
        if label != -1:
            clusters_dict[label].append(art)

    sorted_clusters = sorted(clusters_dict.items(), key=lambda x: len(x[1]), reverse=True)

    print("\n" + "=" * 65)
    print("DBSCAN EVENT CLUSTERING SUMMARY")
    print("=" * 65)
    summary_table = [
        ["Total Articles Processed", total],
        ["Distinct Event Clusters Found", n_clusters],
        ["Articles Grouped into Events", f"{clustered_count} ({(clustered_count/total*100):.1f}%)"],
        ["Isolated Noise Articles (-1)", f"{noise_count} ({noise_pct:.1f}%)"],
        ["DBSCAN Hyperparameters", f"eps={eps}, min_samples={min_samples}, metric='cosine'"],
        ["Clustering Compute Time", f"{cluster_time:.3f}s"],
    ]
    print(tabulate(summary_table, headers=["Metric", "Value"], tablefmt="rounded_outline"))

    # Print Top 3 Largest Clusters
    print("\nTOP 3 LARGEST EVENT CLUSTERS:")
    top_3 = sorted_clusters[:3]
    for rank, (cid, arts) in enumerate(top_3, start=1):
        print(f"\nCluster #{cid} ({len(arts)} articles):")
        for a in arts[:3]:
            print(f"  - [{a['source']}] {a['title'][:70]}")
        if len(arts) > 3:
            print(f"    ... and {len(arts) - 3} more articles")
    print("=" * 65 + "\n")

    conn.close()

    run_stats = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "eps": eps,
        "min_samples": min_samples,
        "articles_processed": total,
        "clusters_found": n_clusters,
        "clustered_articles": clustered_count,
        "noise_articles": noise_count,
        "noise_percentage": round(noise_pct, 2),
    }
    append_metrics(run_stats)
    return run_stats


# ─────────────────────────────────────────────────────────────────────────────
# Epsilon Parameter Tuning Grid [0.2, 0.3, 0.4, 0.5]
# ─────────────────────────────────────────────────────────────────────────────

def run_eps_tuning(eps_values: list[float] = [0.2, 0.3, 0.4, 0.5]) -> None:
    """
    Run DBSCAN across multiple eps values to report cluster counts and noise %.
    """
    init_db()
    conn = get_connection()
    articles = conn.execute("SELECT id, title, summary FROM articles").fetchall()
    conn.close()

    if not articles:
        log.error("No articles found in DB.")
        sys.exit(1)

    cache, model = load_cached_embeddings()
    texts = [extract_title_and_first_two_sentences(a["title"], a["summary"]) for a in articles]
    ids = [a["id"] for a in articles]

    # Fill any missing
    missing_ids = [i for i in ids if i not in cache]
    if missing_ids:
        missing_texts = [extract_title_and_first_two_sentences(a["title"], a["summary"]) for a in articles if a["id"] in missing_ids]
        embs = model.encode(missing_texts, normalize_embeddings=True)
        for i, emb in zip(missing_ids, embs):
            cache[i] = emb
        save_cached_embeddings(cache)

    matrix = np.array([cache[i] for i in ids])
    total = len(ids)

    rows = []
    for eps in eps_values:
        db = DBSCAN(eps=eps, min_samples=DEFAULT_MIN_SAMPLES, metric="cosine")
        lbls = db.fit_predict(matrix)
        clusters = len(set(lbls) - {-1})
        noise = int(np.sum(lbls == -1))
        noise_pct = (noise / total * 100) if total > 0 else 0.0
        clustered = total - noise

        rows.append([
            f"eps = {eps}",
            clusters,
            clustered,
            noise,
            f"{noise_pct:.1f}%",
            "Tight / High Precision" if eps == 0.2 else ("Recommended Balance" if eps == 0.3 else ("Broader groupings" if eps == 0.4 else "Over-merging risk")),
        ])

    print("\n" + "=" * 70)
    print("DBSCAN EPSILON HYPERPARAMETER TUNING REPORT")
    print("=" * 70)
    print(
        tabulate(
            rows,
            headers=["Epsilon (eps)", "Clusters", "Articles in Events", "Noise (-1)", "Noise %", "Characteristics"],
            tablefmt="rounded_outline",
        )
    )
    print(f"Total articles tested: {total}")
    print(f"Embedding model:       {MODEL_NAME}")
    print(f"Distance metric:       Cosine Distance (1 - cosine_similarity)")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Validation Set Generator (20 Article Pairs)
# ─────────────────────────────────────────────────────────────────────────────

def generate_validation_file(n_pairs: int = 20) -> Path:
    """
    Generate validation/cluster_validation.jsonl with 20 article pairs
    (10 from same cluster, 10 from different/noise) for human validation.
    """
    init_db()
    conn = get_connection()
    rows = conn.execute("""
        SELECT a.id, a.title, a.summary, c.cluster_id, a.source
        FROM articles a
        JOIN clusters c ON a.id = c.article_id
    """).fetchall()
    conn.close()

    if not rows:
        log.error("No clustered articles found. Run `python cluster.py` first!")
        sys.exit(1)

    from collections import defaultdict
    clusters_dict = defaultdict(list)
    noise_list = []
    for r in rows:
        if r["cluster_id"] != -1:
            clusters_dict[r["cluster_id"]].append(r)
        else:
            noise_list.append(r)

    cache, _ = load_cached_embeddings()
    pairs = []
    pair_id = 1

    # 1. Collect 10 pairs from same cluster (predicted_same_event = 1)
    for cid, arts in clusters_dict.items():
        if len(arts) >= 2 and len(pairs) < 10:
            a1 = arts[0]
            a2 = arts[1]
            e1 = cache.get(a1["id"])
            e2 = cache.get(a2["id"])
            dist = float(cosine_distances([e1], [e2])[0][0]) if e1 is not None and e2 is not None else None

            pairs.append({
                "pair_id": pair_id,
                "article_1_id": a1["id"],
                "article_1_title": a1["title"],
                "article_1_source": a1["source"],
                "article_2_id": a2["id"],
                "article_2_title": a2["title"],
                "article_2_source": a2["source"],
                "cosine_distance": round(dist, 4) if dist is not None else None,
                "predicted_same_event": 1,
                "manual_same_event": 1,  # Pre-filled best guess
            })
            pair_id += 1

    # 2. Collect 10 pairs from different clusters or noise (predicted_same_event = 0)
    all_cluster_keys = list(clusters_dict.keys())
    for i in range(len(all_cluster_keys) - 1):
        if len(pairs) >= n_pairs:
            break
        c1 = all_cluster_keys[i]
        c2 = all_cluster_keys[i + 1]
        a1 = clusters_dict[c1][0]
        a2 = clusters_dict[c2][0]
        e1 = cache.get(a1["id"])
        e2 = cache.get(a2["id"])
        dist = float(cosine_distances([e1], [e2])[0][0]) if e1 is not None and e2 is not None else None

        pairs.append({
            "pair_id": pair_id,
            "article_1_id": a1["id"],
            "article_1_title": a1["title"],
            "article_1_source": a1["source"],
            "article_2_id": a2["id"],
            "article_2_title": a2["title"],
            "article_2_source": a2["source"],
            "cosine_distance": round(dist, 4) if dist is not None else None,
            "predicted_same_event": 0,
            "manual_same_event": 0,  # Pre-filled best guess
        })
        pair_id += 1

    # Pad with noise if needed
    while len(pairs) < n_pairs and len(noise_list) >= 2:
        a1 = noise_list.pop(0)
        a2 = noise_list.pop(0)
        e1 = cache.get(a1["id"])
        e2 = cache.get(a2["id"])
        dist = float(cosine_distances([e1], [e2])[0][0]) if e1 is not None and e2 is not None else None
        pairs.append({
            "pair_id": pair_id,
            "article_1_id": a1["id"],
            "article_1_title": a1["title"],
            "article_1_source": a1["source"],
            "article_2_id": a2["id"],
            "article_2_title": a2["title"],
            "article_2_source": a2["source"],
            "cosine_distance": round(dist, 4) if dist is not None else None,
            "predicted_same_event": 0,
            "manual_same_event": 0,
        })
        pair_id += 1

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    with open(VALIDATION_FILE, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    log.info("Wrote %d validation pairs to: %s", len(pairs), VALIDATION_FILE)
    print("\n" + "=" * 65)
    print("CLUSTER VALIDATION TEMPLATE GENERATED")
    print("=" * 65)
    print(f"File:     {VALIDATION_FILE}")
    print(f"Pairs:    {len(pairs)}")
    print("\nReview the pairs and adjust 'manual_same_event' (1 = same event, 0 = different),")
    print("then run:  python evaluate.py --phase cluster")
    print("=" * 65 + "\n")
    return VALIDATION_FILE


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 5: Event Clustering with Sentence Transformers + DBSCAN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cluster.py                     # Cluster all articles with default eps=0.3
  python cluster.py --tune-eps          # Grid search eps [0.2, 0.3, 0.4, 0.5]
  python cluster.py --rerun             # Re-cluster all articles
  python cluster.py --generate-validation # Generate 20-pair validation template
""",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=DEFAULT_EPS,
        help="DBSCAN cosine distance threshold (default: 0.3)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help="DBSCAN min_samples (default: 2)",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Clear existing clusters and recompute",
    )
    parser.add_argument(
        "--tune-eps",
        action="store_true",
        help="Evaluate cluster count and noise % across eps=[0.2, 0.3, 0.4, 0.5]",
    )
    parser.add_argument(
        "--generate-validation",
        action="store_true",
        help="Generate 20-pair validation JSONL for evaluation",
    )

    args = parser.parse_args()

    if args.tune_eps:
        run_eps_tuning()
    elif args.generate_validation:
        generate_validation_file()
    else:
        run_clustering(
            eps=args.eps,
            min_samples=args.min_samples,
            rerun=args.rerun,
        )
