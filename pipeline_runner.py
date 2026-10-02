"""
pipeline_runner.py — Lightweight Orchestrator for 1-Click Live News Ingestion & Processing.

Runs the full OSINT intelligence pipeline sequentially:
1. Ingest RSS feeds (ingest.py)
2. Extract Named Entities (ner.py)
3. Classify Threat Categories (classify.py)
4. Score Escalation Severity (score.py)
5. Cluster Event Stories (cluster.py)
"""

import logging
import time
from typing import Callable

from db import init_db, migrate_db
import ingest
import ner
import classify
import score
import cluster

log = logging.getLogger("pipeline_runner")


def run_full_pipeline(
    limit: int | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
) -> dict:
    """
    Execute all pipeline stages in sequence.

    Parameters
    ----------
    limit : int or None
        Optional limit on newly fetched or processed articles (useful for fast updates).
    progress_callback : Callable[[str, float], None] or None
        Optional callback accepting (status_message, progress_fraction [0.0 - 1.0]).

    Returns
    -------
    dict
        Summary of execution results and elapsed time.
    """
    start_time = time.perf_counter()
    summary = {
        "success": True,
        "stages": {},
        "error": None,
    }

    def _notify(msg: str, progress: float):
        log.info(msg)
        if progress_callback:
            progress_callback(msg, progress)

    try:
        # Step 0: Ensure DB is ready
        init_db()
        migrate_db()

        # Step 1: Ingest RSS feeds (0% -> 20%)
        _notify("📡 Ingesting latest feeds from 10 global RSS sources...", 0.1)
        ingest.main(dry_run=False, limit=limit)
        summary["stages"]["ingest"] = "completed"

        # Step 2: Named Entity Recognition (20% -> 40%)
        _notify("🏷️ Extracting geopolitical entities with spaCy...", 0.3)
        ner_res = ner.run_ner(limit=limit)
        summary["stages"]["ner"] = ner_res

        # Step 3: Zero-Shot Threat Classification (40% -> 60%)
        _notify("🎯 Classifying threat domains (BART-large-MNLI)...", 0.5)
        clf_res = classify.run_classification(limit=limit)
        summary["stages"]["classify"] = clf_res

        # Step 4: Escalation Scoring (60% -> 80%)
        _notify("⚡ Assessing escalation severity via Gemini...", 0.7)
        score_res = score.run_scoring(limit=limit)
        summary["stages"]["score"] = score_res

        # Step 5: Event Clustering (80% -> 100%)
        _notify("🔗 Grouping corroborated stories with DBSCAN...", 0.9)
        cluster_res = cluster.run_clustering()
        summary["stages"]["cluster"] = cluster_res

        elapsed = time.perf_counter() - start_time
        summary["elapsed_seconds"] = round(elapsed, 2)
        _notify(f"✅ Live intelligence sync complete in {summary['elapsed_seconds']}s!", 1.0)

    except Exception as e:
        log.error("Pipeline sync failed: %s", e, exc_info=True)
        summary["success"] = False
        summary["error"] = str(e)
        if progress_callback:
            progress_callback(f"❌ Error during sync: {e}", 1.0)

    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    )
    print("=" * 60)
    print("STARTING LIVE INTELLIGENCE PIPELINE SYNC")
    print("=" * 60)
    result = run_full_pipeline()
    print("\nSYNC RESULT:", result)
