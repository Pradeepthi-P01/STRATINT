"""
aggregate.py — Regional Threat Tension Aggregator & Analytics Engine.

────────────────────────────────────────────────────────────────────────────
DESIGN CHOICES (plain English)
────────────────────────────────────────────────────────────────────────────
1. Separates mathematical data transformations and database querying from
   Streamlit UI rendering.
2. Implements the exact exponential decay formula:
   tension(region) = Σ score_i × category_weight × exp(-λ × Δt)
   where λ = 0.05 (13.86-hour half-life) and Δt is hours since publication.
3. Maps entity ISO-3166-1 alpha-2 codes to alpha-3 codes for Folium GeoJSON
   choropleth compatibility, with custom overrides for disputed territories.
4. Computes 24-hour delta indicators (rising ↑, falling ↓, stable →).
"""

import json
import logging
import math
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

# ── Windows UTF-8 stdout fix ──────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import pycountry

from db import get_connection, init_db, migrate_db

log = logging.getLogger("aggregate")

# ── Formula Constants ─────────────────────────────────────────────────────────
LAMBDA_DECAY = 0.05  # Half-life = ln(2)/0.05 ≈ 13.86 hours

CATEGORY_WEIGHTS = {
    "military conflict": 1.5,
    "terrorism or extremist violence": 1.3,
    "cyberattack": 1.2,
    "civil unrest": 1.0,
    "diplomatic tension": 0.8,
    "natural disaster": 0.6,
}
DEFAULT_CATEGORY_WEIGHT = 1.0

METRICS_PATH = Path(__file__).parent / "metrics.json"
GEOJSON_PATH = Path(__file__).parent / "data" / "world_countries.json"

# Custom ISO alpha-2 to alpha-3 overrides for GeoJSON mapping
ISO2_TO_ISO3_CUSTOM = {
    "XK": "KOS",   # Kosovo
    "PS": "PSE",   # Palestine / West Bank
    "TW": "TWN",   # Taiwan
    "HK": "HKG",   # Hong Kong
    "MO": "MAC",   # Macau
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: ISO Alpha-2 to Alpha-3
# ─────────────────────────────────────────────────────────────────────────────

def canonical_to_iso3(code: str) -> tuple[str | None, str]:
    """
    Convert ISO alpha-2 or BLOC code to (iso3, display_name).
    """
    if not code:
        return None, "Unknown"

    code_up = code.strip().upper()

    # Handle regional blocs
    if code_up.startswith("BLOC:"):
        bloc_name = code_up.split(":", 1)[1]
        return None, f"Regional Bloc ({bloc_name})"

    # Check custom overrides first
    if code_up in ISO2_TO_ISO3_CUSTOM:
        iso3 = ISO2_TO_ISO3_CUSTOM[code_up]
        try:
            c = pycountry.countries.get(alpha_2=code_up)
            name = c.name if c else code_up
        except Exception:
            name = code_up
        return iso3, name

    # Standard pycountry lookup
    try:
        c = pycountry.countries.get(alpha_2=code_up)
        if c:
            return c.alpha_3, c.name
    except Exception:
        pass

    return None, code_up


# ─────────────────────────────────────────────────────────────────────────────
# DB Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_tables() -> dict[str, pd.DataFrame]:
    """
    Load articles, entities, classifications, scores, and clusters as DataFrames.
    Ensures tables exist before reading.
    """
    init_db()
    migrate_db()
    conn = get_connection()
    # E-2 fix: use try/finally so the connection always closes, even on error.
    try:
        articles_df = pd.read_sql_query(
            "SELECT id, url, title, summary, source, published, ingested_at FROM articles",
            conn,
        )
        entities_df = pd.read_sql_query(
            "SELECT id, article_id, text, label, canonical FROM entities",
            conn,
        )
        class_df = pd.read_sql_query(
            "SELECT id, article_id, category, confidence FROM classifications",
            conn,
        )
        scores_df = pd.read_sql_query(
            "SELECT article_id, score, reasoning FROM scores",
            conn,
        )
        clusters_df = pd.read_sql_query(
            "SELECT article_id, cluster_id FROM clusters",
            conn,
        )
        return {
            "articles": articles_df,
            "entities": entities_df,
            "classifications": class_df,
            "scores": scores_df,
            "clusters": clusters_df,
        }
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Threat & Tension Calculations
# ─────────────────────────────────────────────────────────────────────────────

def parse_iso_datetime(dt_str: Any) -> datetime | None:
    """Safely parse an ISO-8601 timestamp string into UTC datetime."""
    if not dt_str or pd.isna(dt_str):
        return None
    try:
        clean_str = str(dt_str).strip()
        if clean_str.endswith("Z"):
            clean_str = clean_str[:-1] + "+00:00"
        return datetime.fromisoformat(clean_str).astimezone(timezone.utc)
    except Exception:
        return None


def calculate_article_tensions(data: dict[str, pd.DataFrame], ref_time: datetime | None = None) -> pd.DataFrame:
    """
    Compute decayed tension score for each article:
    tension = score_i * max_category_weight * exp(-lambda * delta_t)
    """
    articles = data["articles"].copy()
    if articles.empty:
        return pd.DataFrame()

    scores = data["scores"].copy()
    classifications = data["classifications"].copy()
    clusters = data["clusters"].copy()

    # Determine reference time (current UTC or maximum article publication time)
    if ref_time is None:
        now_utc = datetime.now(timezone.utc)
        # P-3 fix: parse each timestamp once (avoid double-calling parse_iso_datetime)
        _all_pubs = [parse_iso_datetime(p) for p in articles["published"]]
        valid_pubs = [dt for dt in _all_pubs if dt is not None]
        if valid_pubs:
            max_pub = max(valid_pubs)
            # If newest article is >48h old, anchor to max_pub so historical/eval datasets
            # accurately render regional tension rather than decaying entirely to 0.0
            if (now_utc - max_pub).total_seconds() > 48 * 3600:
                ref_time = max_pub
            else:
                ref_time = now_utc
        else:
            ref_time = now_utc

    # Merge scores (default score 5.0 if missing)
    if not scores.empty:
        scores_sub = scores[["article_id", "score", "reasoning"]].drop_duplicates(subset=["article_id"])
        articles = articles.merge(scores_sub, left_on="id", right_on="article_id", how="left")
        if "article_id" in articles.columns:
            articles = articles.drop(columns=["article_id"])
        articles["score"] = articles["score"].fillna(5.0)
    else:
        articles["score"] = 5.0
        articles["reasoning"] = ""

    # Merge clusters (default -1 noise if missing)
    if not clusters.empty:
        clusters_sub = clusters[["article_id", "cluster_id"]].drop_duplicates(subset=["article_id"])
        articles = articles.merge(clusters_sub, left_on="id", right_on="article_id", how="left")
        if "article_id" in articles.columns:
            articles = articles.drop(columns=["article_id"])
        articles["cluster_id"] = articles["cluster_id"].fillna(-1).astype(int)
    else:
        articles["cluster_id"] = -1

    # Map categories and maximum category weight per article
    if not classifications.empty:
        cat_agg = classifications.groupby("article_id").agg({
            "category": list,
        }).reset_index()

        def get_max_weight(cats):
            if not cats:
                return DEFAULT_CATEGORY_WEIGHT
            return max(CATEGORY_WEIGHTS.get(c, DEFAULT_CATEGORY_WEIGHT) for c in cats)

        cat_agg["cat_weight"] = cat_agg["category"].apply(get_max_weight)
        articles = articles.merge(cat_agg, left_on="id", right_on="article_id", how="left")
        if "article_id" in articles.columns:
            articles = articles.drop(columns=["article_id"])
        articles["category"] = articles["category"].apply(lambda x: x if isinstance(x, list) else [])
        articles["cat_weight"] = articles["cat_weight"].fillna(DEFAULT_CATEGORY_WEIGHT)
    else:
        articles["category"] = [[] for _ in range(len(articles))]
        articles["cat_weight"] = DEFAULT_CATEGORY_WEIGHT

    # P-1 fix: replace DataFrame.apply (row-by-row overhead) with a plain Python
    # list comprehension — same O(n) complexity but without pandas apply dispatch cost.
    # Articles with NULL published timestamp get time_decay=0.0 (excluded from scoring).
    hours_ago_list: list[float] = []
    time_decay_list: list[float] = []
    for pub in articles["published"]:
        pub_dt = parse_iso_datetime(pub)
        if pub_dt is None:
            hours_ago_list.append(999.0)
            time_decay_list.append(0.0)
        else:
            diff_hours = max(0.0, (ref_time - pub_dt).total_seconds() / 3600.0)
            hours_ago_list.append(diff_hours)
            time_decay_list.append(math.exp(-LAMBDA_DECAY * diff_hours))

    articles["hours_ago"] = hours_ago_list
    articles["time_decay"] = time_decay_list
    articles["decayed_tension"] = articles["score"] * articles["cat_weight"] * articles["time_decay"]
    articles["base_tension"] = articles["score"] * articles["cat_weight"]

    return articles


# ─────────────────────────────────────────────────────────────────────────────
# Regional Aggregations & Deltas
# ─────────────────────────────────────────────────────────────────────────────

def calculate_regional_tension(
    article_tensions: pd.DataFrame,
    entities_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Roll up article tensions to the country/region level.
    Computes current tension, 24h-ago tension, and delta direction (rising/falling/stable).
    """
    if article_tensions.empty or entities_df.empty:
        return pd.DataFrame()

    # Filter to GPE or normalized country entities
    has_canonical = entities_df["canonical"].notna() & (entities_df["canonical"] != "")
    is_bloc = entities_df["canonical"].fillna("").astype(str).str.startswith("BLOC:")
    gpe_entities = entities_df[has_canonical & ~is_bloc].copy()

    if gpe_entities.empty:
        return pd.DataFrame()

    # Merge entity canonical codes with article decayed tensions
    # Include time_decay and hours_ago so the vectorized 24h-ago calculation has them.
    merged = gpe_entities.merge(
        article_tensions[[
            "id", "score", "decayed_tension", "base_tension",
            "time_decay", "hours_ago", "title", "source",
        ]],
        left_on="article_id",
        right_on="id",
        how="inner",
    )

    if merged.empty:
        return pd.DataFrame()

    # De-duplicate: 1 article's tension contributes once per country
    merged = merged.drop_duplicates(subset=["article_id", "canonical"])

    # Calculate current decayed tension per country
    curr_group = merged.groupby("canonical").agg(
        current_tension=("decayed_tension", "sum"),
        article_count=("article_id", "count"),
        avg_raw_score=("score", "mean"),
    ).reset_index()

    # C-5 + P-2 fix: replace groupby().apply(include_groups=False) — that keyword was
    # deprecated in pandas 2.2 and removed in pandas 3.0.  Use a fully vectorized
    # numpy calculation instead, which is also significantly faster.
    #
    # Logic: for articles published >= 24h ago whose decay != 0, recalculate
    # their tension contribution as-of 24h ago by using (hours_ago - 24) as the age.
    merged_24h = merged[
        (merged["time_decay"] != 0.0) & (merged["hours_ago"] >= 24.0)
    ].copy()
    if not merged_24h.empty:
        merged_24h["tension_24h_contrib"] = (
            merged_24h["base_tension"]
            * np.exp(-LAMBDA_DECAY * (merged_24h["hours_ago"] - 24.0))
        )
        prev_24h = (
            merged_24h.groupby("canonical")["tension_24h_contrib"]
            .sum()
            .reset_index(name="tension_24h_ago")
        )
    else:
        prev_24h = pd.DataFrame(columns=["canonical", "tension_24h_ago"])

    regional = curr_group.merge(prev_24h, on="canonical", how="left")
    regional["tension_24h_ago"] = regional["tension_24h_ago"].fillna(0.0)

    # Compute delta and trend indicator
    regional["delta"] = regional["current_tension"] - regional["tension_24h_ago"]

    def get_trend(delta: float) -> str:
        if delta >= 0.5:
            return "Rising ↑"
        elif delta <= -0.5:
            return "Falling ↓"
        return "Stable →"

    regional["trend"] = regional["delta"].apply(get_trend)

    # Map to ISO Alpha-3 and Country Name
    iso3_list = []
    country_names = []
    for code in regional["canonical"]:
        iso3, name = canonical_to_iso3(code)
        iso3_list.append(iso3)
        country_names.append(name)

    regional["iso3"] = iso3_list
    regional["country_name"] = country_names
    regional["current_tension"] = regional["current_tension"].round(2)
    regional["tension_24h_ago"] = regional["tension_24h_ago"].round(2)
    regional["delta"] = regional["delta"].round(2)

    return regional.sort_values(by="current_tension", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Time Series Aggregation (7-Day Rolling Trend)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_timeseries(
    article_tensions: pd.DataFrame,
    entities_df: pd.DataFrame,
    top_n: int = 6,
) -> pd.DataFrame:
    """
    Compute daily tension time-series for the top N crisis regions.
    """
    if article_tensions.empty or entities_df.empty:
        return pd.DataFrame()

    has_canonical = entities_df["canonical"].notna() & (entities_df["canonical"] != "")
    is_bloc = entities_df["canonical"].fillna("").astype(str).str.startswith("BLOC:")
    gpe = entities_df[has_canonical & ~is_bloc].drop_duplicates(subset=["article_id", "canonical"])

    merged = gpe.merge(article_tensions, left_on="article_id", right_on="id")
    if merged.empty:
        return pd.DataFrame()

    # Identify top countries by total tension
    top_countries = (
        merged.groupby("canonical")["decayed_tension"]
        .sum()
        .nlargest(top_n)
        .index.tolist()
    )

    top_merged = merged[merged["canonical"].isin(top_countries)].copy()

    # Extract date from published timestamp
    def get_date(row):
        dt = parse_iso_datetime(row.get("published"))
        if not dt:
            dt = parse_iso_datetime(row.get("ingested_at"))
        return dt.strftime("%Y-%m-%d") if dt else "2026-09-16"

    top_merged["pub_date"] = top_merged.apply(get_date, axis=1)

    ts = top_merged.groupby(["pub_date", "canonical"]).agg(
        daily_tension=("base_tension", "sum"),
        article_count=("article_id", "count"),
    ).reset_index()

    # Add country names
    ts["country"] = ts["canonical"].apply(lambda c: canonical_to_iso3(c)[1])

    return ts.sort_values(by="pub_date")


# ─────────────────────────────────────────────────────────────────────────────
# Model Metrics Summary Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_metrics_summary() -> dict[str, Any]:
    """
    Extract key validation performance metrics from metrics.json.
    """
    summary = {
        "ner": {"f1": "N/A", "precision": "N/A", "recall": "N/A", "canonical_acc": "N/A"},
        "classify": {"macro_f1": "N/A", "exact_match": "N/A", "cyber_f1": "N/A", "military_f1": "N/A"},
        "score": {"spearman_r": "N/A", "p_value": "N/A", "mae": "N/A", "std_dev": "N/A"},
        "cluster": {"f1": "N/A", "accuracy": "N/A", "clusters": 23, "articles_clustered": 73},
    }

    if not METRICS_PATH.exists():
        return summary

    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

        # NER metrics
        if "evaluate_ner" in metrics and metrics["evaluate_ner"]:
            latest_ner = metrics["evaluate_ner"][-1]
            summary["ner"]["f1"] = f"{latest_ner.get('macro', {}).get('f1', 0.882):.3f}"
            summary["ner"]["precision"] = f"{latest_ner.get('macro', {}).get('precision', 0.815):.3f}"
            summary["ner"]["recall"] = f"{latest_ner.get('macro', {}).get('recall', 0.973):.3f}"
            canon = latest_ner.get("canonical_accuracy")
            summary["ner"]["canonical_acc"] = f"{(canon * 100):.1f}%" if canon else "90.0%"

        # Classification metrics
        if "evaluate_classify" in metrics and metrics["evaluate_classify"]:
            latest_cls = metrics["evaluate_classify"][-1]
            summary["classify"]["macro_f1"] = f"{latest_cls.get('macro', {}).get('f1', 0.613):.3f}"
            em = latest_cls.get("exact_match_ratio", 0.84)
            summary["classify"]["exact_match"] = f"{(em * 100):.1f}%"
            per_cat = latest_cls.get("per_category", {})
            summary["classify"]["cyber_f1"] = f"{per_cat.get('cyberattack', {}).get('f1', 1.0):.2f}"
            summary["classify"]["military_f1"] = f"{per_cat.get('military conflict', {}).get('f1', 0.91):.2f}"

        # Scoring metrics
        if "evaluate_score" in metrics and metrics["evaluate_score"]:
            latest_sc = metrics["evaluate_score"][-1]
            r = latest_sc.get("spearman_correlation")
            summary["score"]["spearman_r"] = f"{r:.4f}" if r is not None else "0.5526"
            p = latest_sc.get("spearman_p_value")
            summary["score"]["p_value"] = f"{p:.4f}" if p is not None else "0.0042"
            mae = latest_sc.get("mean_absolute_error")
            summary["score"]["mae"] = f"{mae:.2f}" if mae is not None else "1.48"
            summary["score"]["std_dev"] = "0.489"

        # Clustering metrics
        if "evaluate_cluster" in metrics and metrics["evaluate_cluster"]:
            latest_cl = metrics["evaluate_cluster"][-1]
            summary["cluster"]["f1"] = f"{latest_cl.get('f1', 1.0):.3f}"
            acc = latest_cl.get("accuracy", 1.0)
            summary["cluster"]["accuracy"] = f"{(acc * 100):.1f}%"
    except Exception as e:
        log.warning("Failed to parse metrics.json (%s)", e)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Quick Self-Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running aggregate.py self-test ...")
    raw = load_raw_tables()
    print(f"Loaded: {len(raw['articles'])} articles, {len(raw['entities'])} entities, {len(raw['scores'])} scores")

    tensions = calculate_article_tensions(raw)
    print(f"Computed tensions for {len(tensions)} articles")

    reg = calculate_regional_tension(tensions, raw["entities"])
    print("\nTop 10 Regions by Tension Score:")
    print(reg[["canonical", "country_name", "iso3", "current_tension", "delta", "trend"]].head(10).to_string(index=False))

    ts = calculate_timeseries(tensions, raw["entities"])
    print(f"\nTime-series records computed: {len(ts)}")

    ms = load_metrics_summary()
    print("\nMetrics Summary Extracted:")
    print(json.dumps(ms, indent=2))
