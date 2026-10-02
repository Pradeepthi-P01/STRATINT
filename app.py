"""
OSINT Threat Intelligence Aggregator - Streamlit Dashboard
Phase 6: Interactive Geopolitical Crisis & Threat Intelligence Dashboard
"""

import copy
import html as _html
import json
from datetime import datetime, timezone
from pathlib import Path
import folium
from folium.features import GeoJsonTooltip
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from aggregate import (
    calculate_article_tensions,
    calculate_regional_tension,
    calculate_timeseries,
    load_metrics_summary,
    load_raw_tables,
    CATEGORY_WEIGHTS,
    DEFAULT_CATEGORY_WEIGHT,
    LAMBDA_DECAY,
)
import pipeline_runner

# ─────────────────────────────────────────────────────────────────────────────
# 1. Page Configuration & Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="STRATINT | AI Geopolitical Threat Intelligence",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject modern, military/cyber intelligence dark-theme styling
st.markdown(
    """
    <style>
    /* Global font & background enhancements */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Top banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 22px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        background: linear-gradient(90deg, #f8fafc 0%, #38bdf8 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* KPI Cards */
    .kpi-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 10px;
        padding: 16px 20px;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #f1f5f9;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 5px;
    }

    /* Threat Badges */
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-right: 6px;
    }
    .badge-critical { background-color: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .badge-high { background-color: rgba(249, 115, 22, 0.2); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.4); }
    .badge-medium { background-color: rgba(234, 179, 8, 0.2); color: #facc15; border: 1px solid rgba(234, 179, 8, 0.4); }
    .badge-low { background-color: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }

    .category-pill {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 500;
        background: rgba(51, 65, 85, 0.6);
        color: #cbd5e1;
        border: 1px solid rgba(100, 116, 139, 0.3);
        margin-right: 4px;
        margin-bottom: 4px;
    }

    /* Article feed card */
    .article-card {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        transition: border-color 0.2s ease;
    }
    .article-card:hover {
        border-color: rgba(56, 189, 248, 0.35);
    }
    .article-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #f8fafc;
        text-decoration: none;
    }
    .article-meta {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Trend Indicators */
    .trend-rising { color: #f87171; font-weight: 700; }
    .trend-falling { color: #4ade80; font-weight: 700; }
    .trend-stable { color: #94a3b8; font-weight: 700; }

    /* Streamlit tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cached Data Access Functions (@st.cache_data with TTL=300)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_raw_tables() -> dict[str, pd.DataFrame]:
    """Load all 5 SQLite tables with 5-minute cache."""
    return load_raw_tables()


@st.cache_data(ttl=300)
def fetch_article_tensions(_raw_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate decayed tensions for all articles."""
    return calculate_article_tensions(_raw_tables)


@st.cache_data(ttl=300)
def fetch_regional_tension(
    _article_tensions: pd.DataFrame,
    _entities: pd.DataFrame,
) -> pd.DataFrame:
    """Roll up country-level tension scores and 24h trends."""
    return calculate_regional_tension(_article_tensions, _entities)


@st.cache_data(ttl=300)
def fetch_timeseries(
    _article_tensions: pd.DataFrame,
    _entities: pd.DataFrame,
    top_n: int = 6,
) -> pd.DataFrame:
    """Calculate 7-day rolling time-series per crisis region."""
    return calculate_timeseries(_article_tensions, _entities, top_n=top_n)


@st.cache_data(ttl=300)
def fetch_metrics_summary() -> dict:
    """Load model benchmarks from metrics.json."""
    return load_metrics_summary()


@st.cache_data(ttl=300)
def load_world_geojson() -> dict:
    """Load cached world countries GeoJSON."""
    geojson_path = Path("data/world_countries.json")
    if not geojson_path.exists():
        return {}
    # E-1 fix: wrap in try/except so a corrupt file doesn't crash the dashboard.
    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        import logging
        logging.getLogger("app").error("Failed to load world_countries.json: %s", e)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sidebar Controls & Manual Refresh
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.markdown("### 🛰️ **System Controls**")

if st.sidebar.button("📡 Sync Live News (1-Click)", use_container_width=True, type="primary"):
    status_box = st.sidebar.status("🔄 Initializing Live Ingestion...", expanded=True)
    def _ui_callback(msg: str, progress: float):
        status_box.update(label=msg, state="running")

    result = pipeline_runner.run_full_pipeline(limit=5, progress_callback=_ui_callback)
    if result.get("success"):
        status_box.update(label=f"✅ Live sync complete ({result.get('elapsed_seconds', 0)}s)!", state="complete", expanded=False)
        st.cache_data.clear()
        st.toast("Intelligence feed updated with live news!", icon="🛰️")
        time_module_needed = False
        st.rerun()
    else:
        status_box.update(label="❌ Sync encountered an error", state="error", expanded=True)
        st.sidebar.error(f"Error: {result.get('error')}")

col_btn1, col_btn2 = st.sidebar.columns([3, 2])
with col_btn1:
    if st.button("🔄 Reload Cache", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with col_btn2:
    st.caption("TTL: 5 min")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 **Feed & Map Filters**")

# Load base data
raw_data = fetch_raw_tables()
articles_df = raw_data.get("articles", pd.DataFrame())
entities_df = raw_data.get("entities", pd.DataFrame())
classifications_df = raw_data.get("classifications", pd.DataFrame())
scores_df = raw_data.get("scores", pd.DataFrame())
clusters_df = raw_data.get("clusters", pd.DataFrame())

if articles_df.empty:
    st.info("👋 **Welcome to STRATINT!** Your cloud intelligence database is freshly initialized and ready for its first intelligence cycle.")
    if st.button("🚀 Ingest & Process Initial Intelligence Feed Now", type="primary", use_container_width=True):
        status_container = st.status("🔄 Running initial intelligence pipeline...", expanded=True)
        def _onboard_cb(msg: str, progress: float):
            status_container.update(label=msg, state="running")
        res = pipeline_runner.run_full_pipeline(limit=5, progress_callback=_onboard_cb)
        if res.get("success"):
            status_container.update(label=f"✅ Initial intelligence sync complete ({res.get('elapsed_seconds', 0)}s)!", state="complete")
            st.cache_data.clear()
            st.toast("Initial intelligence loaded!", icon="🚀")
            st.rerun()
        else:
            status_container.update(label="❌ Ingestion failed", state="error")
            st.error(f"Error: {res.get('error')}")
    st.stop()

# Compute calculated datasets
article_tensions = fetch_article_tensions(raw_data)
regional_tensions = fetch_regional_tension(article_tensions, entities_df)
timeseries_df = fetch_timeseries(article_tensions, entities_df)
metrics_summary = fetch_metrics_summary()
world_geojson = load_world_geojson()

# Sidebar: Threat Category Filter
available_categories = sorted(list(CATEGORY_WEIGHTS.keys()))
selected_categories = st.sidebar.multiselect(
    "Threat Categories",
    options=available_categories,
    default=available_categories,
    help="Filter threat intelligence articles and map overlays by classified threat category.",
)

# Sidebar: Region / Country Filter
all_countries = []
if not regional_tensions.empty:
    all_countries = sorted(regional_tensions["country_name"].dropna().unique().tolist())

selected_regions = st.sidebar.multiselect(
    "Focus Regions / Countries",
    options=all_countries,
    default=[],
    help="Leave empty to display all global regions, or select specific hotspots.",
)

# Sidebar: Minimum Escalation Score
min_score_filter = st.sidebar.slider(
    "Min Escalation Score (0-10)",
    min_value=0.0,
    max_value=10.0,
    value=0.0,
    step=0.5,
    help="Filter articles with Gemini escalation score >= threshold.",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Decay Parameters:**
    - $\\lambda = 0.05$ (Half-life $\\approx 13.9$ hrs)
    - Weights: Military (1.5), Cyber (1.2), Terror (1.3), Civil (1.0), Diplomatic (0.8), Disaster (0.6)
    """
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Hero Header & Top KPI Metrics
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div class="hero-banner">
        <div class="hero-title">🌐 STRATINT <span style="font-weight: 300; color: #94a3b8; margin: 0 8px;">—</span> AI Threat Intelligence Aggregator</div>
        <div class="hero-subtitle">
            Autonomous multi-source OSINT monitoring &bull; Zero-shot classification &bull; Gemini-1.5 escalation scoring &bull; DBSCAN clustering
            &bull; <span style="color: #38bdf8;">Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top KPI metrics
total_articles = len(articles_df)
total_entities = len(entities_df)
total_clusters = len(clusters_df[clusters_df["cluster_id"] != -1]["cluster_id"].unique()) if not clusters_df.empty else 0
top_crisis_country = "N/A"
top_crisis_score = 0.0
top_crisis_trend = ""

if not regional_tensions.empty:
    top_row = regional_tensions.iloc[0]
    top_crisis_country = top_row["country_name"]
    top_crisis_score = top_row["current_tension"]
    top_crisis_trend = top_row["trend"]

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Monitored Ingest</div>
            <div class="kpi-value">{total_articles:,}</div>
            <div class="kpi-sub">Total intelligence feeds processed</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Recognized Entities</div>
            <div class="kpi-value">{total_entities:,}</div>
            <div class="kpi-sub">GPE, ORG, NORP with ISO normalization</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Active Event Clusters</div>
            <div class="kpi-value">{total_clusters}</div>
            <div class="kpi-sub">DBSCAN grouped events (&epsilon;=0.3)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c4:
    trend_class = "trend-rising" if "Rising" in top_crisis_trend else "trend-falling" if "Falling" in top_crisis_trend else "trend-stable"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Top Crisis Hotspot</div>
            <div class="kpi-value" style="font-size: 1.45rem;">{top_crisis_country}</div>
            <div class="kpi-sub">Tension: <b>{top_crisis_score:.2f}</b> | <span class="{trend_class}">{top_crisis_trend}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Interactive Folium Choropleth Map & Plotly Timeseries
# ─────────────────────────────────────────────────────────────────────────────

col_map, col_trend = st.columns([3, 2])

with col_map:
    st.markdown("#### 🗺️ **Global Geopolitical Tension Map**")
    st.caption("Color intensity represents decay-adjusted regional tension score $\\sum (\\text{score}_i \\times w_c \\times e^{-\\lambda \\Delta t})$. Unmonitored/quiet regions shown in grey.")

    # Create Folium base map
    m = folium.Map(
        location=[20, 10],
        zoom_start=2,
        tiles="CartoDB positron",
        min_zoom=1,
        max_zoom=7,
    )

    # P-4 fix: deep-copy the cached GeoJSON dict before mutating it.
    # The @st.cache_data decorator returns a shared reference — mutating it
    # in-place would permanently corrupt the cache on subsequent renders.
    geojson_data = copy.deepcopy(world_geojson) if world_geojson else {}

    if not regional_tensions.empty and geojson_data:
        # Prepare lookup dict for tooltip
        tension_lookup = {}
        for _, row in regional_tensions.iterrows():
            tension_lookup[row["iso3"]] = {
                "country": row["country_name"],
                "tension": f"{row['current_tension']:.2f}",
                "delta": f"{row['delta']:+.2f}",
                "trend": row["trend"],
                "articles": int(row["article_count"]),
            }

        # Add properties to GeoJSON copy for rich tooltip
        for feature in geojson_data.get("features", []):
            cid = feature.get("id")
            if cid in tension_lookup:
                feature["properties"]["tension_score"] = tension_lookup[cid]["tension"]
                feature["properties"]["trend_status"] = tension_lookup[cid]["trend"]
                feature["properties"]["delta_24h"] = tension_lookup[cid]["delta"]
                feature["properties"]["article_count"] = tension_lookup[cid]["articles"]
                feature["properties"]["display_name"] = tension_lookup[cid]["country"]
            else:
                feature["properties"]["tension_score"] = "0.00"
                feature["properties"]["trend_status"] = "No Active Alerts"
                feature["properties"]["delta_24h"] = "+0.00"
                feature["properties"]["article_count"] = 0
                feature["properties"]["display_name"] = feature.get("properties", {}).get("name", cid)

        # Choropleth layer
        folium.Choropleth(
            geo_data=geojson_data,
            name="Crisis Tension",
            data=regional_tensions,
            columns=["iso3", "current_tension"],
            key_on="feature.id",
            fill_color="YlOrRd",
            fill_opacity=0.75,
            line_opacity=0.3,
            line_color="#475569",
            nan_fill_color="#1e293b",
            nan_fill_opacity=0.45,
            legend_name="Tension Score (Decayed)",
        ).add_to(m)

        # Tooltip layer
        folium.GeoJson(
            geojson_data,
            style_function=lambda x: {"fillColor": "transparent", "color": "transparent"},
            highlight_function=lambda x: {"weight": 2, "color": "#38bdf8", "fillOpacity": 0.3},
            tooltip=GeoJsonTooltip(
                fields=["display_name", "tension_score", "trend_status", "delta_24h", "article_count"],
                aliases=["Country:", "Tension Score:", "24h Trend:", "24h Delta:", "Articles:"],
                style="background-color: #0f172a; color: #f8fafc; font-family: Inter; font-size: 12px; padding: 10px; border-radius: 6px; border: 1px solid #334155;",
            ),
        ).add_to(m)

    # Render map
    st_folium(m, width="100%", height=460, returned_objects=[])

with col_trend:
    st.markdown("#### 📈 **7-Day Regional Tension Trends**")
    st.caption("Daily tension trajectory for top crisis regions over rolling window.")

    if not timeseries_df.empty:
        # Filter timeseries if regions selected in sidebar
        ts_data = timeseries_df
        if selected_regions:
            ts_data = ts_data[ts_data["country"].isin(selected_regions)]

        if ts_data.empty:
            ts_data = timeseries_df

        fig = px.line(
            ts_data,
            x="pub_date",
            y="daily_tension",
            color="country",
            markers=True,
            labels={"daily_tension": "Daily Base Tension", "pub_date": "Date", "country": "Region"},
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.4)",
            margin=dict(l=20, r=20, t=20, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            height=435,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No time-series data available.")

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Detailed Tabs: Article Feed, Clusters, Model Benchmarks, Hotspots
# ─────────────────────────────────────────────────────────────────────────────

tab_articles, tab_clusters, tab_hotspots, tab_metrics = st.tabs([
    "📰 High-Threat Article Feed",
    "🔗 Event Clusters (DBSCAN)",
    "🌍 Regional Crisis Ranking",
    "📊 AI Model Benchmarks (metrics.json)",
])


# ── TAB 1: High-Threat Article Feed ──
# D-4 fix: define match_cat at module level (not inside the render loop) so
# it can be reused and is clearly scoped.
def _match_cat(cats, selected):
    """Return True if any category in cats is in selected."""
    if not isinstance(cats, list):
        return False
    return any(c in selected for c in cats)


with tab_articles:
    st.markdown("##### 🚨 Top Escalation Threat Articles")
    st.caption("Showing articles sorted by escalation score and recency. Filterable by category, region, and minimum score.")

    # Filter articles
    filtered_articles = article_tensions.copy()

    # Apply category filter
    if selected_categories:
        filtered_articles = filtered_articles[
            filtered_articles["category"].apply(lambda c: _match_cat(c, selected_categories))
        ]

    # Apply region filter
    # C-1 fix: entities_df column is "text" (the surface mention), not "name".
    if selected_regions and not entities_df.empty:
        matching_entities = entities_df[
            entities_df["text"].isin(selected_regions)
            | entities_df["canonical"].isin(selected_regions)
        ]
        matched_aids = set(matching_entities["article_id"].unique())
        filtered_articles = filtered_articles[filtered_articles["id"].isin(matched_aids)]

    # Apply min score filter
    filtered_articles = filtered_articles[filtered_articles["score"] >= min_score_filter]

    # Sort by score descending, then decayed tension descending; cap at 20 articles
    filtered_articles = filtered_articles.sort_values(
        by=["score", "decayed_tension"], ascending=[False, False]
    ).head(20)

    if filtered_articles.empty:
        st.info("No articles match the current filter criteria.")
    else:
        for _, art in filtered_articles.iterrows():
            score = float(art.get("score", 5.0))
            if score >= 8.0:
                badge_class = "badge-critical"
                status_label = "CRITICAL"
            elif score >= 6.0:
                badge_class = "badge-high"
                status_label = "HIGH"
            elif score >= 4.0:
                badge_class = "badge-medium"
                status_label = "ELEVATED"
            else:
                badge_class = "badge-low"
                status_label = "ROUTINE"

            categories = art.get("category", [])
            # S-1 fix: escape category names before embedding in HTML
            cat_pills = "".join(
                [f'<span class="category-pill">{_html.escape(str(c))}</span>' for c in categories]
            )

            pub_str = str(art.get("published", art.get("ingested_at", "Unknown")))
            title = art.get("title", "Untitled") or "Untitled"
            source = art.get("source", "Unknown Feed") or "Unknown Feed"
            link = art.get("url", "#") or "#"
            raw_reasoning = art.get("reasoning", "")

            # S-1 fix: sanitize URL to only allow http/https (prevent javascript: etc.)
            if not str(link).startswith(("http://", "https://")):
                link = "#"

            # A-1 fix: use pd (already imported at top) instead of re-importing pandas inline
            if raw_reasoning is None or (isinstance(raw_reasoning, float) and pd.isna(raw_reasoning)) or str(raw_reasoning).strip().lower() in ("", "nan", "none", "null"):
                reasoning = ""
                reasoning_display = "Reasoning not available"
            else:
                reasoning = str(raw_reasoning).strip()
                reasoning_display = reasoning

            cluster_id = art.get("cluster_id", -1)

            # S-1 fix: html.escape all user-derived strings before injecting into HTML
            title_safe   = _html.escape(str(title))
            source_safe  = _html.escape(str(source))
            pub_safe     = _html.escape(str(pub_str))
            reason_safe  = _html.escape(str(reasoning_display))
            cluster_label = f"#{cluster_id}" if cluster_id != -1 else "Noise"

            st.markdown(
                f"""
                <div class="article-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <span class="badge {badge_class}">{status_label} &bull; {score:.1f}/10</span>
                            {cat_pills}
                        </div>
                        <span style="font-size: 0.75rem; color: #94a3b8;">Cluster: {cluster_label}</span>
                    </div>
                    <div style="margin-top: 8px;">
                        <a href="{link}" target="_blank" rel="noopener noreferrer" class="article-title">{title_safe}</a>
                    </div>
                    <div class="article-meta">
                        Source: <b>{source_safe}</b> &bull; Published: {pub_safe} &bull; Decayed Tension Contribution: <b>{art.get('decayed_tension', 0.0):.2f}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander(f"Gemini Escalation Assessment: {title[:60]}..."):
                st.write(f"**Escalation Score:** {score}/10")
                if reasoning:
                    st.write(f"**Gemini Reasoning:** {reason_safe}")
                else:
                    st.caption("Reasoning not available — article was not individually scored by Gemini.")
                st.write(f"**Category Multiplier:** {art.get('cat_weight', 1.0)}x")


# ── TAB 2: Event Clusters (DBSCAN) ──
with tab_clusters:
    st.markdown("##### 🔗 DBSCAN Event Clusters (all-MiniLM-L6-v2, &epsilon;=0.3)")
    st.caption("Articles grouped into distinct geopolitical crisis events using semantic sentence embeddings.")

    if clusters_df.empty:
        st.info("No clustering data available.")
    else:
        clustered_articles = article_tensions[article_tensions["cluster_id"] != -1].copy()
        unique_clusters = sorted(clustered_articles["cluster_id"].unique().tolist())

        col_c_metric1, col_c_metric2, col_c_metric3 = st.columns(3)
        with col_c_metric1:
            st.metric("Total Event Clusters", len(unique_clusters))
        with col_c_metric2:
            st.metric("Articles in Clusters", len(clustered_articles))
        with col_c_metric3:
            noise_count = len(article_tensions[article_tensions["cluster_id"] == -1])
            st.metric("Noise Articles (Isolated)", noise_count)

        st.markdown("---")

        for cid in unique_clusters:
            group = clustered_articles[clustered_articles["cluster_id"] == cid]
            avg_score = group["score"].mean()
            sources = ", ".join(group["source"].unique().tolist())
            first_title = group.iloc[0]["title"]

            with st.expander(f"📍 Event #{cid}: {first_title} ({len(group)} articles, Avg Escalation: {avg_score:.1f}/10)"):
                st.caption(f"Sources reporting: {sources}")
                for _, a in group.iterrows():
                    score_val = a.get("score", 5.0)
                    item_link = a.get("url", a.get("link", "#"))
                    item_title = a.get("title", "Untitled")
                    item_source = a.get("source", "N/A")
                    item_date = a.get("published", "N/A")
                    st.markdown(
                        f"""
                        - **[{item_title}]({item_link})**
                          *(Score: {score_val:.1f} | Source: {item_source} | Date: {item_date})*
                        """
                    )


# ── TAB 3: Regional Crisis Ranking ──
with tab_hotspots:
    st.markdown("##### 🌍 Country & Regional Threat Leaderboard")
    st.caption("Aggregated tension scores and 24-hour delta indicators across monitored nations.")

    if regional_tensions.empty:
        st.info("No regional tension scores available.")
    else:
        display_df = regional_tensions[[
            "country_name", "canonical", "iso3", "current_tension", "delta", "trend", "article_count", "avg_raw_score"
        ]].copy()

        display_df.columns = [
            "Country", "ISO2", "ISO3", "Decayed Tension", "24h Delta", "24h Trend", "Article Count", "Avg Gemini Score"
        ]

        display_df["Decayed Tension"] = display_df["Decayed Tension"].map("{:.2f}".format)
        display_df["24h Delta"] = display_df["24h Delta"].map("{:+.2f}".format)
        display_df["Avg Gemini Score"] = display_df["Avg Gemini Score"].map("{:.1f}".format)

        st.dataframe(display_df, use_container_width=True, height=450)


# ── TAB 4: AI Model Benchmarks ──
with tab_metrics:
    st.markdown("##### 🔬 Pipeline Evaluation & Component Benchmarks")
    st.caption("Quantitative performance metrics extracted from metrics.json across all 5 operational phases.")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.markdown("###### 🏷️ Phase 2: NER (spaCy)")
        ner = metrics_summary.get("ner", {})
        st.markdown(
            f"""
            - **Macro F1:** `{ner.get('f1', 'N/A')}`
            - **Precision:** `{ner.get('precision', 'N/A')}`
            - **Recall:** `{ner.get('recall', 'N/A')}`
            - **ISO Canonical Acc:** `{ner.get('canonical_acc', 'N/A')}`
            """
        )

    with m_col2:
        st.markdown("###### 🎯 Phase 3: Zero-Shot (BART)")
        cls_m = metrics_summary.get("classify", {})
        st.markdown(
            f"""
            - **Macro F1:** `{cls_m.get('macro_f1', 'N/A')}`
            - **Exact Match:** `{cls_m.get('exact_match', 'N/A')}`
            - **Cyberattack F1:** `{cls_m.get('cyber_f1', 'N/A')}`
            - **Military Conflict F1:** `{cls_m.get('military_f1', 'N/A')}`
            """
        )

    with m_col3:
        st.markdown("###### ⚖️ Phase 4: Scoring (Gemini)")
        scr = metrics_summary.get("score", {})
        st.markdown(
            f"""
            - **Spearman $r_s$:** `{scr.get('spearman_r', 'N/A')}`
            - **p-value:** `{scr.get('p_value', 'N/A')}`
            - **MAE:** `{scr.get('mae', 'N/A')} pts`
            - **Consistency $\\sigma$:** `{scr.get('std_dev', 'N/A')}`
            """
        )

    with m_col4:
        st.markdown("###### 🧬 Phase 5: DBSCAN (MiniLM)")
        clu = metrics_summary.get("cluster", {})
        st.markdown(
            f"""
            - **Pairwise F1:** `{clu.get('f1', 'N/A')}`
            - **Clustering Acc:** `{clu.get('accuracy', 'N/A')}`
            - **Distinct Clusters:** `{clu.get('clusters', 'N/A')}`
            - **Clustered Articles:** `{clu.get('articles_clustered', 'N/A')}`
            """
        )

    st.markdown("---")
    st.markdown("###### 📐 System Architecture & Data Flow")
    st.markdown(
        """
        ```text
        [10 RSS Feeds] ──> [SQLite: articles]
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
        [Phase 2: spaCy NER]   [Phase 3: BART Zero-Shot] [Phase 5: MiniLM DBSCAN]
          │ (GPE/ORG/NORP)        │ (6 Threat Labels)     │ (Event Clustering)
          ▼                       ▼                       ▼
        [SQLite: entities]     [SQLite: classifications]  [SQLite: clusters]
                                  │
                                  ▼
                        [Phase 4: Gemini-1.5-Flash]
                                  │ (0-10 Escalation Rubric)
                                  ▼
                        [SQLite: scores]
                                  │
                                  ▼
        [Phase 6: aggregate.py ──> Streamlit app.py (Folium Choropleth + Plotly)]
        ```
        """
    )
