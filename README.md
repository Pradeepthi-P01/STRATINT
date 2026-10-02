# 🌐 STRATINT — AI Geopolitical Threat Intelligence Aggregator

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Database](https://img.shields.io/badge/database-SQLite%20WAL-lightgrey.svg)](https://www.sqlite.org/)
[![NER](https://img.shields.io/badge/NER-spaCy%20v3-09a3d5.svg)](https://spacy.io/)
[![Zero--Shot](https://img.shields.io/badge/classifier-BART--large--MNLI-yellow.svg)](https://huggingface.co/facebook/bart-large-mnli)
[![LLM Scoring](https://img.shields.io/badge/LLM-Gemini--1.5--Flash-4285F4.svg)](https://ai.google.dev/)
[![Clustering](https://img.shields.io/badge/clustering-Sentence--Transformers%20%2B%20DBSCAN-green.svg)](https://www.sbert.net/)
[![UI](https://img.shields.io/badge/dashboard-Streamlit%20%2B%20Folium-FF4B4B.svg)](https://streamlit.io/)

An automated, end-to-end Open-Source Intelligence (OSINT) processing pipeline and real-time tactical situational awareness dashboard. STRATINT continuously ingests multi-source global wire feeds, extracts recognized entities, categorizes threat vectors, computes LLM-calibrated escalation scores, clusters related reports into discrete crisis events, and models regional tension using exponential time-decay mathematics.

---

## 📌 Table of Contents

- [Overview & Capabilities](#-overview--capabilities)
- [System Architecture](#-system-architecture)
- [Database Schema (SQLite WAL)](#-database-schema-sqlite-wal)
- [Monitored Feed Sources](#-monitored-feed-sources)
- [Pipeline Phases & Technical Implementation](#-pipeline-phases--technical-implementation)
  - [Phase 1: Ingestion & Deduplication (`ingest.py`, `db.py`)](#phase-1-ingestion--deduplication-ingestpy-dbpy)
  - [Phase 2: Named Entity Recognition & Geopolitical Normalization (`ner.py`)](#phase-2-named-entity-recognition--geopolitical-normalization-nerpy)
  - [Phase 3: Multi-Label Zero-Shot Threat Classification (`classify.py`)](#phase-3-multi-label-zero-shot-threat-classification-classifypy)
  - [Phase 4: LLM Escalation Scoring & Reasoning (`score.py`)](#phase-4-llm-escalation-scoring--reasoning-scorepy)
  - [Phase 5: Event Clustering & Corroboration (`cluster.py`)](#phase-5-event-clustering--corroboration-clusterpy)
  - [Phase 6: Mathematical Tension Decay & Analytics Engine (`aggregate.py`)](#phase-6-mathematical-tension-decay--analytics-engine-aggregatepy)
  - [Phase 6: Tactical Intelligence Dashboard (`app.py`)](#phase-6-tactical-intelligence-dashboard-apppy)
- [Mathematical Tension Formulation](#-mathematical-tension-formulation)
- [Quantitative Evaluation Benchmarks](#-quantitative-evaluation-benchmarks)
- [CLI Reference & Script Options](#-cli-reference--script-options)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Execution Guide](#-execution-guide)
- [Evaluation CLI](#-evaluation-cli)
- [Production Hardening & Reliability Measures](#-production-hardening--reliability-measures)
- [Troubleshooting & FAQ](#-troubleshooting--faq)

---

## 🚀 Overview & Capabilities

Modern intelligence analysts face an overwhelming volume of open-source reporting. STRATINT automates triage and geopolitical situational assessment through a robust 6-stage NLP/ML pipeline:

1. **Continuous Multi-Source Ingestion**: Ingests 10 international news wire and cybersecurity RSS/Atom feeds with deterministic URL hashing and 100% deduplication.
2. **Geopolitical Entity Extraction**: Extracts GPE (geopolitical entities), ORG (organizations), and NORP (nationalities/religious/political groups) with ISO-3166-1 alpha-2/alpha-3 canonical normalization and regional bloc standardization (`NATO`, `EU`, `BRICS`, etc.).
3. **Zero-Shot Threat Taxonomy**: Classifies articles into 6 operational threat domains using NLI transformer models with multi-label support.
4. **Calibrated Escalation Assessment**: Evaluates crisis severity on a 0–10 scale using Google's `gemini-1.5-flash` model with few-shot anchor reasoning, strict rate-limiting (14 req/min), and 429 quota backoff handling.
5. **Semantic Event Clustering**: Detects breaking events and groups disparate reporting using dense sentence embeddings (`all-MiniLM-L6-v2`) and DBSCAN ($\varepsilon=0.3$, cosine distance).
6. **Mathematical Decay Modeling**: Aggregates country-level tension scores with an exponential half-life model ($\lambda = 0.05$, $t_{1/2} \approx 13.86\text{h}$) and calculates 24-hour crisis trajectory deltas (Rising 🔺, Falling 🔻, Stable ➖).
7. **Tactical Command Dashboard**: Renders interactive choropleth tension heatmaps (using clean `CartoDB positron` tiles), 7-day trend trajectories, event cluster drilldowns, and a live high-threat article feed.

---

## 🏗️ System Architecture

```text
                                [10 RSS / Atom Wire Feeds]
                                             │
                                             ▼
                                   [ Phase 1: Ingestion ]
                                  (SHA-256 Deduplication)
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │   SQLite DB (WAL Mode)    │
                               │      table: articles      │
                               └─────────────┬─────────────┘
                                             │
                 ┌───────────────────────────┼───────────────────────────┐
                 ▼                           ▼                           ▼
        [ Phase 2: spaCy NER ]     [ Phase 3: BART-MNLI ]     [ Phase 5: MiniLM DBSCAN ]
       (en_core_web_lg/trf)        (Zero-Shot Multi-Label)     (all-MiniLM-L6-v2)
        • GPE / ORG / NORP          • 6 Threat Categories       • ε=0.3 Cosine Distance
        • ISO-3166-1 Normalizer     • Confidence Threshold      • Dense Vector Cache
                 │                           │                           │
                 ▼                           ▼                           ▼
         table: entities             table: classifications      table: clusters
                                             │
                                             ▼
                                  [ Phase 4: Gemini LLM ]
                                 (gemini-1.5-flash Pinned)
                                  • 0–10 Calibrated Rubric
                                  • 14 req/min Rate Limiter
                                  • 429 Quota Auto-Backoff
                                             │
                                             ▼
                                       table: scores
                                             │
                                             ▼
                            [ Phase 6: Analytical Engine (aggregate.py) ]
                           Tension = Σ Score_i × Weight_c × exp(-λ × Δt_i)
                                             │
                                             ▼
                         [ Phase 6: Streamlit UI Dashboard (app.py) ]
                           • CartoDB Positron Choropleth Map
                           • Plotly 7-Day Regional Trend Trajectories
                           • Live High-Threat Article Feed
                           • Event Cluster Cards & Metrics Explorer
```

---

## 🗄️ Database Schema (SQLite WAL)

The project uses a unified SQLite database (`osint.db`) configured with:
- **Write-Ahead Logging (`WAL`)**: Enables non-blocking concurrent reads by Streamlit while background scripts write.
- **Foreign Key Enforcement**: `PRAGMA foreign_keys = ON;` with `ON DELETE CASCADE`.
- **Page Cache Tuning**: `PRAGMA cache_size = -16000;` (16 MB in-memory cache) for instant query responsiveness.

```sql
-- ── Core article store ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS articles (
    id          TEXT PRIMARY KEY,   -- SHA-256(url)
    url         TEXT UNIQUE NOT NULL,
    title       TEXT,
    summary     TEXT,               -- RSS <description> or <summary>
    source      TEXT,               -- Feed source name (e.g., "BBC World")
    published   TEXT,               -- ISO-8601 UTC timestamp
    ingested_at TEXT NOT NULL       -- Wall-clock ingestion timestamp
);

-- ── Phase 2: spaCy NER entities ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS entities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id  TEXT NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    text        TEXT NOT NULL,      -- Surface mention (e.g., "Ukraine")
    label       TEXT NOT NULL,      -- GPE | ORG | NORP
    canonical   TEXT,               -- ISO-3166-1 alpha-2 (e.g., "UA") or BLOC:xxx
    start_char  INTEGER,            -- Character offset in title+summary
    end_char    INTEGER
);

-- ── Phase 3: Zero-shot multi-label classification ─────────────────────────
CREATE TABLE IF NOT EXISTS classifications (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id  TEXT NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    category    TEXT NOT NULL,      -- One of 6 threat categories
    confidence  REAL NOT NULL,      -- Model confidence [0.0, 1.0]
    UNIQUE (article_id, category)
);

CREATE TABLE IF NOT EXISTS classified_articles (
    article_id    TEXT PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
    classified_at TEXT NOT NULL
);

-- ── Phase 4: Gemini escalation scores ────────────────────────────────────
CREATE TABLE IF NOT EXISTS scores (
    article_id  TEXT PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
    score       REAL NOT NULL,      -- Escalation rating [0.0 - 10.0]
    reasoning   TEXT                -- One-sentence rationale from Gemini
);

-- ── Phase 5: DBSCAN event clusters ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS clusters (
    article_id  TEXT PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
    cluster_id  INTEGER NOT NULL    -- -1 = DBSCAN noise (isolated report)
);

-- ── Indices for optimized dashboard query performance ────────────────────
CREATE INDEX IF NOT EXISTS idx_entities_article   ON entities(article_id);
CREATE INDEX IF NOT EXISTS idx_entities_label     ON entities(label);
CREATE INDEX IF NOT EXISTS idx_class_article      ON classifications(article_id);
CREATE INDEX IF NOT EXISTS idx_articles_source    ON articles(source);
CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published);
CREATE INDEX IF NOT EXISTS idx_scores_score       ON scores(score);
CREATE INDEX IF NOT EXISTS idx_clusters_cluster   ON clusters(cluster_id);
```

---

## 📡 Monitored Feed Sources

STRATINT monitors 10 curated global wires and cybersecurity threat reporting endpoints:

| Source | Category / Scope | Feed URL |
|---|---|---|
| **France 24 English** | Global wire / International | `https://www.france24.com/en/rss` |
| **NPR World** | Global wire / Diplomacy & Conflict | `https://feeds.npr.org/1004/rss.xml` |
| **BBC World** | Global wire / World news | `http://feeds.bbci.co.uk/news/world/rss.xml` |
| **Al Jazeera** | Regional (Middle East & Global South) | `https://www.aljazeera.com/xml/rss/all.xml` |
| **The Hindu** | Regional (South Asia & Indo-Pacific) | `https://www.thehindu.com/news/international/feeder/default.rss` |
| **South China Morning Post** | Regional (East Asia & China) | `https://www.scmp.com/rss/91/feed` |
| **DW World (Deutsche Welle)**| Regional (Eastern Europe & EU) | `https://rss.dw.com/atom/rss-en-world` |
| **The Guardian World** | Global wire / Geopolitics | `https://www.theguardian.com/world/rss` |
| **Krebs on Security** | Cyber threat intelligence & Zero-days | `https://krebsonsecurity.com/feed/` |
| **The Hacker News** | Cyber attacks, vulnerabilities & APTs | `https://feeds.feedburner.com/TheHackersNews` |

---

## ⚙️ Pipeline Phases & Technical Implementation

### Phase 1: Ingestion & Deduplication (`ingest.py`, `db.py`)
- **Feed Parsing**: Uses `feedparser` with a 15-second network socket timeout wrapper to prevent hangs from stalled feeds.
- **Deduplication Engine**: Uses the SHA-256 hash of the canonical article URL as the primary key. Re-running ingestion is fully idempotent and yields a 100% deduplication rate without duplicating records.
- **Date Normalization**: Parses diverse RFC-822 and ISO-8601 published timestamps into standardized UTC ISO-8601 strings (`YYYY-MM-DDTHH:MM:SSZ`).
- **Telemetry**: Appends run statistics to `metrics.json` under `"ingest"`.

### Phase 2: Named Entity Recognition & Geopolitical Normalization (`ner.py`)
- **Model Engine**: spaCy transformer-compatible pipeline (`en_core_web_lg` default, with seamless support for `en_core_web_trf`).
- **Target Entities**:
  - `GPE`: Countries, sovereign states, cities, territories.
  - `ORG`: Supranational bodies, military formations, defense contractors, government ministries.
  - `NORP`: Nationalities, religious groups, political factions.
- **Canonical Geopolitical Resolver**:
  - Direct exact and fuzzy matching against ISO-3166-1 alpha-2 codes via `pycountry`.
  - Regional bloc resolver for multilateral alliances (`NATO`, `EU`, `BRICS`, `ASEAN`, `G7`, `UN`, `AU`, `OPEC`) standardized as `BLOC:<NAME>`.
  - Nationality-to-country mapping (e.g., *"Ukrainian"* $\to$ `"UA"`, *"Chinese"* $\to$ `"CN"`, *"Russian"* $\to$ `"RU"`).

### Phase 3: Multi-Label Zero-Shot Threat Classification (`classify.py`)
- **Model Engine**: HuggingFace `facebook/bart-large-mnli` running natural language inference.
- **Hypothesis Formulation**: Evaluates premise text with `"This news article is about {label}."`
- **6 Operational Threat Categories**:
  1. `military conflict` (Multiplier: 1.5x)
  2. `terrorism or extremist violence` (Multiplier: 1.3x)
  3. `cyberattack` (Multiplier: 1.2x)
  4. `civil unrest` (Multiplier: 1.0x)
  5. `diplomatic tension` (Multiplier: 0.8x)
  6. `natural disaster` (Multiplier: 0.6x)
- **Multi-Label Thresholding**: Articles are evaluated across all categories independently. Any category scoring $\ge 0.45$ (configurable via `--threshold`) is persisted in `classifications`.

### Phase 4: LLM Escalation Scoring & Reasoning (`score.py`)
- **Model Engine**: Pinned strictly to **`gemini-1.5-flash`** via Google's `google.genai` SDK.
- **Threat Escalation Rubric (0–10 Scale)**:
  - `0–2`: Routine/background news (diplomatic courtesies, peaceful trade talks, cultural events).
  - `3–4`: Elevated concern, no immediate violence (harsh rhetoric, sanctions, cyber espionage, peaceful protests).
  - `5–6`: Active conflict or disruption (artillery exchanges, infrastructure cyberattacks, border clashes).
  - `7–8`: Major escalation, casualties reported (missile strikes causing deaths, assassinations, emergencies).
  - `9–10`: Catastrophic/regional war-level event (multi-nation invasion, nuclear threat, collapse of government in war).
- **Prompt Engineering**: Uses few-shot anchor calibration examples (routine logistics $\to$ 1, Gitea zero-day cyberattack $\to$ 5, kinetic civilian strike $\to$ 8) and enforces a strict JSON schema: `{"score": <int>, "reasoning": "<one concise sentence>"}`.
- **Rate-Limiting & Quota Hardening**:
  - Sliding-window throttle capping requests at **14 req/min** (safe buffer under the 15 req/min free-tier cap).
  - Automatic HTTP 429 (`RESOURCE_EXHAUSTED`) backoff: waits 60 seconds and retries once.
  - Quota-exceeded articles are tagged `_QUOTA_EXCEEDED` and left uninserted into `scores` so subsequent runs pick them up without data loss or pipeline interruption.

### Phase 5: Event Clustering & Corroboration (`cluster.py`)
- **Dense Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional semantic vectors over concatenated titles and summaries.
- **Clustering Algorithm**: DBSCAN with cosine distance (`eps=0.3`, `min_samples=2`).
- **Embedding Cache**: Cached in `data/embeddings.npz` to eliminate redundant inference and enable rapid hyperparameter tuning (`--tune-eps`).
- **Noise Rejection**: Uncorroborated, single-outlet articles are tagged as cluster ID `-1` (isolated reports), while multi-outlet crisis reporting is clustered into distinct event IDs.

### Phase 6: Mathematical Tension Decay & Analytics Engine (`aggregate.py`)
- **Tension Decay Computation**: Evaluates time-decayed threat scores for all monitored nations and regional blocs.
- **Disputed Territory & ISO-3 Mapping**: Converts ISO-2 codes to ISO-3 for Folium GeoJSON compatibility with custom overrides for disputed regions (`XK` $\to$ `"KOS"`, `PS` $\to$ `"PSE"`, `TW` $\to$ `"TWN"`, `HK` $\to$ `"HKG"`, `MO` $\to$ `"MAC"`).
- **24-Hour Delta Trajectory**: Measures rate of change over the last 24 hours to output trend status badges.
- **7-Day Rolling Timeseries**: Computes daily baseline tension per country over the past 7 days for trend analysis.

### Phase 6: Tactical Intelligence Dashboard (`app.py`)
Built with Streamlit, custom CSS, Folium, and Plotly:
- **Hero KPI Banner**: Displays total ingested articles, recognized entities, active event clusters, and the #1 crisis hotspot with trend indicator.
- **Global Geopolitical Tension Map**:
  - Folium Choropleth using clean **`CartoDB positron`** basemap tiles (no watermark, free, no API key).
  - Hover tooltips displaying country name, tension score, 24h trend, 24h delta, and reporting volume.
- **Plotly 7-Day Regional Trend Trajectories**: Interactive dark-themed time-series charting tension trajectory per region.
- **4 Operational Tab Views**:
  1. 🚨 **High-Threat Article Feed**: Filterable by threat category, region, and minimum escalation score; shows severity badges (`CRITICAL`, `HIGH`, `ELEVATED`, `ROUTINE`), category pills, cluster tags, and expandable Gemini rationale with sanitized null fallback (`Reasoning not available`).
  2. 🔗 **Event Clusters (DBSCAN)**: Collapsible cards grouping corroborating sources, article links, publication dates, and average cluster escalation ratings.
  3. 🌍 **Regional Crisis Ranking**: Interactive leaderboard table showing Country, ISO2, ISO3, Decayed Tension, 24h Delta, 24h Trend, Article Count, and Avg Gemini Score.
  4. 📊 **AI Model Benchmarks**: Live scorecard extracting quantitative evaluation metrics directly from `metrics.json` alongside an ASCII architecture diagram.

---

## 📐 Mathematical Tension Formulation

The regional tension score is defined by an exponential decay formulation that weights severity by threat domain and discounts stale reporting:

$$\text{Tension}(R, t) = \sum_{i \in \text{Articles}(R)} \text{Score}_i \times w_{c(i)} \times \exp\left(-\lambda \cdot \Delta t_i\right)$$

Where:
- $\text{Score}_i \in [0.0, 10.0]$: The Gemini LLM escalation rating.
- $w_{c(i)}$: Category severity multiplier.
- $\Delta t_i = \max\left(0, \frac{t - t_{\text{published}, i}}{3600}\right)$: Elapsed time in hours since publication (articles with null timestamps are excluded).
- $\lambda = 0.05$: Exponential decay rate constant, yielding a half-life of:
  $$t_{1/2} = \frac{\ln(2)}{\lambda} = \frac{0.69315}{0.05} \approx 13.86 \text{ hours}$$

### Category Multiplier Table ($w_c$)
| Threat Category | Multiplier ($w_c$) | Rationale |
|---|---|---|
| `military conflict` | **1.5x** | Kinetic warfare poses immediate existential and regional stability risks |
| `terrorism or extremist violence` | **1.3x** | High lethality and psychological destabilization |
| `cyberattack` | **1.2x** | Disruption to critical infrastructure and strategic assets |
| `civil unrest` | **1.0x** | Baseline societal friction and public order disruption |
| `diplomatic tension` | **0.8x** | Non-kinetic dispute; potential precursor to escalation |
| `natural disaster` | **0.6x** | Humanitarian impact without direct hostile geopolitical intent |

### 24-Hour Crisis Trajectory Classification
$$\Delta_{24\text{h}} = \text{Tension}(R, t) - \text{Tension}(R, t - 24\text{h})$$

- 🔺 **Rising**: $\Delta_{24\text{h}} \ge +0.50$
- 🔻 **Falling**: $\Delta_{24\text{h}} \le -0.50$
- ➖ **Stable**: $-0.50 < \Delta_{24\text{h}} < +0.50$

---

## 📊 Quantitative Evaluation Benchmarks

Each machine learning component was quantitatively benchmarked against hand-annotated gold-standard test sets in `validation/`:

| Pipeline Phase | Primary Model | Evaluation Metric | Result |
|---|---|---|---|
| **Phase 1: Ingest** | RSS / SHA-256 | Re-run Deduplication Rate | **100.0%** (0 duplicate rows inserted) |
| **Phase 2: NER** | spaCy `en_core_web_lg` | Strict Span Macro F1 | **0.8825** |
| | | Strict Span Macro Precision | **0.8148** |
| | | Strict Span Macro Recall | **0.9733** |
| | | Canonical ISO-3166-1 Accuracy | **90.0%** |
| | | Entity Type Breakdown: | GPE F1: **0.881** \| ORG F1: **0.767** \| NORP F1: **1.000** |
| **Phase 3: Classify** | `facebook/bart-large-mnli` | Exact Match Ratio | **0.8400** |
| | | Cyberattack F1 | **1.0000** (Precision: 1.0, Recall: 1.0) |
| | | Military Conflict F1 | **0.9091** (Precision: 0.83, Recall: 1.0) |
| | | Civil Unrest F1 | **0.6667** |
| | | Diplomatic Tension F1 | **0.6000** |
| | | Terrorism / Extremism F1 | **0.5000** |
| **Phase 4: Score** | `gemini-1.5-flash` | Spearman Rank Correlation ($\rho$) | **0.5526** ($p = 0.00418$, statistically significant) |
| | | Mean Absolute Error (MAE) | **1.48 pts** (on 0–10 scale) |
| **Phase 5: Cluster** | `all-MiniLM-L6-v2` + DBSCAN | Pairwise Cluster Precision | **1.0000** |
| | | Pairwise Cluster Recall | **1.0000** |
| | | Pairwise Cluster F1 | **1.0000** (tested on 20 verified gold pairs) |

---

## 💻 CLI Reference & Script Options

All scripts provide rich command-line interfaces:

### `ingest.py`
```powershell
python ingest.py                     # Full ingestion run across all 10 feeds
python ingest.py --dry-run           # Fetch feeds and display stats table without writing to DB
python ingest.py --limit 5           # Ingest at most 5 entries per feed (smoke-test)
```

### `ner.py`
```powershell
python ner.py                        # Extract entities for all unprocessed articles
python ner.py --limit 50             # Process at most 50 articles
python ner.py --rerun                # Clear entities table and reprocess all articles
python ner.py --generate-validation  # Generate validation/ner_validation.jsonl template
python ner.py --val-size 30          # Set size of generated validation template (default: 25)
```

### `classify.py`
```powershell
python classify.py                   # Classify all unclassified articles
python classify.py --batch-size 16   # Set HuggingFace inference batch size (default: 8)
python classify.py --threshold 0.40  # Set multi-label confidence threshold (default: 0.45)
python classify.py --limit 20        # Test on 20 articles
python classify.py --rerun           # Clear classifications table and reclassify
python classify.py --generate-validation # Generate classification validation JSONL
```

### `score.py`
```powershell
python score.py                      # Score all unscored articles with gemini-1.5-flash
python score.py --limit 10           # Score 10 articles (safe quota test)
python score.py --rerun              # Clear scores table and re-score all articles
python score.py --generate-validation # Generate validation/score_validation.jsonl
python score.py --consistency-check  # Run 3 scoring passes over 10 articles to check standard deviation
```

### `cluster.py`
```powershell
python cluster.py                    # Cluster articles with default eps=0.3
python cluster.py --eps 0.25         # Cluster with custom cosine distance threshold
python cluster.py --min-samples 2    # Set minimum cluster size
python cluster.py --tune-eps         # Grid search eps [0.2, 0.3, 0.4, 0.5] and display silhouette stats
python cluster.py --rerun            # Clear clusters table and recompute
python cluster.py --generate-validation # Generate 20-pair cluster validation template
```

### `aggregate.py`
```powershell
python aggregate.py                  # Run self-test: prints regional tension table & metrics summary
```

---

## 📂 Repository Structure

```text
d:\OSINT_Project\
├── docs/                               # Comprehensive engineering documentation
│   ├── ARCHITECTURE.md                 # System architecture diagrams, ERDs & component specs
│   ├── DECISIONS.md                    # Architectural Decision Records (ADRs)
│   ├── DEPLOYMENT.md                   # Cloud & container deployment instructions
│   ├── EVALUATION.md                   # Benchmark results, metrics & ground-truth specifications
│   ├── PIPELINE_GUIDE.md               # Complete CLI reference and operational manual
│   └── SECURITY.md                     # Threat modeling, XSS mitigation & credential hardening
├── .streamlit/
│   └── config.toml                     # Streamlit theme and server configuration
├── Dockerfile                          # Production container configuration
├── TASKS.md                            # Phased project task tracker & implementation status
├── app.py                              # Streamlit tactical intelligence dashboard
├── pipeline_runner.py                  # 1-Click live news orchestrator with progress reporting
├── run_pipeline.bat                    # 1-Click Windows batch runner for full pipeline sync
├── run_pipeline.ps1                    # 1-Click PowerShell runner for pipeline sync
├── run_dashboard.bat                   # 1-Click Windows launcher for the Streamlit dashboard
├── aggregate.py                        # Tension decay formula, timeseries & analytics engine
├── ingest.py                           # Multi-source RSS feed fetcher & deduplicator
├── ner.py                              # spaCy entity recognition & ISO-3166-1 normalizer
├── classify.py                         # HuggingFace BART zero-shot threat classifier
├── score.py                            # Gemini 1.5 Flash escalation scoring engine
├── cluster.py                          # Sentence-Transformers + DBSCAN event clusterer
├── db.py                               # SQLite WAL schema, indices & connection manager
├── evaluate.py                         # Unified evaluation harness for all phases
├── metrics.json                        # Recorded evaluation runs & benchmark history
├── osint.db                            # SQLite database (articles, entities, scores, etc.)
├── requirements.txt                    # Python dependencies
├── .env.example                        # Environment template for GEMINI_API_KEY
├── .gitignore                          # Ignore rules (.env, osint.db, models, cache)
├── data/
│   ├── world_countries.json            # GeoJSON polygon boundaries for Folium choropleth
│   ├── embeddings.npz                  # Cached compressed dense vector representations
│   └── metrics.json                    # Benchmark snapshot used by the AI Model Benchmarks tab
└── validation/
    ├── ner_validation.jsonl            # Gold-standard annotations for NER evaluation
    ├── classification_validation.jsonl # Gold-standard labels for classification evaluation
    ├── score_validation.jsonl          # Gold-standard escalation ratings evaluation
    └── cluster_validation.jsonl        # Gold-standard pairwise event links evaluation
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.10 or higher
- Git
- Google Gemini API Key ([Google AI Studio](https://aistudio.google.com/))

### 2. Clone and Create Virtual Environment
```powershell
# Clone the repository
git clone <repo-url> OSINT_Project
cd OSINT_Project

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt

# Download the spaCy large English model
python -m spacy download en_core_web_lg
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and insert your Gemini API key:
```ini
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### 5. Initialize the SQLite Database
```powershell
python -c "import db; db.init_db()"
```

---

## 🚦 Execution Guide

### Option A: 1-Click Automated Live Pipeline Sync
Run all 5 stages (`ingest` → `ner` → `classify` → `score` → `cluster`) with a single command or double-click:

```powershell
# Run the automated pipeline orchestrator
python pipeline_runner.py

# Or double-click run_pipeline.bat on Windows
```

### Option B: Step-by-Step Manual Execution
Run each pipeline stage individually in sequence:

```powershell
# Step 1: Ingest 10 RSS feeds into SQLite
python ingest.py

# Step 2: Extract GPE, ORG, and NORP entities
python ner.py

# Step 3: Run zero-shot multi-label threat classification
python classify.py

# Step 4: Compute LLM escalation scores (0–10) via Gemini 1.5 Flash
python score.py

# Step 5: Cluster articles into events using DBSCAN
python cluster.py

# Step 6: Verify mathematical tension aggregation
python aggregate.py
```

### Launch the Tactical Dashboard
```powershell
python -m streamlit run app.py

# Or double-click run_dashboard.bat on Windows
```
Open your browser at `http://localhost:8501`. Inside the dashboard sidebar, you can also click **"📡 Sync Live News (1-Click)"** to fetch and process breaking intelligence on demand without touching the terminal!
Open your browser at `http://localhost:8501`.

---

## 🧪 Evaluation CLI

STRATINT includes a unified evaluation CLI (`evaluate.py`) that benchmarks any stage against the gold-standard test sets in `validation/`:

```powershell
# Evaluate Named Entity Recognition (strict exact span match & canonical accuracy)
python evaluate.py --phase ner

# Evaluate Threat Classification (macro F1, per-class metrics, exact match ratio)
python evaluate.py --phase classify

# Evaluate Gemini Escalation Scoring (Spearman rank correlation & MAE)
python evaluate.py --phase score

# Evaluate Event Clustering (pairwise cluster precision, recall, F1)
python evaluate.py --phase cluster
```

Every evaluation execution automatically appends its timestamped metrics record into `metrics.json`.

---

## 🛡️ Production Hardening & Reliability Measures

- **Tile Layer Watermark Elimination**: Switched Folium tiles from CartoDB Voyager to `CartoDB positron` to permanently eliminate the `"API KEY REQUIRED"` watermark without requiring commercial map tokens.
- **Gemini Free-Tier Protection**:
  - Hard-pinned to `gemini-1.5-flash` (1,500 req/day quota), strictly avoiding `gemini-2.5-flash` which is limited to 20 req/day on the free tier.
  - Implemented a 14 req/min sliding-window throttle (safely beneath the 15 req/min threshold).
  - On HTTP 429 (`RESOURCE_EXHAUSTED`), waits 60s and retries once; if still exhausted, flags article as `_QUOTA_EXCEEDED` and skips insertion so subsequent runs resume cleanly without DB corruption.
- **Sanitized UI Fallbacks**: Replaced raw `nan` / `null` reasoning values in the High-Threat Article Feed with `"Reasoning not available"` when articles have not been individually scored.
- **Temporal Calculation Guard**: Safely filters out articles with `NULL` or malformed `published_at` timestamps before evaluating decay equations, preventing calculation failures.
- **Windows UTF-8 Console Support**: Integrated `sys.stdout.reconfigure(encoding="utf-8")` across all CLI scripts to prevent Unicode encoding crashes with tabular box characters.
- **Complete Pipeline Idempotency**: Verified all database insert operations use primary keys, foreign key cascades, and unique constraints so repeat script executions never corrupt or duplicate records.

---

## ❓ Troubleshooting & FAQ

#### Q: Why did `score.py` report skipped articles?
**A:** `score.py` respects the Google Gemini free-tier rate limits (15 requests/minute). If a 429 quota error is encountered twice in succession, the article is flagged as `_QUOTA_EXCEEDED` and skipped. Re-running `python score.py` will resume scoring only the unscored articles.

#### Q: How do I inspect the SQLite database directly?
**A:** Use the SQLite CLI or Python:
```powershell
python -c "import sqlite3; conn = sqlite3.connect('osint.db'); print(conn.execute('SELECT source, COUNT(*) FROM articles GROUP BY source').fetchall())"
```

#### Q: Can I run `ner.py` with the transformer-based model?
**A:** Yes. Install `en_core_web_trf`:
```powershell
python -m spacy download en_core_web_trf
```
`ner.py` will automatically utilize `en_core_web_trf` if available, falling back to `en_core_web_lg`.

#### Q: How do I refresh data in the dashboard?
**A:** Use the **"🔄 Refresh Intelligence"** button in the sidebar or enable automated rerun intervals in Streamlit.
