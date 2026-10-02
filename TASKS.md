# OSINT Threat Intelligence Aggregator — Tasks Tracker

This document tracks all project tasks across the 8 core engineering phases of the OSINT Threat Intelligence Aggregator pipeline (`STRATINT`).

---

## Task Workflow Standard

Every task follows a strict lifecycle before being marked complete:

```text
TASK-XXX ──► Implement / Verify ──► Test / Evaluate ──► Code Review ──► Mark Complete [x]
```

---

## Phase 1: Database & Ingestion Engine (`db.py`, `ingest.py`)

- [x] **TASK-001**: Design and implement SQLite schema with WAL mode and foreign-key constraints (`articles`, `entities`, `classifications`, `classified_articles`, `scores`, `clusters`).
- [x] **TASK-002**: Implement multi-source RSS feedparser ingestion engine with 10 global/regional security feeds.
- [x] **TASK-003**: Implement SHA-256 canonical URL hashing with `INSERT OR IGNORE` for zero-redundancy deduplication.
- [x] **TASK-004**: Add socket timeout wrapper (15s) and per-feed try/except error isolation to handle unreachable/malformed feeds.
- [x] **TASK-005**: Add CLI flags (`--dry-run`, `--limit <N>`) and append ingest run statistics to `metrics.json`.

---

## Phase 2: Named Entity Recognition & Geopolitical Normalization (`ner.py`)

- [x] **TASK-006**: Integrate spaCy NER pipeline with `en_core_web_lg` (floret vectors) for geographic & actor entity extraction.
- [x] **TASK-007**: Extract and filter target entity types: `GPE` (Geopolitical Entities), `ORG` (Organizations/Militaries), and `NORP` (Nationalities/Religious/Political Groups).
- [x] **TASK-008**: Build canonical entity normalizer mapping surface forms/demonyms/capital metonyms to ISO-3166-1 alpha-2 codes and regional blocs (`BLOC:EU`, `BLOC:NATO`, `BLOC:UN`).
- [x] **TASK-009**: Implement incremental batch ingestion (`nlp.pipe()`, batch size 16) with `--rerun` and `--limit` CLI controls.
- [x] **TASK-010**: Build validation template generator (`--generate-validation`) outputting to `validation/ner_validation.jsonl`.

---

## Phase 3: Zero-Shot Multi-Label Threat Classification (`classify.py`)

- [x] **TASK-011**: Implement zero-shot classification engine using HuggingFace Transformers (`facebook/bart-large-mnli`, CPU-optimized).
- [x] **TASK-012**: Define standard 6-class threat taxonomy: `military conflict`, `cyberattack`, `civil unrest`, `diplomatic tension`, `terrorism or extremist violence`, `natural disaster`.
- [x] **TASK-013**: Configure independent multi-label classification (`multi_label=True`) with 0.45 sigmoid confidence threshold.
- [x] **TASK-014**: Store classified threat labels and confidence distributions in `classifications` table with `(article_id, category)` uniqueness.
- [x] **TASK-015**: Build validation template generator outputting to `validation/classification_validation.jsonl`.

---

## Phase 4: Escalation Scoring & LLM Reasoning (`score.py`)

- [x] **TASK-016**: Implement Google Gemini client using `gemini-1.5-flash` with few-shot anchor reasoning for 0.0–10.0 escalation severity scoring.
- [x] **TASK-017**: Implement defensive token-bucket rate limiter enforcing a strict ceiling of 14 requests/minute.
- [x] **TASK-018**: Implement 429 `RESOURCE_EXHAUSTED` backoff (60s retry) and defensive JSON extraction parsing `{"score": int, "reasoning": str}`.
- [x] **TASK-019**: Implement resilient mock/heuristic fallback (score 5.0) when API key is missing or quota is exhausted.
- [x] **TASK-020**: Store escalation scores and qualitative rationales in `scores` table with `--rerun` support.
- [x] **TASK-021**: Build validation template generator outputting to `validation/score_validation.jsonl`.

---

## Phase 5: Semantic Clustering & Event Detection (`cluster.py`)

- [x] **TASK-022**: Generate 384-dimensional dense semantic embeddings using `sentence-transformers/all-MiniLM-L6-v2` over article title and lead sentences.
- [x] **TASK-023**: Implement embedding persistence and caching via `data/embeddings.npz` to eliminate redundant vector computation.
- [x] **TASK-024**: Implement DBSCAN clustering with cosine distance ($\varepsilon=0.30$, $min\_samples=2$) to dynamically group co-occurring reports into events.
- [x] **TASK-025**: Store cluster assignments in `clusters` table with noise isolation (`cluster_id = -1`).
- [x] **TASK-026**: Build validation template generator outputting to `validation/cluster_validation.jsonl`.

---

## Phase 6: Threat Aggregation & Tension Modeling (`aggregate.py`)

- [x] **TASK-027**: Implement mathematical time-decay model using exponential half-life formula ($\lambda=0.05$, $t_{1/2} \approx 13.86\text{ hours}$).
- [x] **TASK-028**: Apply threat category multipliers ($1.5\times$ military conflict, $1.3\times$ terrorism, $1.2\times$ cyberattack, $1.0\times$ civil unrest, $0.8\times$ diplomatic tension, $0.6\times$ natural disaster).
- [x] **TASK-029**: Map ISO-3166-1 alpha-2 codes to alpha-3 codes (`data/world_countries.json`) with overrides for disputed territories (Kosovo, Palestine, Taiwan, Hong Kong, Macau).
- [x] **TASK-030**: Calculate 24-hour regional crisis trajectory delta indicators (Rising 🔺, Falling 🔻, Stable ➖) and 7-day rolling time-series.
- [x] **TASK-031**: Validate metric transformations and JSON export functions for UI consumption.

---

## Phase 7: Interactive Intelligence Dashboard (`app.py`)

- [x] **TASK-032**: Build responsive Streamlit dashboard with dark-mode tactical command aesthetic (`STRATINT`).
- [x] **TASK-033**: Implement Global Tension Heatmap using Folium GeoJSON choropleth with interactive tooltips and decay-adjusted tension scaling.
- [x] **TASK-034**: Implement 7-day regional crisis trend time-series chart with Plotly Dark.
- [x] **TASK-035**: Build High-Threat Article Feed tab with multi-select filtering by threat category, region, and minimum escalation score.
- [x] **TASK-036**: Build Event Clusters tab displaying DBSCAN-grouped multi-source crisis stories with cross-feed corroboration.
- [x] **TASK-037**: Build Regional Crisis Ranking tab displaying sorted risk hotspots and 24h trajectory trends.
- [x] **TASK-038**: Build AI Model Benchmarks tab visualizing latest metrics from `metrics.json`.
- [x] **TASK-039**: Secure UI against HTML injection (XSS prevention with `html.escape()` on all rendered article titles, summaries, and categories).

---

## Phase 8: Unified Evaluation & Quality Auditing (`evaluate.py`)

- [x] **TASK-040**: Build unified evaluation CLI (`evaluate.py --phase ner|classify|score|cluster`) supporting CoNLL strict span evaluation, multi-label F1, Spearman $\rho$, and pair-wise clustering accuracy.
- [x] **TASK-041**: Benchmark all ML stages against curated ground-truth datasets in `validation/` and record metrics in `metrics.json`.
- [x] **TASK-042**: Conduct comprehensive repository audit across Correctness, Architecture, Security, Error Handling, Accessibility, and Performance.
- [x] **TASK-043**: Verify end-to-end execution from RSS ingestion through dashboard rendering and evaluation.

---

- [x] **TASK-044**: Add in-dashboard 1-click live intelligence sync and modular orchestrator (`pipeline_runner.py`, `run_pipeline.bat`, `run_pipeline.ps1`).
- [ ] **TASK-045**: Add Telegram/Slack/Email webhook alerts for high-severity threshold triggers (Score $\ge 8.0$).
- [ ] **TASK-046**: Add multi-language translation pipeline for non-English foreign OSINT news sources.
- [ ] **TASK-047**: Support export of intelligence reports to structured PDF / STIX 2.1 threat intelligence format.
