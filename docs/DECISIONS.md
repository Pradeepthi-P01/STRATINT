# Architectural Decision Records (ADR)

This document records the design decisions, trade-offs, and technical rationale underlying the STRATINT pipeline.

---

## ADR-001: SQLite with WAL Mode & Threading Configuration

- **Status**: Accepted
- **Context**: The application requires zero-infrastructure deployment for single-workstation analyst workflows while supporting concurrent reads from Streamlit during active ingestion.
- **Decision**: Use embedded SQLite with Write-Ahead Logging (`PRAGMA journal_mode=WAL;`), `check_same_thread=False`, foreign keys enabled (`PRAGMA foreign_keys=ON;`), and a 16MB page cache (`PRAGMA cache_size=-16000;`).
- **Consequences**:
  - **Pros**: Zero installation overhead, portable database file (`osint.db`), seamless non-blocking reads during batch write transactions.
  - **Cons**: Write serialization limits pipeline to a single active write process at a time.

---

## ADR-002: spaCy `en_core_web_lg` for Geopolitical Entity Extraction

- **Status**: Accepted
- **Context**: The NER stage needs to extract `GPE`, `ORG`, and `NORP` reliably on Windows and Python 3.13 without Cython compilation failures or slow runtime transformer downloads.
- **Decision**: Default to `en_core_web_lg` (floret vectors) with an optional upgrade path to `en_core_web_trf` via `--model`.
- **Consequences**:
  - **Pros**: Self-contained, robust cross-platform execution on Windows, ~85.5 OntoNotes F1 benchmark.
  - **Cons**: Slightly lower accuracy than transformer-based RoBERTa models on complex sentence boundaries.

---

## ADR-003: Multi-Label Zero-Shot NLI Classification

- **Status**: Accepted
- **Context**: Threat reports frequently span overlapping operational categories (e.g. an incident involving both cyber warfare and armed conflict).
- **Decision**: Use `facebook/bart-large-mnli` with independent Sigmoid activation (`multi_label=True`) and a 0.45 confidence threshold across 6 operational categories.
- **Consequences**:
  - **Pros**: Assigns zero, one, or multiple threat tags without assuming mutual exclusivity; requires no supervised dataset re-training.
  - **Cons**: 1.63 GB model footprint on CPU.

---

## ADR-004: Google Gemini 1.5 Flash with Token-Bucket Rate Limiting

- **Status**: Accepted
- **Context**: Geopolitical escalation severity requires contextual LLM reasoning, but must operate within free-tier API quotas and stay resilient to rate limits.
- **Decision**: Use `gemini-1.5-flash` with a 14 req/min token bucket, few-shot prompt anchors, 60s 429 quota backoff, and a defensive mock heuristic fallback (score 5.0).
- **Consequences**:
  - **Pros**: Highly calibrated scoring with qualitative rationale; 100% crash-proof fallback if credentials or quotas fail.
  - **Cons**: Sequential rate-limited throughput on large backlogs.

---

## ADR-005: Dense Sentence Embeddings with DBSCAN for Event Detection

- **Status**: Accepted
- **Context**: Ingested articles cover an unpredictable number of simultaneous breaking events; keyword-based TF-IDF fails across journalistic synonym variance.
- **Decision**: Encode headline and lead sentences using `all-MiniLM-L6-v2` (384-dim), cache vectors in `data/embeddings.npz`, and cluster using DBSCAN ($\varepsilon=0.30$, $min\_samples=2$).
- **Consequences**:
  - **Pros**: Automatically discovers variable clusters; isolates uncorroborated singletons as noise (`cluster_id = -1`).
  - **Cons**: Requires vector persistence across re-runs.

---

## ADR-006: Exponential Time-Decay Modeling for Regional Tension

- **Status**: Accepted
- **Context**: Tactical situational awareness requires recent incidents to contribute heavily while older reports naturally diminish in tension impact.
- **Decision**: Model tension with an exponential half-life decay function:
  $$\text{Tension}(R) = \sum_{i \in R} \text{Score}_i \times w_c \times e^{-\lambda \Delta t}$$
  where $\lambda = 0.05$ ($t_{1/2} \approx 13.86\text{ hours}$) and $w_c$ is the threat category weight.
- **Consequences**:
  - **Pros**: Produces realistic tension dynamics where hot spots cool down if no fresh reports emerge.
  - **Cons**: Dependent on accurate publication timestamps from RSS feeds (handled via fallback to ingestion time).
