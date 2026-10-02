# Security Policy & Hardening Guidelines

## Threat Model & Security Posture

**STRATINT** ingests untrusted text payloads from public RSS feeds, parses and structures them with NLP/ML models, stores them in SQLite, and visualizes them on a Streamlit dashboard.

---

## Attack Surface Analysis & Hardening Measures

### 1. Cross-Site Scripting (XSS) in Dashboard UI
- **Vulnerability**: Malicious RSS feed entries or manipulated news titles/summaries containing injected `<script>` tags, HTML attributes, or `javascript:` URIs.
- **Implemented Mitigation**:
  - All dynamic data rendered into custom HTML containers (`st.markdown(..., unsafe_allow_html=True)`) is sanitized using Python's `html.escape()`.
  - Article titles, summary snippets, source names, and category pills are escaped prior to HTML string formatting.
  - GeoJSON tooltip fields and Plotly hover text are strictly typed and escaped.

### 2. SQL Injection (SQLi)
- **Vulnerability**: Malicious strings in RSS fields attempting SQL query manipulation.
- **Implemented Mitigation**:
  - 100% of SQL statements across `db.py`, `ingest.py`, `ner.py`, `classify.py`, `score.py`, `cluster.py`, and `aggregate.py` use parameterized queries with `?` placeholders.
  - String concatenation/interpolation is prohibited in all database interactions.
  - Foreign key constraints (`PRAGMA foreign_keys=ON;`) prevent orphaned child rows.

### 3. API Key & Credential Isolation
- **Vulnerability**: Exposure of the Google Gemini API key via source control or logs.
- **Implemented Mitigation**:
  - API keys are loaded strictly from the local `.env` file via `python-dotenv`.
  - `.gitignore` explicitly excludes `.env`, `*.db`, `*.log`, `data/embeddings.npz`, and cache folders.
  - If `GEMINI_API_KEY` is absent or invalid, `score.py` automatically falls back to an offline heuristic score (5.0) without exposing errors or halting execution.

### 4. Denial of Service & Network Hangs
- **Vulnerability**: Slow or hung remote RSS servers blocking ingestion indefinitely.
- **Implemented Mitigation**:
  - Global socket timeout is set to 15 seconds around feed parsing.
  - Feed parsing errors are caught in per-feed `try/except` blocks, allowing remaining feeds to process uninterrupted.
  - Deduplication prevents re-processing previously ingested articles.
