# Pipeline Operations & CLI Reference Guide

## Pipeline Execution Sequence

Execute the STRATINT pipeline sequentially from raw news ingestion to dashboard visualization:

```bash
# 1. Ingest RSS feeds (or use --dry-run / --limit for testing)
python ingest.py

# Or run the complete automated 1-click sync pipeline:
python pipeline_runner.py
# (Or double-click run_pipeline.bat on Windows)

# 2. Extract geopolitical entities (spaCy en_core_web_lg)
python ner.py

# 3. Classify threat categories (BART-large-MNLI)
python classify.py

# 4. Score escalation severity (Gemini 1.5 Flash)
python score.py

# 5. Cluster articles into crisis events (all-MiniLM-L6-v2 + DBSCAN)
python cluster.py

# 6. Run evaluation benchmarks across phases
python evaluate.py --phase ner
python evaluate.py --phase classify
python evaluate.py --phase score
python evaluate.py --phase cluster

# 7. Launch Interactive Intelligence Dashboard
python -m streamlit run app.py
# (Or double-click run_dashboard.bat on Windows)
```

---

## Detailed CLI Parameter Reference

### 1. Ingestion (`ingest.py`)

```bash
python ingest.py [OPTIONS]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--dry-run` | Flag | `False` | Fetches and parses all 10 RSS feeds, prints formatted table statistics, but skips database writes. |
| `--limit <N>` | Integer | `None` | Processes at most `N` entries per feed (useful for fast smoke-testing). |

---

### 2. Named Entity Recognition (`ner.py`)

```bash
python ner.py [OPTIONS]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--rerun` | Flag | `False` | Deletes existing entity records and re-extracts entities for all articles in the database. |
| `--model <NAME>` | String | `en_core_web_lg` | spaCy model name (e.g., `en_core_web_lg` or `en_core_web_trf`). |
| `--batch-size <N>` | Integer | `16` | Batch size for `nlp.pipe()` extraction throughput. |
| `--limit <N>` | Integer | `None` | Restricts extraction to at most `N` unprocessed articles. |
| `--generate-validation` | Flag | `False` | Generates a pre-filled JSONL template at `validation/ner_validation.jsonl`. |
| `--val-size <N>` | Integer | `25` | Number of articles to sample when generating validation templates. |

---

### 3. Multi-Label Zero-Shot Classification (`classify.py`)

```bash
python classify.py [OPTIONS]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--rerun` | Flag | `False` | Deletes existing classification records and re-classifies all articles. |
| `--limit <N>` | Integer | `None` | Limits inference to at most `N` unprocessed articles. |
| `--threshold <FLOAT>` | Float | `0.45` | Sigmoid confidence threshold for retaining candidate threat categories. |
| `--generate-validation` | Flag | `False` | Generates a pre-filled validation template at `validation/classification_validation.jsonl`. |
| `--val-size <N>` | Integer | `25` | Number of articles to sample for the validation template. |

---

### 4. Escalation Scoring (`score.py`)

```bash
python score.py [OPTIONS]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--rerun` | Flag | `False` | Clears existing scores and re-evaluates all articles with the LLM. |
| `--limit <N>` | Integer | `None` | Limits scoring to at most `N` unprocessed articles. |
| `--generate-validation` | Flag | `False` | Generates a pre-filled validation template at `validation/score_validation.jsonl`. |
| `--val-size <N>` | Integer | `25` | Number of articles to sample for the validation template. |

---

### 5. Semantic Event Clustering (`cluster.py`)

```bash
python cluster.py [OPTIONS]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--rerun` | Flag | `False` | Clears cached embeddings and recomputes all cluster assignments. |
| `--eps <FLOAT>` | Float | `0.30` | Maximum cosine distance for DBSCAN neighborhood grouping. |
| `--min-samples <N>` | Integer | `2` | Minimum number of core samples to establish an event cluster. |
| `--generate-validation` | Flag | `False` | Generates pair-wise cluster validation file at `validation/cluster_validation.jsonl`. |
| `--val-size <N>` | Integer | `20` | Number of article pairs to sample for cluster validation. |

---

### 6. Unified Evaluation Harness (`evaluate.py`)

```bash
python evaluate.py --phase <ner|classify|score|cluster> [OPTIONS]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--phase <NAME>` | Choice | *Required* | Phase to benchmark: `ner`, `classify`, `score`, or `cluster`. |
| `--input <PATH>` | Path | Phase default | Override path to custom validation JSONL file. |
| `--no-save` | Flag | `False` | Print results to stdout without appending to `metrics.json`. |

---

## Environment Variables (`.env`)

```env
# Google Gemini API Key for Phase 4 Escalation Scoring
GEMINI_API_KEY=your_actual_gemini_api_key_here
```
