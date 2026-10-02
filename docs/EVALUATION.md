# Evaluation Methodology & Benchmark Results

## Overview

The evaluation harness (`evaluate.py`) benchmarks all machine learning stages against hand-curated ground-truth datasets stored in `validation/`. Every evaluation run appends structured timestamped records to `metrics.json`.

---

## Ground-Truth Validation Datasets

| Phase | Validation File | Evaluation Method | Primary Target |
|---|---|---|---|
| **NER** (Phase 2) | `validation/ner_validation.jsonl` | Strict CoNLL character span match (`start_char`, `end_char`, `label`) + Canonical ISO mapping accuracy | `GPE`, `ORG`, `NORP` entity extraction |
| **Classification** (Phase 3) | `validation/classification_validation.jsonl` | Multi-label confusion metrics (TP, FP, FN per category) | 6-class threat taxonomy with multi-label assignment |
| **Scoring** (Phase 4) | `validation/score_validation.jsonl` | Spearman's rank correlation ($\rho$), p-value, and Mean Absolute Error (MAE) | 0.0–10.0 escalation severity rating |
| **Clustering** (Phase 5) | `validation/cluster_validation.jsonl` | Pair-wise event co-occurrence accuracy, Precision, Recall, F1 | Semantic grouping of multi-source crisis reports |

---

## Benchmark Results (from `metrics.json`)

### 1. Zero-Shot Threat Classification (`classify.py`)
- **Model**: `facebook/bart-large-mnli` (Sigmoid threshold = 0.45)
- **Validation Dataset**: `validation/classification_validation.jsonl` (25 annotated articles)

| Category | Precision | Recall | F1-Score |
|---|---|---|---|
| **Military Conflict** | 0.8333 | 1.0000 | **0.9091** |
| **Cyberattack** | 1.0000 | 1.0000 | **1.0000** |
| **Civil Unrest** | 0.5000 | 1.0000 | **0.6667** |
| **Diplomatic Tension** | 0.4286 | 1.0000 | **0.6000** |
| **Terrorism / Extremist Violence** | 0.3333 | 1.0000 | **0.5000** |
| **Natural Disaster** | N/A | N/A | Baseline |
| **Macro Average** | **0.5159** | **0.8333** | **0.6126** |
| **Exact Match Ratio** | — | — | **0.8400** |

---

### 2. Escalation Scoring (`score.py`)
- **Model**: `gemini-1.5-flash` (few-shot calibrated prompt)
- **Validation Dataset**: `validation/score_validation.jsonl` (25 annotated articles)

| Metric | Result | Interpretation |
|---|---|---|
| **Spearman Rank Correlation ($\rho$)** | **0.5526** | Statistically significant positive monotonic correlation with human expert ranking ($p = 0.004175$). |
| **Mean Absolute Error (MAE)** | **1.48** | On the 0.0–10.0 continuous scale, predictions deviate on average by under 1.5 points. |

---

### 3. Event Clustering (`cluster.py`)
- **Model**: `all-MiniLM-L6-v2` + DBSCAN ($\varepsilon=0.30$, $min\_samples=2$)
- **Validation Dataset**: `validation/cluster_validation.jsonl` (20 evaluated pairs)

| Metric | Result |
|---|---|
| **Pair-wise Precision** | **1.0000** (10/10) |
| **Pair-wise Recall** | **1.0000** (10/10) |
| **Pair-wise F1 Score** | **1.0000** |
| **Pair-wise Accuracy** | **1.0000** (20/20) |

---

## Running Benchmarks via CLI

```bash
# Run NER evaluation
python evaluate.py --phase ner

# Run classification evaluation
python evaluate.py --phase classify

# Run escalation score evaluation
python evaluate.py --phase score

# Run clustering evaluation
python evaluate.py --phase cluster
```
