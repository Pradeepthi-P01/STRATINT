"""
evaluate.py — Unified evaluation CLI for all ML-driven pipeline stages.

════════════════════════════════════════════════════════════════════════════════
OVERVIEW
════════════════════════════════════════════════════════════════════════════════

Each phase has its own sub-command:
    python evaluate.py --phase ner      [--input validation/ner_validation.jsonl]
    python evaluate.py --phase classify [--input validation/classify_validation.jsonl]
    python evaluate.py --phase score    [--input validation/score_validation.jsonl]
    python evaluate.py --phase cluster  [--input validation/cluster_validation.jsonl]

All phases:
    - Print a formatted results table to stdout.
    - Append a timestamped result record to metrics.json under the phase key.
    - Return exit code 0 on success.

════════════════════════════════════════════════════════════════════════════════
NER EVALUATION DESIGN  (plain English)
════════════════════════════════════════════════════════════════════════════════

Metric: Precision, Recall, F1 per entity type (GPE / ORG / NORP) + macro avg.

Matching criterion: EXACT SPAN MATCH (CoNLL strict evaluation).
    A prediction is a True Positive only if:
        predicted.start_char == gold.start_char  AND
        predicted.end_char   == gold.end_char    AND
        predicted.label      == gold.label

This is strict but meaningful. Two failure modes are tracked separately:

  - TYPE ERROR:  The model got the span boundaries right but the label wrong.
                 Example: gold=(start=0, end=5, label=GPE)
                          pred=(start=0, end=5, label=ORG)
                 This is BOTH a FP (for ORG) and a FN (for GPE).
                 Reported separately as "type_errors" in the output so you
                 can tell whether the model sees entities but mislabels them.

  - TOTAL MISS:  The gold span was not extracted at all.
                 Reported as part of FN count.

  - SPURIOUS:    A predicted span that the annotator never labeled.
                 Reported as part of FP count.

Canonical evaluation:
    We also separately evaluate the normaliser: among correctly predicted
    entities (TPs), what fraction got the right canonical code?
    This tells you whether the ISO mapping is accurate independent of spaCy.

Input JSONL format (one JSON object per line):
    {
      "article_id": "...",
      "source": "BBC World",
      "text": "...",
      "entities": [
        {"text": "Ukraine", "label": "GPE", "canonical": "UA",
         "start_char": 8, "end_char": 15},
        ...
      ]
    }

    After the model generates the template with `python ner.py --generate-validation`,
    you edit the file to correct errors. The edited file is the gold standard.
"""

import argparse
import io
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ── UTF-8 stdout fix for Windows ─────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from tabulate import tabulate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("evaluate")

VALIDATION_DIR = Path(__file__).parent / "validation"
METRICS_PATH   = Path(__file__).parent / "metrics.json"

DEFAULT_INPUTS = {
    "ner":      VALIDATION_DIR / "ner_validation.jsonl",
    "classify": VALIDATION_DIR / "classification_validation.jsonl" if (VALIDATION_DIR / "classification_validation.jsonl").exists() else VALIDATION_DIR / "classify_validation.jsonl",
    "score":    VALIDATION_DIR / "score_validation.jsonl",
    "cluster":  VALIDATION_DIR / "cluster_validation.jsonl",
}


# ─────────────────────────────────────────────────────────────────────────────
# metrics.json helpers  (shared with ingest.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_metrics() -> dict:
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("metrics.json is corrupt -- starting fresh.")
    return {}


def save_metrics(metrics: dict) -> None:
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def append_phase_result(phase: str, result: dict) -> None:
    """Append a timestamped result to metrics.json under metrics[phase]."""
    metrics = load_metrics()
    record = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **result,
    }
    metrics.setdefault(phase, []).append(record)
    save_metrics(metrics)
    log.info("Result appended to %s under key '%s'.", METRICS_PATH, phase)


# ─────────────────────────────────────────────────────────────────────────────
# NER evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute precision, recall, F1 from raw counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return precision, recall, f1


def evaluate_ner(input_path: Path) -> dict:
    """
    Load gold JSONL, run spaCy NER on each article text, compute P/R/F1.

    Exact span match: TP requires (start_char, end_char, label) all to match.
    Type errors (correct boundary, wrong label) counted separately.

    Returns a dict of per-type and macro metrics suitable for metrics.json.
    """
    if not input_path.exists():
        log.error("Validation file not found: %s", input_path)
        log.error("Run:  python ner.py --generate-validation")
        sys.exit(1)

    # ── Load gold annotations ─────────────────────────────────────────────
    gold_records: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                gold_records.append(json.loads(line))

    if not gold_records:
        log.error("Validation file is empty: %s", input_path)
        sys.exit(1)

    log.info("Loaded %d gold articles from %s", len(gold_records), input_path)

    # ── Load spaCy model ──────────────────────────────────────────────────
    # Import here so that evaluate.py can be imported without spaCy installed
    # for phases that don't need it.
    try:
        from ner import load_model, extract_entities_from_doc, normalize_entity
    except ImportError as exc:
        log.error("Cannot import ner.py: %s", exc)
        sys.exit(1)

    nlp = load_model("en_core_web_lg")

    # ── Run NER on gold texts ─────────────────────────────────────────────
    texts = [rec["text"] for rec in gold_records]
    all_predictions: list[list[dict]] = []
    for doc in nlp.pipe(texts, batch_size=8):
        # Use extract_entities_from_doc() — same deduplication logic as ner.py
        # so FP counts aren't inflated by repeated spans in a single article.
        preds = extract_entities_from_doc(doc, article_id="_eval")
        all_predictions.append(preds)

    # ── Compute span-level metrics ────────────────────────────────────────
    # Counters per label type
    tp_by_label:   dict[str, int] = defaultdict(int)
    fp_by_label:   dict[str, int] = defaultdict(int)
    fn_by_label:   dict[str, int] = defaultdict(int)
    type_errors:   int = 0   # correct boundary, wrong label (subset of FP+FN)
    canonical_tp:  int = 0   # TP predictions with correct canonical code
    canonical_tot: int = 0   # TP predictions that have a gold canonical code

    for rec, preds in zip(gold_records, all_predictions):
        gold_spans: list[dict] = rec.get("entities", [])

        # Build lookup sets: (start, end, label) for exact match
        gold_set = {
            (e["start_char"], e["end_char"], e["label"]): e
            for e in gold_spans
        }
        pred_set = {
            (e["start_char"], e["end_char"], e["label"]): e
            for e in preds
        }

        # True Positives: prediction matches a gold span exactly
        for key, pred in pred_set.items():
            if key in gold_set:
                tp_by_label[pred["label"]] += 1
                # Canonical accuracy for TPs
                gold_canon = gold_set[key].get("canonical")
                if gold_canon is not None:
                    canonical_tot += 1
                    if pred.get("canonical") == gold_canon:
                        canonical_tp += 1
            else:
                fp_by_label[pred["label"]] += 1

        # False Negatives: gold span not in predictions
        for key, gold in gold_set.items():
            if key not in pred_set:
                fn_by_label[gold["label"]] += 1

        # Type errors: same (start, end) but different label
        gold_bounds = {(e["start_char"], e["end_char"]): e["label"] for e in gold_spans}
        pred_bounds = {(e["start_char"], e["end_char"]): e["label"] for e in preds}
        for bounds, pred_label in pred_bounds.items():
            if bounds in gold_bounds and gold_bounds[bounds] != pred_label:
                type_errors += 1

    # ── Build results dict ────────────────────────────────────────────────
    labels = ["GPE", "ORG", "NORP"]
    per_type: dict[str, dict] = {}
    macro_p = macro_r = macro_f1 = 0.0

    for label in labels:
        tp = tp_by_label[label]
        fp = fp_by_label[label]
        fn = fn_by_label[label]
        p, r, f1 = _prf(tp, fp, fn)
        per_type[label] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f1, 4),
        }
        macro_p  += p
        macro_r  += r
        macro_f1 += f1

    n_labels = len(labels)
    macro = {
        "precision": round(macro_p  / n_labels, 4),
        "recall":    round(macro_r  / n_labels, 4),
        "f1":        round(macro_f1 / n_labels, 4),
    }

    canon_acc = (
        round(canonical_tp / canonical_tot, 4) if canonical_tot > 0 else None
    )

    result = {
        "model":            "en_core_web_lg",
        "validation_file":  str(input_path),
        "articles_evaluated": len(gold_records),
        "per_type":         per_type,
        "macro":            macro,
        "type_errors":      type_errors,
        "canonical_accuracy": canon_acc,
    }

    # ── Print results table ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("NER EVALUATION RESULTS  (exact span match)")
    print("=" * 65)

    table_rows = []
    for label in labels:
        m = per_type[label]
        table_rows.append([
            label,
            m["tp"], m["fp"], m["fn"],
            f"{m['precision']:.3f}",
            f"{m['recall']:.3f}",
            f"{m['f1']:.3f}",
        ])
    table_rows.append([
        "MACRO AVG", "", "", "",
        f"{macro['precision']:.3f}",
        f"{macro['recall']:.3f}",
        f"{macro['f1']:.3f}",
    ])

    print(tabulate(
        table_rows,
        headers=["Label", "TP", "FP", "FN", "Precision", "Recall", "F1"],
        tablefmt="rounded_outline",
    ))

    print(f"\nType errors (correct boundary, wrong label): {type_errors}")
    if canon_acc is not None:
        print(f"Canonical code accuracy (among TPs):        {canon_acc:.1%}")
    else:
        print("Canonical code accuracy: N/A (no canonical annotations in gold set)")
    print(f"Articles evaluated: {len(gold_records)}")
    print(f"Model: en_core_web_lg")
    print("=" * 65 + "\n")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Phase stubs (filled in later phases)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_classify(input_path: Path) -> dict:
    """
    Load gold classification JSONL, run zero-shot classification, compute P/R/F1.

    Evaluates multi-label classification across the 6 OSINT threat categories:
      - Precision, Recall, F1 per category
      - Macro Average Precision, Recall, F1
      - Exact match (subset accuracy: articles where predicted tags == gold tags)

    Returns a dict suitable for appending to metrics.json.
    """
    if not input_path.exists():
        # Also check fallback name
        alt_path = (
            VALIDATION_DIR / "classification_validation.jsonl"
            if input_path.name == "classify_validation.jsonl"
            else VALIDATION_DIR / "classify_validation.jsonl"
        )
        if alt_path.exists():
            input_path = alt_path
        else:
            log.error("Validation file not found: %s", input_path)
            log.error("Run:  python classify.py --generate-validation")
            sys.exit(1)

    gold_records: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                gold_records.append(json.loads(line))

    if not gold_records:
        log.error("Validation file is empty: %s", input_path)
        sys.exit(1)

    log.info("Loaded %d gold articles from %s", len(gold_records), input_path)

    try:
        from classify import (
            THREAT_CATEGORIES,
            HYPOTHESIS_TEMPLATE,
            DEFAULT_THRESHOLD,
            load_classifier,
        )
    except ImportError as exc:
        log.error("Cannot import classify.py: %s", exc)
        sys.exit(1)

    classifier = load_classifier()

    texts = [rec["text"] for rec in gold_records]
    log.info("Running zero-shot classification on %d gold articles ...", len(texts))

    outputs = classifier(
        texts,
        candidate_labels=THREAT_CATEGORIES,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        multi_label=True,
    )
    if not isinstance(outputs, list):
        outputs = [outputs]

    # Metrics counters
    tp_by_cat: dict[str, int] = defaultdict(int)
    fp_by_cat: dict[str, int] = defaultdict(int)
    fn_by_cat: dict[str, int] = defaultdict(int)
    exact_matches = 0

    for rec, out in zip(gold_records, outputs):
        gold_cats = set(rec.get("categories") or rec.get("labels") or [])
        pred_cats = {
            label for label, score in zip(out["labels"], out["scores"])
            if score >= DEFAULT_THRESHOLD
        }

        if pred_cats == gold_cats:
            exact_matches += 1

        for cat in THREAT_CATEGORIES:
            in_gold = cat in gold_cats
            in_pred = cat in pred_cats

            if in_gold and in_pred:
                tp_by_cat[cat] += 1
            elif in_pred and not in_gold:
                fp_by_cat[cat] += 1
            elif in_gold and not in_pred:
                fn_by_cat[cat] += 1

    per_category: dict[str, dict] = {}
    macro_p = macro_r = macro_f1 = 0.0

    for cat in THREAT_CATEGORIES:
        tp = tp_by_cat[cat]
        fp = fp_by_cat[cat]
        fn = fn_by_cat[cat]
        p, r, f1 = _prf(tp, fp, fn)
        per_category[cat] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f1, 4),
        }
        macro_p  += p
        macro_r  += r
        macro_f1 += f1

    n_cats = len(THREAT_CATEGORIES)
    macro = {
        "precision": round(macro_p  / n_cats, 4),
        "recall":    round(macro_r  / n_cats, 4),
        "f1":        round(macro_f1 / n_cats, 4),
    }
    subset_acc = round(exact_matches / len(gold_records), 4)

    result = {
        "model": "facebook/bart-large-mnli",
        "threshold": DEFAULT_THRESHOLD,
        "validation_file": str(input_path),
        "articles_evaluated": len(gold_records),
        "per_category": per_category,
        "macro": macro,
        "exact_match_ratio": subset_acc,
    }

    print("\n" + "=" * 70)
    print("ZERO-SHOT CLASSIFICATION EVALUATION RESULTS (multi_label=True)")
    print("=" * 70)

    table_rows = []
    for cat in THREAT_CATEGORIES:
        m = per_category[cat]
        table_rows.append([
            cat,
            m["tp"], m["fp"], m["fn"],
            f"{m['precision']:.3f}",
            f"{m['recall']:.3f}",
            f"{m['f1']:.3f}",
        ])
    table_rows.append([
        "MACRO AVG", "", "", "",
        f"{macro['precision']:.3f}",
        f"{macro['recall']:.3f}",
        f"{macro['f1']:.3f}",
    ])

    print(tabulate(
        table_rows,
        headers=["Category", "TP", "FP", "FN", "Precision", "Recall", "F1"],
        tablefmt="rounded_outline",
    ))

    print(f"\nExact Match (all tags match exactly): {subset_acc:.1%} ({exact_matches}/{len(gold_records)})")
    print(f"Articles evaluated:                  {len(gold_records)}")
    print(f"Confidence threshold:                >= {DEFAULT_THRESHOLD}")
    print(f"Model:                               facebook/bart-large-mnli")
    print("=" * 70 + "\n")

    return result


def evaluate_score(input_path: Path) -> dict:
    """
    Evaluate Gemini escalation scores against human manual ratings in score_validation.jsonl.

    Computes:
      - Spearman rank correlation (r_s and p-value)
      - Mean Absolute Error (MAE)
      - LLM Consistency evaluation (std deviation)
    """
    if not input_path.exists():
        log.error("Validation file not found: %s", input_path)
        log.error("Run:  python score.py --generate-validation")
        sys.exit(1)

    gold_records: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                gold_records.append(json.loads(line))

    if not gold_records:
        log.error("Validation file is empty: %s", input_path)
        sys.exit(1)

    log.info("Loaded %d scored articles from %s", len(gold_records), input_path)

    # Check for manual ratings
    pairs = []
    for r in gold_records:
        g_score = r.get("gemini_score")
        m_score = r.get("manual_score")
        if g_score is not None and m_score is not None:
            try:
                pairs.append((float(g_score), float(m_score), r.get("text", "")[:60]))
            except (ValueError, TypeError):
                continue

    if not pairs:
        print("\n" + "=" * 65)
        print("ATTENTION: 'manual_score' fields are currently empty/null!")
        print("=" * 65)
        print("Open validation/score_validation.jsonl and add your manual 0-10")
        print("ratings in 'manual_score' next to 'gemini_score'.")
        print("Example: {\"gemini_score\": 7.0, \"manual_score\": 8.0, ...}")
        print("=" * 65 + "\n")
        return {
            "validation_file": str(input_path),
            "articles_loaded": len(gold_records),
            "manual_annotations": 0,
            "status": "awaiting_manual_scores",
        }

    gemini_scores = [p[0] for p in pairs]
    manual_scores = [p[1] for p in pairs]

    # Calculate MAE
    mae = sum(abs(g - m) for g, m in zip(gemini_scores, manual_scores)) / len(pairs)
    mae = round(mae, 4)

    # Calculate Spearman rank correlation
    try:
        from scipy.stats import spearmanr
        res = spearmanr(gemini_scores, manual_scores)
        spearman_corr = round(float(res.correlation), 4)
        spearman_p = round(float(res.pvalue), 6)
    except Exception as e:
        log.warning("scipy spearmanr calculation failed: %s", e)
        spearman_corr = None
        spearman_p = None

    result = {
        "model": "gemini-1.5-flash",
        "validation_file": str(input_path),
        "articles_evaluated": len(pairs),
        "spearman_correlation": spearman_corr,
        "spearman_p_value": spearman_p,
        "mean_absolute_error": mae,
    }

    print("\n" + "=" * 65)
    print("GEMINI ESCALATION SCORING EVALUATION RESULTS")
    print("=" * 65)

    metric_rows = [
        ["Spearman Rank Correlation (r_s)", f"{spearman_corr:.4f}" if spearman_corr is not None else "N/A", "Measures monotonic ranking alignment (1.0 = perfect)"],
        ["Spearman p-value", f"{spearman_p:.6f}" if spearman_p is not None else "N/A", "Statistical significance (p < 0.05 is significant)"],
        ["Mean Absolute Error (MAE)", f"{mae:.3f} points", "Average absolute error on the 0-10 scale"],
        ["Articles Compared", str(len(pairs)), "Articles with both Gemini and manual scores"],
    ]

    print(tabulate(
        metric_rows,
        headers=["Metric", "Result", "Interpretation"],
        tablefmt="rounded_outline",
    ))

    # Sample comparison preview
    sample_rows = [
        [p[2] + "...", f"{p[0]:.1f}", f"{p[1]:.1f}", f"{abs(p[0] - p[1]):.1f}"]
        for p in pairs[:5]
    ]
    print("\nSample Comparisons (First 5 Articles):")
    print(tabulate(
        sample_rows,
        headers=["Article Snippet", "Gemini", "Manual", "|Diff|"],
        tablefmt="simple",
    ))
    print("=" * 65 + "\n")

    return result


def evaluate_cluster(input_path: Path) -> dict:
    """
    Evaluate DBSCAN pairwise clustering accuracy on cluster_validation.jsonl.

    Computes:
      - Pairwise Precision, Recall, and F1 (same event identification)
      - Pairwise Accuracy across all 20 validation pairs
    """
    if not input_path.exists():
        log.error("Validation file not found: %s", input_path)
        log.error("Run:  python cluster.py --generate-validation")
        sys.exit(1)

    pairs: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    if not pairs:
        log.error("Validation file is empty: %s", input_path)
        sys.exit(1)

    log.info("Loaded %d validation pairs from %s", len(pairs), input_path)

    tp = fp = fn = tn = 0
    for p in pairs:
        pred = p.get("predicted_same_event")
        manual = p.get("manual_same_event")

        if manual is None or pred is None:
            continue

        if pred == 1 and manual == 1:
            tp += 1
        elif pred == 1 and manual == 0:
            fp += 1
        elif pred == 0 and manual == 1:
            fn += 1
        elif pred == 0 and manual == 0:
            tn += 1

    total_evaluated = tp + fp + fn + tn
    if total_evaluated == 0:
        print("\n" + "=" * 65)
        print("ATTENTION: 'manual_same_event' fields are empty or null!")
        print("=" * 65)
        print("Please edit validation/cluster_validation.jsonl and fill in")
        print("'manual_same_event' with 1 (same event) or 0 (different event).")
        print("=" * 65 + "\n")
        return {"status": "awaiting_manual_labels"}

    precision, recall, f1 = _prf(tp, fp, fn)
    accuracy = round((tp + tn) / total_evaluated, 4)

    result = {
        "model": "all-MiniLM-L6-v2 + DBSCAN",
        "validation_file": str(input_path),
        "pairs_evaluated": total_evaluated,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": accuracy,
    }

    print("\n" + "=" * 65)
    print("DBSCAN PAIRWISE EVENT CLUSTERING EVALUATION RESULTS")
    print("=" * 65)

    metrics_table = [
        ["Pairwise Precision", f"{precision:.3f}", "Fraction of predicted same-event pairs that actually match"],
        ["Pairwise Recall", f"{recall:.3f}", "Fraction of actual same-event pairs correctly grouped"],
        ["Pairwise F1-Score", f"{f1:.3f}", "Harmonic mean of pairwise precision & recall"],
        ["Overall Pair Accuracy", f"{accuracy:.1%}", f"{tp + tn}/{total_evaluated} pairs correctly classified"],
    ]

    print(tabulate(
        metrics_table,
        headers=["Metric", "Value", "Interpretation"],
        tablefmt="rounded_outline",
    ))

    print(f"\nConfusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print("=" * 65 + "\n")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

PHASE_FNS = {
    "ner":      evaluate_ner,
    "classify": evaluate_classify,
    "score":    evaluate_score,
    "cluster":  evaluate_cluster,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ML pipeline stages against hand-labeled validation sets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --phase ner
  python evaluate.py --phase ner --input my_labels.jsonl
  python evaluate.py --phase classify
""",
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=["ner", "classify", "score", "cluster"],
        help="Which pipeline stage to evaluate.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to validation JSONL. Defaults to validation/<phase>_validation.jsonl.",
    )
    args = parser.parse_args()

    input_path = args.input or DEFAULT_INPUTS[args.phase]
    eval_fn    = PHASE_FNS[args.phase]

    try:
        result = eval_fn(input_path)
        append_phase_result(f"evaluate_{args.phase}", result)
        print(f"Results saved to: {METRICS_PATH}")
    except NotImplementedError as exc:
        log.error("%s", exc)
        sys.exit(1)
