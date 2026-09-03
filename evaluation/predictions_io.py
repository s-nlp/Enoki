"""
Raw predictions I/O and threshold curve computation.

Workflow:
  1. Run `enoki evaluate ...`  →  saves a *_preds.json file with raw NLI
     scores (no thresholding yet).
  2. Run `enoki threshold <preds_file>` to sweep thresholds and compute
     P/R/F1 curves for all available score signals without re-running inference.

File format (JSON):
  {
    "type": "sentence" | "span" | "entity",
    "dataset": str,
    "method": str,
    "timestamp": str,
    "samples": [...]   # see per-type schemas below
  }

Sentence sample:
  {
    "gold": 0 | 1,           # 1 = hallucinated response
    "invalid_context": bool,
    "no_facts": bool,
    "facts": [str, ...],     # extracted fact strings
    "fact_nli_scores": [{"entailment": float, "neutral": float, "contradiction": float}, ...]
  }

Span sample:
  {
    "gold_spans": [[start, end], ...],
    "fact_spans": [{"start": int, "end": int, "hall_prob": float,
                    "entailment": float, "neutral": float, "contradiction": float,
                    "fact": str}, ...]
  }

Entity sample:
  {
    "gold_labels": [0 | 1, ...],   # per-entity labels (1 = hallucinated)
    "entity_scores": [float, ...], # per-entity hallucination probability
    "facts": [{"fact": str, "orig_span_start": int, "orig_span_end": int,
               "hall_prob": float, "entailment": float, "neutral": float,
               "contradiction": float, "span_kind": str}, ...]
  }
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# I/O helpers
# =============================================================================

def save_raw_predictions(path: Path, data: Dict[str, Any]) -> None:
    """Save raw predictions dict to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved raw predictions to {path}")


def load_raw_predictions(path: Path) -> Dict[str, Any]:
    """Load raw predictions from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_curves_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Save threshold curve rows to a CSV file."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    has_conf = any("conf_threshold" in r for r in rows)
    has_iou = any("iou" in r for r in rows)
    fieldnames = ["metric", "threshold", "precision", "recall", "f1"]
    if has_iou:
        fieldnames.append("iou")
    if has_conf:
        fieldnames = ["metric", "conf_threshold", "threshold", "precision", "recall", "f1"]
        if has_iou:
            fieldnames.append("iou")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved threshold curves to {path}")


# =============================================================================
# Shared helpers
# =============================================================================

def _hal_prob_from_nli(fs: Dict[str, float], mode: str = "default") -> float:
    n = float(fs.get("neutral", 0.0))
    c = float(fs.get("contradiction", 0.0))
    if mode == "contradiction_only":
        return c
    if mode == "neutral_only":
        return n
    return n + c  # default: contradiction + neutral


def _prf(
    golds: List[int], preds: List[int]
) -> Tuple[float, float, float]:
    """Precision/recall/F1 for the positive class (class 1)."""
    tp = sum(p == 1 and g == 1 for p, g in zip(preds, golds))
    fp = sum(p == 1 and g == 0 for p, g in zip(preds, golds))
    fn = sum(p == 0 and g == 1 for p, g in zip(preds, golds))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _macro_prf(
    golds: List[int], preds: List[int]
) -> Tuple[float, float, float]:
    """Macro-averaged precision/recall/F1 across both classes (matches sklearn macro)."""
    tp1 = sum(p == 1 and g == 1 for p, g in zip(preds, golds))
    fp1 = sum(p == 1 and g == 0 for p, g in zip(preds, golds))
    fn1 = sum(p == 0 and g == 1 for p, g in zip(preds, golds))
    tn1 = len(golds) - tp1 - fp1 - fn1

    prec1 = tp1 / (tp1 + fp1) if (tp1 + fp1) > 0 else 1.0
    rec1 = tp1 / (tp1 + fn1) if (tp1 + fn1) > 0 else 0.0
    f1_1 = 2.0 * prec1 * rec1 / (prec1 + rec1) if (prec1 + rec1) > 0 else 0.0

    # Class 0: TP0=TN1, FP0=FN1, FN0=FP1
    prec0 = tn1 / (tn1 + fn1) if (tn1 + fn1) > 0 else 1.0
    rec0 = tn1 / (tn1 + fp1) if (tn1 + fp1) > 0 else 0.0
    f1_0 = 2.0 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) > 0 else 0.0

    return (prec0 + prec1) / 2.0, (rec0 + rec1) / 2.0, (f1_0 + f1_1) / 2.0


def _sweep(
    golds: List[int],
    scores: List[float],
    metric_name: str,
    thresholds: List[float],
    macro: bool = False,
) -> List[Dict[str, Any]]:
    """Sweep threshold values and collect P/R/F1 rows."""
    prf_fn = _macro_prf if macro else _prf
    rows = []
    for t in thresholds:
        preds = [1 if s > t else 0 for s in scores]
        prec, rec, f1 = prf_fn(golds, preds)
        rows.append(
            {
                "metric": metric_name,
                "threshold": float(t),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
            }
        )
    return rows


def _default_thresholds(n: int = 50) -> List[float]:
    return list(np.linspace(0.0, 1.0, n + 1))


# =============================================================================
# Sentence-level threshold curves
# =============================================================================

def compute_sentence_threshold_curves(
    samples: List[Dict],
    n_thresholds: int = 50,
) -> List[Dict]:
    """
    Compute P/R/F1 curves for sentence-level raw predictions.

    Score signals:
      hal_prob_max_default      — max(contradiction+neutral) over facts
      hal_prob_mean_default     — mean(contradiction+neutral) over facts
      hal_prob_max_contradiction — max(raw contradiction) over facts
      hal_prob_mean_contradiction — mean(raw contradiction) over facts
      n_hallucinations          — count of facts with hal_prob_default > 0.5
                                   (threshold sweeps integer counts)

    Invalid-context samples contribute a fixed score of 0.5 to every metric.
    Samples with no facts contribute 0.0.

    Returns list of dicts: {metric, threshold, precision, recall, f1}
    """
    golds = [int(s["gold"]) for s in samples]
    thresholds = _default_thresholds(n_thresholds)
    rows: List[Dict] = []

    def _sample_score(s: Dict, fn) -> float:
        if s.get("invalid_context"):
            return 0.5
        fns = s.get("fact_nli_scores") or []
        if not fns:
            return 0.0
        # filter to argument spans only (sentence cache always stores arguments)
        arg_fns = [fs for fs in fns if fs.get("span_kind", "argument") == "argument"]
        if not arg_fns:
            arg_fns = fns
        return fn(arg_fns)

    # --- Continuous metrics ---
    continuous: List[Tuple[str, Any]] = [
        (
            "hal_prob_max_default",
            lambda fns: max((_hal_prob_from_nli(fs, "default") for fs in fns), default=0.0),
        ),
        (
            "hal_prob_mean_default",
            lambda fns: float(np.mean([_hal_prob_from_nli(fs, "default") for fs in fns])),
        ),
        (
            "hal_prob_max_contradiction",
            lambda fns: max((float(fs.get("contradiction", 0.0)) for fs in fns), default=0.0),
        ),
        (
            "hal_prob_mean_contradiction",
            lambda fns: float(np.mean([float(fs.get("contradiction", 0.0)) for fs in fns])),
        ),
    ]

    for metric_name, fn in continuous:
        scores = [_sample_score(s, fn) for s in samples]
        rows.extend(_sweep(golds, scores, metric_name, thresholds, macro=True))

    # --- n_hallucinations: integer count, sweep -0.5, 0.5, 1.5, ... ---
    n_hal_scores: List[float] = []
    for s in samples:
        if s.get("invalid_context"):
            # treat as 0 hallucinations (score = 0) so it never triggers
            n_hal_scores.append(0.0)
        else:
            fns = s.get("fact_nli_scores") or []
            arg_fns = [fs for fs in fns if fs.get("span_kind", "argument") == "argument"] or fns
            count = sum(
                1 for fs in arg_fns if _hal_prob_from_nli(fs, "default") > 0.5
            )
            n_hal_scores.append(float(count))

    max_count = int(max(n_hal_scores)) if n_hal_scores else 0
    # threshold=k-0.5 → predict 1 if count >= k
    count_thresholds = [k - 0.5 for k in range(0, max_count + 2)]
    rows.extend(_sweep(golds, n_hal_scores, "n_hallucinations", count_thresholds, macro=True))

    # --- min_triple_conf: filter low-confidence ModernOpenIE triples ---
    # Only emitted when at least one sample has triple_conf values saved.
    has_triple_conf = any(
        any("triple_conf" in fs for fs in (s.get("fact_nli_scores") or []))
        for s in samples
    )
    if has_triple_conf:
        for metric_name, nli_mode in [
            ("min_triple_conf_default", "default"),
            ("min_triple_conf_contradiction", "contradiction_only"),
        ]:
            for min_conf in thresholds:
                scores_at_conf = []
                for s in samples:
                    if s.get("invalid_context"):
                        scores_at_conf.append(0.5)
                        continue
                    fns = s.get("fact_nli_scores") or []
                    arg_fns = [
                        fs for fs in fns
                        if fs.get("span_kind", "argument") == "argument"
                        and fs.get("triple_conf", 1.0) >= min_conf
                    ] or [fs for fs in fns if fs.get("triple_conf", 1.0) >= min_conf]
                    if not arg_fns:
                        scores_at_conf.append(0.0)
                    else:
                        scores_at_conf.append(
                            max(_hal_prob_from_nli(fs, nli_mode) for fs in arg_fns)
                        )
                # Binary classify with fixed NLI threshold 0.5
                preds_bin = [int(sc >= 0.5) for sc in scores_at_conf]
                from sklearn.metrics import precision_score, recall_score, f1_score
                rows.append({
                    "metric": metric_name,
                    "threshold": float(min_conf),
                    "precision": float(precision_score(golds, preds_bin, zero_division=0)),
                    "recall":    float(recall_score(golds, preds_bin, zero_division=0)),
                    "f1":        float(f1_score(golds, preds_bin, zero_division=0)),
                })

    return rows


# =============================================================================
# Incremental group threshold helper
# =============================================================================

def _has_incremental_groups(samples: List[Dict]) -> bool:
    """True if any fact_span in any sample carries a non-None group_info."""
    return any(
        any(fs.get("group_info") is not None for fs in (s.get("fact_spans") or []))
        for s in samples
    )


def _incremental_diff_span(
    curr: List[int],
    prev: Optional[List[int]],
) -> List[int]:
    """Return the new region added by curr relative to prev.

    Three cases:
    - No previous fact: return curr as-is.
    - Shared start (argument extension): prev=[s,e0], curr=[s,e1] → diff=[e0,e1].
    - Non-overlapping (predicate-extension): diff=curr.
    Anything else (e.g. shared end, overlap without shared boundary): diff=curr.
    """
    if prev is None:
        return curr
    cs, ce = int(curr[0]), int(curr[1])
    ps, pe = int(prev[0]), int(prev[1])
    if cs == ps and pe < ce:
        return [pe, ce]
    return curr


def _apply_incremental_group_threshold(
    fact_spans: List[Dict],
    threshold: float,
    mode: str = "default",
) -> List[List[int]]:
    """Apply threshold with incremental early-stopping per group.

    For each group (identified by group_info[0]), iterate facts in position
    order.  At the first fact whose hall_prob exceeds the threshold, emit the
    *diff* span — the new region relative to the previous step — then stop.
    Facts without group_info are treated independently (normal threshold).
    """
    from collections import defaultdict

    groups: Dict[int, List[Dict]] = defaultdict(list)
    ungrouped: List[Dict] = []

    for fs in fact_spans:
        gi = fs.get("group_info")
        if gi is not None:
            groups[gi[0]].append(fs)
        else:
            ungrouped.append(fs)

    pred_spans: List[List[int]] = []

    for fs in ungrouped:
        if _hal_prob_from_nli(fs, mode) > threshold:
            pred_spans.append([int(fs["start"]), int(fs["end"])])

    for _, facts in groups.items():
        # facts are in insertion order = group position order
        prev_span: Optional[List[int]] = None
        for fs in facts:
            curr_span = [int(fs["start"]), int(fs["end"])]
            if _hal_prob_from_nli(fs, mode) > threshold:
                pred_spans.append(_incremental_diff_span(curr_span, prev_span))
                break  # stop at first invalid fact in this group
            prev_span = curr_span

    return pred_spans


# =============================================================================
# Span-level threshold curves
# =============================================================================

def compute_span_threshold_curves(
    samples: List[Dict],
    n_thresholds: int = 50,
    hall_prob_mode: str = "default",
    skip_2d_sweep: bool = False,
) -> List[Dict]:
    """
    Compute P/R/F1 curves for span-level raw predictions.

    Sweeps hal_prob threshold: a fact span is predicted hallucinated iff
    its hall_prob > threshold. Uses span-coverage micro-F1.

    Returns rows with threshold, span-coverage precision/recall/F1, and mean
    character-level IoU. F1 remains the primary metric for model selection.
    """
    from evaluation.span_metrics import span_coverage_micro, span_iou_macro

    golds = [s["gold_spans"] for s in samples]
    thresholds = _default_thresholds(n_thresholds)
    rows: List[Dict] = []

    use_incremental = _has_incremental_groups(samples)

    for t in thresholds:
        preds = []
        for s in samples:
            fact_spans = s.get("fact_spans") or s.get("facts") or []
            if use_incremental:
                pred_spans = _apply_incremental_group_threshold(fact_spans, t, hall_prob_mode)
            else:
                pred_spans = [
                    [int(fs["start"]), int(fs["end"])]
                    for fs in fact_spans
                    if _hal_prob_from_nli(fs, mode=hall_prob_mode) > t
                ]
            preds.append(pred_spans)

        result = span_coverage_micro(golds, preds)
        rows.append(
            {
                "metric": "span_hal_prob",
                "threshold": float(t),
                "precision": float(result.precision),
                "recall": float(result.recall),
                "f1": float(result.fbeta),
                "iou": float(span_iou_macro(golds, preds)),
            }
        )

    # --- min_triple_conf + joint 2D sweep ---
    # Only emitted when at least one sample has triple_conf values saved.
    # Skipped during calibration (skip_2d_sweep=True) since calibrate_span_threshold
    # only uses span_hal_prob rows; 2D grid at n_thresholds=200 would be 645M iterations.
    has_triple_conf = (not skip_2d_sweep) and any(
        any("triple_conf" in fs for fs in (s.get("fact_spans") or []))
        for s in samples
    )
    if has_triple_conf:
        for t in thresholds:
            preds = []
            for s in samples:
                pred_spans = [
                    [int(fs["start"]), int(fs["end"])]
                    for fs in (s.get("fact_spans") or [])
                    if fs.get("triple_conf", 1.0) >= t
                    and _hal_prob_from_nli(fs, mode=hall_prob_mode) > 0.5
                ]
                preds.append(pred_spans)

            result = span_coverage_micro(golds, preds)
            rows.append(
                {
                    "metric": "min_triple_conf",
                    "threshold": float(t),
                    "precision": float(result.precision),
                    "recall": float(result.recall),
                    "f1": float(result.fbeta),
                    "iou": float(span_iou_macro(golds, preds)),
                }
            )

        # 2D joint sweep: all (conf_threshold, nli_threshold) pairs
        for conf_t in thresholds:
            for nli_t in thresholds:
                preds = []
                for s in samples:
                    pred_spans = [
                        [int(fs["start"]), int(fs["end"])]
                        for fs in (s.get("fact_spans") or [])
                        if fs.get("triple_conf", 1.0) >= conf_t
                        and _hal_prob_from_nli(fs, mode=hall_prob_mode) > nli_t
                    ]
                    preds.append(pred_spans)

                result = span_coverage_micro(golds, preds)
                rows.append(
                    {
                        "metric": "joint",
                        "threshold": float(nli_t),
                        "conf_threshold": float(conf_t),
                        "precision": float(result.precision),
                        "recall": float(result.recall),
                        "f1": float(result.fbeta),
                        "iou": float(span_iou_macro(golds, preds)),
                    }
                )

    return rows


# =============================================================================
# Threshold calibration helpers
# =============================================================================

def calibrate_span_threshold(
    samples: List[Dict],
    n_thresholds: int = 200,
    hall_prob_mode: str = "default",
) -> Tuple[float, Dict]:
    """Find the threshold with best span-coverage micro-F1 on the given samples.

    Returns ``(best_threshold, row_dict)``. Selection is based on span-coverage
    F1; the row also contains IoU as a secondary metric.
    """
    rows = compute_span_threshold_curves(samples, n_thresholds=n_thresholds, hall_prob_mode=hall_prob_mode, skip_2d_sweep=True)
    span_rows = [r for r in rows if r["metric"] == "span_hal_prob"]
    best = max(span_rows, key=lambda r: r["f1"])
    return best["threshold"], best


def save_calibrated_threshold(path: Path, data: Dict) -> None:
    """Persist calibrated threshold metadata to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved calibrated threshold to {path}")


def load_calibrated_threshold(path: Path) -> Optional[Dict]:
    """Load calibrated threshold metadata. Returns None if file does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# =============================================================================
# Entity-level threshold curves
# =============================================================================

def compute_entity_threshold_curves(
    samples: List[Dict],
    n_thresholds: int = 50,
) -> List[Dict]:
    """
    Compute P/R/F1 curves for entity-level raw predictions.

    Flattens all entities across samples and sweeps threshold on entity score.

    Returns list of dicts: {metric, threshold, precision, recall, f1}
    """
    all_golds: List[int] = []
    all_scores: List[float] = []
    for s in samples:
        all_golds.extend(int(g) for g in s["gold_labels"])
        all_scores.extend(float(sc) for sc in s["entity_scores"])

    thresholds = _default_thresholds(n_thresholds)
    return _sweep(all_golds, all_scores, "entity_hal_prob", thresholds)


# =============================================================================
# Display
# =============================================================================

def print_curves_summary(rows: List[Dict], full: bool = False) -> None:
    """Print threshold curve table.

    full=False (default): one row per metric showing best-F1 point.
    full=True: all threshold points per metric, best-F1 row marked with *.
    """
    if not rows:
        print("No curve data.")
        return

    by_metric: Dict[str, List[Dict]] = {}
    for r in rows:
        by_metric.setdefault(r["metric"], []).append(r)

    has_iou = any("iou" in row for row in rows)
    iou_header = f" {'IoU':<10}" if has_iou else ""
    header = (
        f"\n{'Metric':<38} {'Precision':<10} {'Recall':<10} "
        f"{'F1':<10}{iou_header} {'Threshold'}"
    )
    sep = "-" * (89 if has_iou else 78)

    joint_rows = by_metric.pop("joint", None)
    by_metric.pop("min_triple_conf", None)

    for metric_name in sorted(by_metric):
        metric_rows = by_metric[metric_name]
        best = max(metric_rows, key=lambda x: x["f1"])

        if not full:
            if metric_name == sorted(by_metric)[0]:
                print(header)
                print(sep)
            print(
                f"{metric_name:<38} {best['precision']:<10.4f}"
                f" {best['recall']:<10.4f} {best['f1']:<10.4f}"
                + (f" {best['iou']:<10.4f}" if "iou" in best else "")
                + f" {best['threshold']:.4f}"
            )
        else:
            print(f"\n=== {metric_name} ===")
            full_iou_header = f" {'IoU':<10}" if has_iou else ""
            print(
                f"{'Threshold':<12} {'Precision':<10} {'Recall':<10} "
                f"{'F1':<10}{full_iou_header}"
            )
            print("-" * (57 if has_iou else 44))
            for r in sorted(metric_rows, key=lambda x: x["threshold"]):
                marker = " *" if r is best else ""
                print(
                    f"{r['threshold']:<12.4f} {r['precision']:<10.4f}"
                    f" {r['recall']:<10.4f} {r['f1']:<10.4f}"
                    + (f" {r['iou']:<10.4f}" if "iou" in r else "")
                    + marker
                )

    if joint_rows:
        best = max(joint_rows, key=lambda x: x["f1"])
        print(f"\nBest joint (conf x nli):  P={best['precision']:.4f}  R={best['recall']:.4f}"
              f"  F1={best['f1']:.4f}  IoU={best['iou']:.4f}"
              f"  conf>={best['conf_threshold']:.4f}  nli>{best['threshold']:.4f}")


# =============================================================================
# Main entry point for the `threshold` CLI command
# =============================================================================

def run_threshold_analysis(
    preds_path: Path,
    n_thresholds: int = 50,
    output_path: Optional[Path] = None,
    full: bool = False,
    hall_prob_mode: str = "default",
) -> None:
    """
    Load raw predictions and compute threshold curves.

    Args:
        preds_path: Path to *_preds.json file
        n_thresholds: Number of threshold grid points
        output_path: Optional path to save curves CSV
    """
    data = load_raw_predictions(preds_path)
    if isinstance(data, list):
        data = {"type": "span", "samples": data}
    pred_type = data.get("type")
    samples = data.get("samples", [])
    dataset = data.get("dataset", "unknown")
    method = data.get("method", "unknown")
    coref = data.get("coref_enabled")

    coref_str = f", coref={'yes' if coref else 'no'}" if coref is not None else ""
    print(f"Loaded {len(samples)} samples  (type={pred_type}, dataset={dataset}, method={method}{coref_str})")

    if hall_prob_mode != "default":
        print(f"hall_prob_mode: {hall_prob_mode}")
    if pred_type == "sentence":
        rows = compute_sentence_threshold_curves(samples, n_thresholds=n_thresholds)
    elif pred_type == "span":
        rows = compute_span_threshold_curves(samples, n_thresholds=n_thresholds, hall_prob_mode=hall_prob_mode)
    elif pred_type == "entity":
        rows = compute_entity_threshold_curves(samples, n_thresholds=n_thresholds)
    else:
        raise ValueError(f"Unknown prediction type: {pred_type!r}")

    print_curves_summary(rows, full=full)

    if output_path is not None:
        save_curves_csv(output_path, rows)
