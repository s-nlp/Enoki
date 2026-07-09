"""
Unified metrics module for all evaluation types.

Provides metric calculation functions for:
- Span-level: Span coverage F1 (exact match and containment-based)
- Sentence-level: ROC-AUC, Macro F1
- Entity-level: ROC-AUC, AUPRC
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    classification_report,
)

from evaluation.span_metrics import (
    span_coverage_micro,
    span_coverage_macro,
    SpanCoveragePRF,
)


# =============================================================================
# Entity-level metrics
# =============================================================================

def calculate_entity_metrics(
    y_true: List[int],
    y_score: List[float]
) -> Tuple[float, float]:
    """
    Calculate AUROC and AUPRC metrics for entity-level evaluation.

    Args:
        y_true: True labels (0 or 1)
        y_score: Predicted scores (0.0 to 1.0)

    Returns:
        Tuple of (auroc, auprc)
        Returns (0.0, auprc) if AUROC cannot be calculated (only one class)
    """
    if not y_true or not y_score:
        return 0.0, 0.0

    # AUROC requires at least 2 classes
    if np.unique(y_true).size == 2 and np.all(np.isfinite(y_score)):
        auroc = roc_auc_score(y_true, y_score)
    else:
        auroc = 0.0

    # AUPRC can work with single class
    auprc = average_precision_score(y_true, y_score)

    return auroc, auprc


def print_entity_metrics_summary(
    auroc_list: List[float],
    auprc_list: List[float],
    skipped: int = 0
):
    """
    Print summary of entity-level evaluation metrics.

    Args:
        auroc_list: List of AUROC scores
        auprc_list: List of AUPRC scores
        skipped: Number of skipped samples
    """
    print(f"Evaluated {len(auroc_list)} samples")
    if skipped > 0:
        print(f"Skipped {skipped} samples due to errors")
    print(f"Mean AUROC: {np.mean(auroc_list):.4f}")
    print(f"Mean AUPRC: {np.mean(auprc_list):.4f}")


# =============================================================================
# Sentence-level metrics
# =============================================================================

def calculate_sentence_metrics(
    y_true: List[int],
    y_score: List[float],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Calculate ROC-AUC and Macro F1 metrics for sentence-level evaluation.

    Args:
        y_true: True labels (0 or 1)
        y_score: Predicted scores (0.0 to 1.0)
        threshold: Classification threshold

    Returns:
        Dict with metrics including:
        - auroc: ROC-AUC score
        - f1_macro: Macro F1 score
        - threshold: Classification threshold used
        - classification_report: Detailed classification report
    """
    if not y_true or not y_score:
        return {
            'auroc': 0.0,
            'f1_macro': 0.0,
            'threshold': threshold,
            'classification_report': {},
        }

    y_true_arr = np.array(y_true)
    y_score_arr = np.array(y_score)

    # ROC-AUC
    if len(np.unique(y_true_arr)) >= 2:
        auroc = roc_auc_score(y_true_arr, y_score_arr)
    else:
        auroc = 0.0

    # F1 at threshold
    binary_preds = (y_score_arr > threshold).astype(int)
    f1_macro = f1_score(y_true_arr, binary_preds, average='macro')

    # Classification report
    report = classification_report(
        y_true_arr,
        binary_preds,
        digits=4,
        output_dict=True,
        zero_division=0
    )

    return {
        'auroc': auroc,
        'f1_macro': f1_macro,
        'threshold': threshold,
        'classification_report': report,
    }


def print_sentence_metrics_summary(metrics: Dict[str, Any]):
    """
    Print summary of sentence-level evaluation metrics.

    Args:
        metrics: Dict returned by calculate_sentence_metrics
    """
    print(f"ROC-AUC: {metrics['auroc']:.4f}")
    print(f"F1 Macro: {metrics['f1_macro']:.4f} (threshold={metrics['threshold']:.2f})")
    if 'classification_report' in metrics and metrics['classification_report'].get('macro avg'):
        report = metrics['classification_report']
        print("\nClassification Report:")
        print(f"  Macro F1: {report['macro avg']['f1-score']:.4f}")
        print(f"  Precision: {report['macro avg']['precision']:.4f}")
        print(f"  Recall: {report['macro avg']['recall']:.4f}")


# =============================================================================
# Span-level metrics
# =============================================================================

def calculate_span_f1(
    gold_spans: List[List[List[int]]],
    pred_spans: List[List[List[int]]],
    delta: int = 0,
    min_pred_len: int = 1,
    aggregation: str = "micro",
) -> Dict[str, Any]:
    """
    Calculate span-level F1 metrics using sophisticated span coverage.

    This uses containment-based matching rather than exact match:
    - A predicted span is "contained" if it falls within a gold span (±delta tolerance)
    - A gold span is "hit" if at least one predicted span is contained within it

    Args:
        gold_spans: List of gold span lists for each sample [[start, end], ...]
        pred_spans: List of predicted span lists for each sample [[start, end], ...]
        delta: Tolerance for span boundaries (default: 0 for exact containment)
        min_pred_len: Minimum prediction span length to consider (default: 1)
        aggregation: "micro" (default) or "macro" averaging

    Returns:
        Dict with metrics:
        - precision: Precision score
        - recall: Recall score
        - f1: F1 score
        - contained_preds: Number of contained predictions (micro only)
        - total_preds: Total predictions (micro only)
        - hit_golds: Number of hit gold spans (micro only)
        - total_golds: Total gold spans (micro only)
    """
    if aggregation == "macro":
        result = span_coverage_macro(
            gold_spans,
            pred_spans,
            delta=delta,
            min_pred_len=min_pred_len,
        )
    else:
        result = span_coverage_micro(
            gold_spans,
            pred_spans,
            delta=delta,
            min_pred_len=min_pred_len,
        )

    metrics = {
        'precision': result.precision,
        'recall': result.recall,
        'f1': result.fbeta,
    }

    # Include raw counts for micro aggregation
    if aggregation == "micro":
        metrics.update({
            'contained_preds': result.contained_preds,
            'total_preds': result.total_preds,
            'hit_golds': result.hit_golds,
            'total_golds': result.total_golds,
        })

    return metrics


def calculate_span_f1_exact_match(
    gold_spans: List[List[List[int]]],
    pred_spans: List[List[List[int]]]
) -> Dict[str, float]:
    """
    Calculate span-level F1 metrics using exact match (legacy method).

    This is simpler but less sophisticated than calculate_span_f1.
    Spans must match exactly to be considered correct.

    Args:
        gold_spans: List of gold span lists for each sample [[start, end], ...]
        pred_spans: List of predicted span lists for each sample [[start, end], ...]

    Returns:
        Dict with metrics:
        - precision: Precision score
        - recall: Recall score
        - f1: F1 score
        - tp: True positives
        - fp: False positives
        - fn: False negatives
    """
    tp = 0
    fp = 0
    fn = 0

    for gold, pred in zip(gold_spans, pred_spans):
        # Convert to sets of tuples for comparison
        gold_set = set(tuple(span) for span in gold)
        pred_set = set(tuple(span) for span in pred)

        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }


def print_span_metrics_summary(metrics: Dict[str, Any]):
    """
    Print summary of span-level evaluation metrics.

    Args:
        metrics: Dict returned by calculate_span_f1
    """
    print(f"Span Coverage F1: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")

    # Print detailed counts if available (micro aggregation)
    if 'contained_preds' in metrics:
        print(f"Contained Predictions: {metrics['contained_preds']}/{metrics['total_preds']}")
        print(f"Hit Gold Spans: {metrics['hit_golds']}/{metrics['total_golds']}")
    elif 'tp' in metrics:
        # Legacy exact match format
        print(f"TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")


# =============================================================================
# Legacy compatibility
# =============================================================================

# Keep the old function name for backward compatibility
calculate_metrics = calculate_entity_metrics
print_metrics_summary = print_entity_metrics_summary
