"""Compatibility facade over the canonical Mycelium span scorer.

All span metrics in Enoki are delegated to ``mycelium-scorer``.  Spans use
half-open character offsets: ``[start, end)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

Span = Tuple[int, int]
SpanSets = Sequence[Sequence[Sequence[int]]]


@dataclass(frozen=True)
class SpanCoveragePRF:
    """Span Coverage F1 values returned by Mycelium.

    Mycelium intentionally exposes aggregate scores rather than implementation
    counts, so the count fields remain unavailable in this compatibility type.
    """

    precision: float
    recall: float
    fbeta: float


def _validate_batch_lengths(golds: SpanSets, preds: SpanSets) -> None:
    if len(golds) != len(preds):
        raise ValueError(
            "golds and preds must have same length, "
            f"got {len(golds)} vs {len(preds)}"
        )


def _metrics():
    """Import the canonical metric implementations from Mycelium."""
    try:
        from mycelium.metrics import iou, span_coverage
    except ModuleNotFoundError as exc:  # pragma: no cover - installation error
        raise ModuleNotFoundError(
            "Span evaluation requires mycelium-scorer. Install Enoki with its "
            "evaluation dependencies: pip install -e '.[eval]'"
        ) from exc
    return iou, span_coverage


def _evaluate_iou(
    golds: SpanSets,
    preds: SpanSets,
    *,
    average: str,
) -> dict:
    _validate_batch_lengths(golds, preds)
    iou, _ = _metrics()
    return iou(golds, preds, average=average)


def span_iou_one(gold: List[List[int]], pred: List[List[int]]) -> float:
    """Return Mycelium character-level IoU for one example."""
    return float(_evaluate_iou([gold], [pred], average="macro")["score"])


def span_iou_macro(
    golds: SpanSets,
    preds: SpanSets,
    *,
    empty_is_perfect: bool = True,
) -> float:
    """Return Mycelium macro character-level IoU."""
    _validate_batch_lengths(golds, preds)
    if not golds:
        return 1.0 if empty_is_perfect else 0.0
    return float(_evaluate_iou(golds, preds, average="macro")["score"])


def _span_coverage(
    golds: SpanSets,
    preds: SpanSets,
    *,
    average: str,
    delta: int,
    min_pred_len: int,
    beta: float,
    empty_is_perfect: bool,
) -> SpanCoveragePRF:
    if beta != 1.0:
        raise ValueError("mycelium-scorer currently provides Span Coverage F1 only (beta=1.0)")
    _validate_batch_lengths(golds, preds)
    if not golds:
        value = 1.0 if empty_is_perfect else 0.0
        return SpanCoveragePRF(value, value, value)

    _, span_coverage = _metrics()
    scores = span_coverage(
        golds,
        preds,
        average=average,
        delta=delta,
        min_pred_len=min_pred_len,
        empty_is_perfect=empty_is_perfect,
    )
    return SpanCoveragePRF(
        precision=float(scores["precision"]),
        recall=float(scores["recall"]),
        fbeta=float(scores["f1"]),
    )


def span_coverage_micro(
    golds: SpanSets,
    preds: SpanSets,
    *,
    delta: int = 0,
    min_pred_len: int = 1,
    beta: float = 1.0,
    empty_is_perfect: bool = True,
) -> SpanCoveragePRF:
    """Return micro-averaged Span Coverage F1 from Mycelium."""
    return _span_coverage(
        golds,
        preds,
        average="micro",
        delta=delta,
        min_pred_len=min_pred_len,
        beta=beta,
        empty_is_perfect=empty_is_perfect,
    )


def span_coverage_macro(
    golds: SpanSets,
    preds: SpanSets,
    *,
    delta: int = 0,
    min_pred_len: int = 1,
    beta: float = 1.0,
    empty_is_perfect: bool = True,
) -> SpanCoveragePRF:
    """Return macro-averaged Span Coverage F1 from Mycelium."""
    return _span_coverage(
        golds,
        preds,
        average="macro",
        delta=delta,
        min_pred_len=min_pred_len,
        beta=beta,
        empty_is_perfect=empty_is_perfect,
    )
