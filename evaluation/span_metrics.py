from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

Span = Tuple[int, int]  # inclusive [start, end]


def _normalize_spans(spans: List[List[int]]) -> List[Span]:
    out: List[Span] = []
    for x in spans:
        if len(x) != 2:
            raise ValueError(f"Bad span (expected [start,end]): {x}")
        s, e = int(x[0]), int(x[1])
        if e < s:
            raise ValueError(f"Bad span with end < start: {x}")
        out.append((s, e))
    return out


def _len_inc(sp: Span) -> int:
    return sp[1] - sp[0] + 1


def _contained(pred: Span, gold: Span, delta: int = 0) -> bool:
    ps, pe = pred
    gs, ge = gold
    return (gs - delta) <= ps and pe <= (ge + delta)


@dataclass
class SpanCoveragePRF:
    contained_preds: int
    total_preds: int
    hit_golds: int
    total_golds: int

    precision: float
    recall: float
    fbeta: float


def span_coverage_counts_one(
    gold: List[List[int]],
    pred: List[List[int]],
    *,
    delta: int = 0,
    min_pred_len: int = 1,
) -> Tuple[int, int, int, int]:
    g = _normalize_spans(gold)
    p = _normalize_spans(pred)

    if min_pred_len > 1:
        p = [sp for sp in p if _len_inc(sp) >= min_pred_len]

    g.sort(key=lambda x: (x[0], x[1]))
    p.sort(key=lambda x: (x[0], x[1]))

    contained_preds = 0
    for ps in p:
        ok = False
        for gs in g:
            if _contained(ps, gs, delta=delta):
                ok = True
                break
        contained_preds += 1 if ok else 0

    hit_golds = 0
    for gs in g:
        hit = False
        for ps in p:
            if _contained(ps, gs, delta=delta):
                hit = True
                break
        hit_golds += 1 if hit else 0

    return contained_preds, len(p), hit_golds, len(g)


def _fbeta_from_pr(precision: float, recall: float, beta: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def span_coverage_micro(
    golds: List[List[List[int]]],
    preds: List[List[List[int]]],
    *,
    delta: int = 0,
    min_pred_len: int = 1,
    beta: float = 1.0,
    empty_is_perfect: bool = True,
) -> SpanCoveragePRF:
    if len(golds) != len(preds):
        raise ValueError(f"golds and preds must have same length, got {len(golds)} vs {len(preds)}")

    contained_preds = total_preds = hit_golds = total_golds = 0

    for g, p in zip(golds, preds):
        cp, tp, hg, tg = span_coverage_counts_one(g, p, delta=delta, min_pred_len=min_pred_len)
        contained_preds += cp
        total_preds += tp
        hit_golds += hg
        total_golds += tg

    if total_preds == 0 and total_golds == 0:
        val = 1.0 if empty_is_perfect else 0.0
        return SpanCoveragePRF(0, 0, 0, 0, val, val, val)

    precision = (contained_preds / total_preds) if total_preds > 0 else 1.0
    recall = (hit_golds / total_golds) if total_golds > 0 else 1.0
    fbeta = _fbeta_from_pr(precision, recall, beta=beta)

    return SpanCoveragePRF(contained_preds, total_preds, hit_golds, total_golds, precision, recall, fbeta)


def span_coverage_macro(
    golds: List[List[List[int]]],
    preds: List[List[List[int]]],
    *,
    delta: int = 0,
    min_pred_len: int = 1,
    beta: float = 1.0,
    empty_is_perfect: bool = True,
) -> SpanCoveragePRF:
    """
    Macro aggregation (per-example, then averaged):
    - precision_i = contained_preds_i / total_preds_i   (or 1.0 if total_preds_i == 0)
    - recall_i    = hit_golds_i       / total_golds_i   (or 1.0 if total_golds_i == 0)
    - fbeta_i computed from (precision_i, recall_i)

    Final precision/recall/fbeta are arithmetic means of per-example values.
    """
    if len(golds) != len(preds):
        raise ValueError(f"golds and preds must have same length, got {len(golds)} vs {len(preds)}")

    if not golds:
        val = 1.0 if empty_is_perfect else 0.0
        return SpanCoveragePRF(0, 0, 0, 0, val, val, val)

    precisions: List[float] = []
    recalls: List[float] = []
    fbetas: List[float] = []

    for g, p in zip(golds, preds):
        cp, tp, hg, tg = span_coverage_counts_one(g, p, delta=delta, min_pred_len=min_pred_len)

        if tp == 0 and tg == 0:
            pr = 1.0 if empty_is_perfect else 0.0
            rc = 1.0 if empty_is_perfect else 0.0
        else:
            pr = (cp / tp) if tp > 0 else 1.0
            rc = (hg / tg) if tg > 0 else 1.0
        fb = _fbeta_from_pr(pr, rc, beta=beta)

        precisions.append(pr)
        recalls.append(rc)
        fbetas.append(fb)

    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    fbeta = sum(fbetas) / len(fbetas)
    return SpanCoveragePRF(0, 0, 0, 0, precision, recall, fbeta)
