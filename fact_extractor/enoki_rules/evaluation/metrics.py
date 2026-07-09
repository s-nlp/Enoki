"""Token-level OIE metrics.

A predicted :class:`Triplet` matches a :class:`GoldTriplet` when:

- subject token-overlap ratio ≥ ``min_overlap``;
- argument token-overlap ratio ≥ ``min_overlap`` (or both args are absent);
- predicate token-overlap ratio ≥ ``min_overlap`` measured on the predicate
  surface text (rendered to a multiset of tokens; case-folded).

We aggregate token-level Precision/Recall/F1 across the dev set and report
**predicate coverage** (fraction of gold predicates with at least one
matching prediction) as a separate signal. The optimization loop's score
is ``S = F1 + λ · predicate_coverage`` (PLAN.md §6.2).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

from .qasrl_to_spo import GoldTriplet

if TYPE_CHECKING:
    from ..models import Triplet


def _tokenize(s: str) -> List[str]:
    return s.lower().split()


def _token_overlap(a: List[str], b: List[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    set_a, set_b = set(a), set(b)
    inter = len(set_a & set_b)
    denom = max(1, min(len(set_a), len(set_b)))
    return inter / denom


def _gold_subject_tokens(g: GoldTriplet) -> List[str]:
    return _tokenize(g.subject_text)


def _gold_argument_tokens(g: GoldTriplet) -> List[str]:
    return _tokenize(g.argument_text) if g.argument_text else []


def _gold_predicate_tokens(g: GoldTriplet) -> List[str]:
    return _tokenize(g.predicate_text)


def _pred_subject_tokens(t: "Triplet") -> List[str]:
    return _tokenize(t.subject.text)


def _pred_argument_tokens(t: "Triplet") -> List[str]:
    if t.argument is None:
        return []
    return _tokenize(t.argument.span.text)


def _pred_predicate_tokens(t: "Triplet") -> List[str]:
    return _tokenize(t.predicate_surface)


def is_match(prediction: "Triplet", gold: GoldTriplet, min_overlap: float = 0.5) -> bool:
    """Return True if ``prediction`` covers ``gold`` per the metric definition."""
    if _token_overlap(_pred_subject_tokens(prediction), _gold_subject_tokens(gold)) < min_overlap:
        return False
    if _token_overlap(
        _pred_predicate_tokens(prediction), _gold_predicate_tokens(gold)
    ) < min_overlap:
        return False
    gold_arg = _gold_argument_tokens(gold)
    pred_arg = _pred_argument_tokens(prediction)
    if not gold_arg and not pred_arg:
        return True
    if not gold_arg or not pred_arg:
        return False
    return _token_overlap(pred_arg, gold_arg) >= min_overlap


@dataclass
class EvalReport:
    """Aggregate evaluation result over a corpus."""

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    predicate_coverage: float = 0.0
    score: float = 0.0  # F1 + lambda * predicate_coverage

    tp: int = 0
    fp: int = 0
    fn: int = 0

    # Per-rule contribution counts (source_rule -> {tp, fp}).
    per_rule_tp: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    per_rule_fp: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def as_dict(self) -> Dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "predicate_coverage": self.predicate_coverage,
            "score": self.score,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "per_rule_tp": dict(self.per_rule_tp),
            "per_rule_fp": dict(self.per_rule_fp),
        }


def score_against_gold(
    per_sentence_preds: Iterable[Tuple[str, List["Triplet"], List[GoldTriplet]]],
    score_lambda: float = 0.25,
    min_overlap: float = 0.5,
) -> EvalReport:
    """Score predictions against gold across many sentences.

    ``per_sentence_preds`` is an iterable of
    ``(sentence_id, predicted_triplets, gold_triplets)``. Returns a
    :class:`EvalReport`.
    """
    report = EvalReport()
    predicate_indices_total = 0
    predicate_indices_covered = 0

    for _sid, preds, golds in per_sentence_preds:
        gold_matched = [False] * len(golds)
        pred_matched = [False] * len(preds)
        for i, p in enumerate(preds):
            for j, g in enumerate(golds):
                if gold_matched[j]:
                    continue
                if is_match(p, g, min_overlap=min_overlap):
                    gold_matched[j] = True
                    pred_matched[i] = True
                    report.tp += 1
                    if p.source_rule:
                        report.per_rule_tp[p.source_rule] += 1
                    break
        for i, p in enumerate(preds):
            if not pred_matched[i]:
                report.fp += 1
                if p.source_rule:
                    report.per_rule_fp[p.source_rule] += 1
        for j, matched in enumerate(gold_matched):
            if not matched:
                report.fn += 1

        # Predicate-level coverage: each distinct gold predicate index is
        # counted once; covered iff at least one prediction matched any
        # gold triplet under that predicate.
        predicates_in_sentence = {g.predicate_verb_index for g in golds}
        for verb_idx in predicates_in_sentence:
            predicate_indices_total += 1
            verb_gold_positions = [
                k for k, g in enumerate(golds) if g.predicate_verb_index == verb_idx
            ]
            if any(gold_matched[k] for k in verb_gold_positions):
                predicate_indices_covered += 1

    if report.tp + report.fp > 0:
        report.precision = report.tp / (report.tp + report.fp)
    if report.tp + report.fn > 0:
        report.recall = report.tp / (report.tp + report.fn)
    if report.precision + report.recall > 0:
        report.f1 = (
            2 * report.precision * report.recall / (report.precision + report.recall)
        )
    if predicate_indices_total > 0:
        report.predicate_coverage = predicate_indices_covered / predicate_indices_total
    report.score = report.f1 + score_lambda * report.predicate_coverage
    return report
