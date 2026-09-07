"""Cluster false negatives by dependency-pattern signature.

Each gold triplet the ruleset missed gets a coarse structural signature
``(predicate POS, predicate tag, subject dep, argument dep, prep, voice)``
so that similar constructions land in the same cluster.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from ..evaluation.qasrl_to_spo import GoldTriplet
from ..models import Triplet
from ..preprocess import Parser


@dataclass(frozen=True)
class FNCase:
    """One false negative gold triplet."""

    sentence_id: str
    sentence_text: str
    gold: GoldTriplet
    signature: Tuple[str, ...]


@dataclass
class FNCluster:
    """A cluster of false negatives sharing a dependency-pattern signature."""

    signature: Tuple[str, ...]
    cases: List[FNCase] = field(default_factory=list)
    cooldown_until_iter: int = 0

    @property
    def size(self) -> int:
        return len(self.cases)


def _gold_predicate_token(parser: Parser, sentence: str, verb_idx: int):
    """Parse the sentence and return the spaCy token at ``verb_idx``."""
    doc = parser.parse(sentence)
    tokens = list(doc)
    if 0 <= verb_idx < len(tokens):
        return tokens[verb_idx]
    return None


def _signature_for(parser: Parser, gold: GoldTriplet) -> Tuple[str, ...]:
    sentence = " ".join(gold.tokens)
    tok = _gold_predicate_token(parser, sentence, gold.predicate_verb_index)
    if tok is None:
        return ("unknown",) * 6
    voice = "passive" if gold.is_passive else "active"
    arg_dep = "_"
    if gold.argument_span is not None and tok.doc is not None:
        arg_start = gold.argument_span[0]
        if 0 <= arg_start < len(tok.doc):
            arg_dep = tok.doc[arg_start].dep_
    subj_dep = "_"
    subj_start = gold.subject_span[0]
    if 0 <= subj_start < len(tok.doc):
        subj_dep = tok.doc[subj_start].dep_
    return (
        tok.pos_,
        tok.tag_,
        subj_dep,
        arg_dep,
        gold.prep or "_",
        voice,
    )


def cluster_false_negatives(
    per_sentence: Iterable[Tuple[str, List[Triplet], List[GoldTriplet]]],
    matched_flags: Dict[Tuple[str, int], bool],
    parser: Optional[Parser] = None,
) -> List[FNCluster]:
    """Group unmatched gold triplets into clusters keyed by signature.

    ``matched_flags`` maps ``(sentence_id, gold_index)`` → True if the gold
    triplet was matched by the current pipeline. Anything False (or missing)
    becomes a false-negative case.
    """
    parser = parser or Parser()
    by_sig: Dict[Tuple[str, ...], FNCluster] = defaultdict(
        lambda: FNCluster(signature=())
    )
    for sentence_id, _preds, golds in per_sentence:
        for idx, gold in enumerate(golds):
            if matched_flags.get((sentence_id, idx), False):
                continue
            sig = _signature_for(parser, gold)
            case = FNCase(
                sentence_id=sentence_id,
                sentence_text=" ".join(gold.tokens),
                gold=gold,
                signature=sig,
            )
            cluster = by_sig[sig]
            if not cluster.signature:
                cluster = FNCluster(signature=sig)
                by_sig[sig] = cluster
            cluster.cases.append(case)
    return sorted(by_sig.values(), key=lambda c: c.size, reverse=True)
