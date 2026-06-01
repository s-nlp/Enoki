"""Canonical-key dedup over a triplet pool.

Two triplets are duplicates if they share
``(subject lemma, predicate lemma + prep, argument lemma, role)``.
When duplicates exist, keep the one with the highest confidence; break ties
by larger argument span (more-specific extraction wins).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from ..models import Triplet


def _lemma_text(span) -> str:
    return " ".join(t.lemma_.lower() for t in span if not t.is_punct)


def _key(triplet: "Triplet") -> Tuple[str, str, str, str]:
    subj_k = _lemma_text(triplet.subject)
    # Prefer the rendered predicate text (handles synthesized 'is' for
    # appositives) over the raw span lemmas.
    if triplet.predicate_text:
        pred_k = triplet.predicate_text.lower()
    else:
        pred_k = _lemma_text(triplet.predicate)
    if triplet.argument is None:
        arg_k = ""
        role = ""
    else:
        arg_k = _lemma_text(triplet.argument.span)
        prep = triplet.argument.prep or ""
        if prep:
            pred_k = f"{pred_k} {prep.lower()}"
        role = triplet.argument.role
    return (subj_k, pred_k, arg_k, role)


def _is_better(candidate: "Triplet", incumbent: "Triplet") -> bool:
    if candidate.confidence != incumbent.confidence:
        return candidate.confidence > incumbent.confidence
    cand_arg_len = 0 if candidate.argument is None else len(candidate.argument.span)
    inc_arg_len = 0 if incumbent.argument is None else len(incumbent.argument.span)
    return cand_arg_len > inc_arg_len


def dedup_triplets(triplets: List["Triplet"]) -> List["Triplet"]:
    best: Dict[Tuple[str, str, str, str], "Triplet"] = {}
    for t in triplets:
        k = _key(t)
        if k not in best or _is_better(t, best[k]):
            best[k] = t
    return list(best.values())
