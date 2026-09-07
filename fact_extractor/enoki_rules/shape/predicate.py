"""Predicate-span shaping: verb head plus auxiliaries, particle and bound preposition.

Negation tokens are excluded from the span and reported as a flag instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

from ..config import ShapeConfig

if TYPE_CHECKING:
    from spacy.tokens import Span, Token


_MODAL_LEMMAS = {"can", "could", "may", "might", "shall", "should",
                 "will", "would", "must", "ought"}


def shape_predicate(
    head: "Token",
    cfg: ShapeConfig,
    explicit_prep: Optional[str] = None,
) -> "Span":
    """Materialize the predicate span for verb ``head``.

    ``explicit_prep`` is a preposition the rule bound into the predicate
    (``"in"`` for ``"born in"``); the matching ``prep`` child is absorbed.
    """
    doc = head.doc
    indices: List[int] = [head.i]

    is_infinitival = head.dep_ in {"xcomp", "ccomp"} and any(
        c.dep_ == "aux" and c.tag_ == "TO" for c in head.children
    )
    for child in head.children:
        if child.dep_ == "prt":
            indices.append(child.i)
        if child.dep_ in {"aux", "auxpass"}:
            # Embedded infinitives keep the bare verb ("asked him to leave"
            # -> "leave"); every other auxiliary, modals included, is absorbed.
            if is_infinitival and child.tag_ == "TO":
                continue
            indices.append(child.i)
        if child.dep_ == "neg":
            continue

    if explicit_prep is not None:
        for child in head.children:
            if child.dep_ == "prep" and child.lower_ == explicit_prep:
                indices.append(child.i)
                break
            # Passive "by" is attached as dep_ "agent" by some spaCy models.
            if child.dep_ == "agent" and explicit_prep == "by":
                indices.append(child.i)
                break

    start = min(indices)
    end = max(indices) + 1
    # Forward expansion (particle, prep) is windowed; auxiliaries before the
    # verb are always kept in full.
    window = cfg.predicate_particle_window
    if end - head.i - 1 > window:
        end = head.i + 1 + window
    return doc[start:end]


def detect_negation_and_modality(head: "Token") -> Tuple[bool, Optional[str]]:
    """Inspect ``head`` for a ``neg`` child and a modal auxiliary."""
    negated = False
    modality: Optional[str] = None
    for child in head.children:
        if child.dep_ == "neg":
            negated = True
        if child.dep_ == "aux" and child.lemma_.lower() in _MODAL_LEMMAS:
            modality = child.lemma_.lower()
    return negated, modality
