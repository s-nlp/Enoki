"""Predicate-span shaping.

Predicate = verb head + attached particle + semantically-bound preposition.
Negation and modality are factored out and live on the :class:`Triplet` as
flags; they do NOT appear in the predicate span.
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
    """Materialize the predicate span for verb head ``head``.

    ``explicit_prep`` is the preposition the rule decided to bind into the
    predicate (e.g. ``"in"`` for ``"born in"``). If given, we look for a
    matching ``prep`` child of ``head`` and absorb it; otherwise the predicate
    is just verb + particle.
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
            # Drop the infinitive 'to' marker on embedded infinitival
            # predicates — gold uses the bare verb ("She asked him to
            # leave" → predicate "leave", "Air allows you to hear birds"
            # → "hear"). Keeps modals/tense aux as usual.
            if is_infinitival and child.tag_ == "TO":
                continue
            # Absorb tense/voice auxiliaries: "was signed", "were chosen",
            # "had arrived". Modality auxiliaries are also absorbed here —
            # the modal flag on Triplet records the lemma in parallel for
            # downstream consumers that want it factored out.
            indices.append(child.i)
        if child.dep_ == "neg":
            # Negation tokens are NOT included in the predicate span — they
            # turn into Triplet.negated. Skip them here.
            continue

    if explicit_prep is not None:
        for child in head.children:
            if child.dep_ == "prep" and child.lower_ == explicit_prep:
                indices.append(child.i)
                break
            # Passive 'by' agents come through dep_=agent on some spaCy
            # models; treat them the same.
            if child.dep_ == "agent" and explicit_prep == "by":
                indices.append(child.i)
                break

    start = min(indices)
    end = max(indices) + 1
    # Clamp forward-only expansion (particles, attached prep) to a window.
    # Backward absorption of auxiliaries is unbounded — there can be 2-3
    # tokens ("had been being seen") and we want all of them.
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
