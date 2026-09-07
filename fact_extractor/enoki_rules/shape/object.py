"""Argument-span shaping.

Like :mod:`.subject`, but also includes numeric modifiers, absorbs a partitive
``of``-PP, and trims at break punctuation so trailing clauses do not bleed in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from ..config import ShapeConfig
from .boundaries import expand_to_entity, strip_leading_dets, trim_break_punct, trim_trailing_punct

if TYPE_CHECKING:
    from spacy.tokens import Span, Token


_INCLUDED_DEPS_BASE = {"poss", "det", "nummod"}

# Quantifier heads whose span covers the whole "of"-PP ("one of the men").
_PARTITIVE_HEADS = {
    "some", "many", "one", "none", "all", "most", "each", "several",
    "few", "half", "both", "any", "much", "lot", "lots", "rest",
    "majority", "minority", "number", "part", "member", "members",
    "two", "three", "four", "five", "ten", "dozens", "hundreds",
    "thousands", "millions", "plenty",
}


def _partitive_of_indices(head: "Token") -> List[int]:
    """Indices of a partitive ``of``-PP subtree hanging off ``head``."""
    if head.lower_ not in _PARTITIVE_HEADS and head.pos_ != "NUM":
        return []
    for child in head.children:
        if child.dep_ == "prep" and child.lower_ == "of":
            return [t.i for t in child.subtree]
    return []


def shape_argument(head: "Token", cfg: ShapeConfig) -> "Span":
    """Return the materialized argument span for ``head``."""
    if cfg.expand_to_full_entity:
        ent = expand_to_entity(head)
        if ent is not None:
            span = _maybe_strip_det(ent, cfg)
            return trim_trailing_punct(span)

    keep = set(_INCLUDED_DEPS_BASE)
    if cfg.include_compound:
        keep.add("compound")
    if cfg.include_amod:
        keep.add("amod")

    indices: List[int] = [head.i]
    for child in head.children:
        if child.dep_ in keep:
            indices.append(child.i)
            for grand in child.children:
                if grand.dep_ == "compound":
                    indices.append(grand.i)

    if getattr(cfg, "include_partitive_of", False):
        indices.extend(_partitive_of_indices(head))

    start = min(indices)
    end = max(indices) + 1
    span = head.doc[start:end]
    span = trim_break_punct(span)
    span = trim_trailing_punct(span)
    return _maybe_strip_det(span, cfg)


def _maybe_strip_det(span: "Span", cfg: ShapeConfig) -> "Span":
    if cfg.strip_leading_det:
        return strip_leading_dets(span)
    return span
