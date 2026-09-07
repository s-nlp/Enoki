"""Subject-span shaping.

Expands a subject head to its compound, adjectival and possessive modifiers,
or to the full named entity containing it, and optionally strips a leading
determiner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List

from ..config import ShapeConfig
from .boundaries import expand_to_entity, strip_leading_dets

if TYPE_CHECKING:
    from spacy.tokens import Span, Token


_INCLUDED_DEPS_BASE = {"poss", "det"}

# Quantifier heads whose span covers the whole "of"-PP ("one of the men").
_PARTITIVE_HEADS = {
    "some", "many", "one", "none", "all", "most", "each", "several",
    "few", "half", "both", "any", "much", "lot", "lots", "rest",
    "majority", "minority", "number", "part", "member", "members",
    "two", "three", "four", "five", "ten", "dozens", "hundreds",
    "thousands", "millions", "plenty",
}


def shape_subject(head: "Token", cfg: ShapeConfig) -> "Span":
    """Return the materialized subject span for ``head`` under ``cfg``."""
    if cfg.expand_to_full_entity:
        ent = expand_to_entity(head)
        if ent is not None:
            return _maybe_strip_det(ent, cfg)

    keep = set(_INCLUDED_DEPS_BASE)
    if cfg.include_compound:
        keep.add("compound")
    if cfg.include_amod:
        keep.add("amod")
        keep.add("nummod")

    indices = _collect_indices(head, keep)
    if getattr(cfg, "include_partitive_of", False):
        indices.extend(_partitive_of_indices(head))
    start = min(indices)
    end = max(indices) + 1
    span = head.doc[start:end]
    return _maybe_strip_det(span, cfg)


def _collect_indices(head: "Token", keep_deps: Iterable[str]) -> List[int]:
    indices = [head.i]
    keep = set(keep_deps)
    for child in head.children:
        if child.dep_ in keep:
            indices.append(child.i)
    return indices


def _partitive_of_indices(head: "Token") -> List[int]:
    """Indices of a partitive ``of``-PP subtree hanging off a quantifier ``head``."""
    if head.lower_ not in _PARTITIVE_HEADS and head.pos_ != "NUM":
        return []
    for child in head.children:
        if child.dep_ == "prep" and child.lower_ == "of":
            return [t.i for t in child.subtree]
    return []


def _maybe_strip_det(span: "Span", cfg: ShapeConfig) -> "Span":
    if cfg.strip_leading_det:
        return strip_leading_dets(span)
    return span
