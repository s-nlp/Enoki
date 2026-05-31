"""Subject-span shaping.

Given a subject head token, materialize a span that includes:

- compound modifiers (``"New York City"`` from head ``City``);
- adjectival modifiers (``"red car"`` from head ``car``);
- possessive determiners (``"his book"``);
- if the head sits inside a named entity, the full entity span;

… and drops leading determiners (``"the"``, ``"a"``, ``"an"``) when configured
to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List

from ..config import ShapeConfig
from .boundaries import expand_to_entity, strip_leading_dets

if TYPE_CHECKING:
    from spacy.tokens import Span, Token


_INCLUDED_DEPS_BASE = {"poss", "det"}

# Heads that take a partitive "of"-PP whose gold subject span is the
# whole phrase ("Some of the methods", "One of the men"). Restricting
# to this curated set + numeric heads keeps precision: a generic NOUN
# head ("the report of the committee") is *not* partitive and its gold
# subject is usually just the head noun.
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
            # One level deep is enough for subject heads; we don't want to
            # absorb relative clauses or PP modifiers here.
            indices.append(child.i)
    return indices


def _partitive_of_indices(head: "Token") -> List[int]:
    """Indices of a partitive ``of``-PP subtree hanging off ``head``.

    Only fires for quantifier/number heads (``_PARTITIVE_HEADS`` or a
    head whose ``pos_`` is ``NUM``); the whole ``of``-PP is contiguous
    with the head so absorbing its subtree yields a clean span.
    """
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
