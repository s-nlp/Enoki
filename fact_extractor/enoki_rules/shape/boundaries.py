"""Shared span-boundary utilities used by subject/predicate/object shapers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from spacy.tokens import Span, Token


_BREAK_PUNCT = {",", ";", ":"}
_LEADING_DETS = {"the", "a", "an"}


def trim_trailing_punct(span: "Span") -> "Span":
    """Drop trailing punctuation tokens."""
    end = span.end
    while end > span.start and span.doc[end - 1].is_punct:
        end -= 1
    if end == span.start:
        return span
    return span.doc[span.start:end]


def trim_break_punct(span: "Span") -> "Span":
    """Trim the span at the first comma/semicolon/colon (exclusive).

    Used by argument shaping to drop trailing clauses like
    ``", which is in California"``.
    """
    for i in range(span.start, span.end):
        tok = span.doc[i]
        if tok.is_punct and tok.text in _BREAK_PUNCT:
            return span.doc[span.start:i]
    return span


def strip_leading_dets(span: "Span") -> "Span":
    """Drop ``the``/``a``/``an`` from the start of a span."""
    start = span.start
    while start < span.end and span.doc[start].lower_ in _LEADING_DETS:
        start += 1
    if start >= span.end:
        return span
    return span.doc[start:span.end]


def expand_to_entity(token: "Token") -> Optional["Span"]:
    """If ``token`` is part of a recognized entity, return that entity's span."""
    if token.ent_iob_ in {"B", "I"} and token.ent_type_:
        for ent in token.doc.ents:
            if ent.start <= token.i < ent.end:
                return ent
    return None
