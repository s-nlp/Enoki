"""Offset-preserving markdown mask.

Replaces markdown syntax with spaces so the parser sees plain text while
character offsets still line up with the original document.
"""

from __future__ import annotations

import re


_HEADER_LINE = re.compile(r"(?m)^[ \t]*#{1,6}.*$")
_HEADER_PREFIX = re.compile(r"(?m)^[ \t]*#{1,6}[ \t]*")
_HORIZONTAL_RULE = re.compile(r"(?m)^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$")
_UNORDERED_LIST_MARKER = re.compile(r"(?m)^[ \t]*[-*+][ \t]+")
_ORDERED_LIST_MARKER = re.compile(r"(?m)^[ \t]*\d+[.)][ \t]+")
_EMPHASIS = re.compile(r"[*_~]+")


def _spaces(match: re.Match) -> str:
    return " " * (match.end() - match.start())


def mask_markdown(text: str) -> str:
    """Return ``text`` with markdown syntax replaced by spaces.

    >>> mask_markdown("# Heading\\nSome **bold** text.")
    '          \\nSome      bold     text.'

    Idempotent: ``mask_markdown(mask_markdown(t)) == mask_markdown(t)``.
    Character count is preserved.
    """
    out = _HEADER_LINE.sub(_spaces, text)
    out = _HEADER_PREFIX.sub(_spaces, out)
    out = _HORIZONTAL_RULE.sub(_spaces, out)
    out = _UNORDERED_LIST_MARKER.sub(_spaces, out)
    out = _ORDERED_LIST_MARKER.sub(_spaces, out)
    out = _EMPHASIS.sub(_spaces, out)
    assert len(out) == len(text), "markdown mask must preserve character count"
    return out
