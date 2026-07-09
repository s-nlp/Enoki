"""Quoted-entity placeholder substitution.

Treats quoted strings (``"Attention Is All You Need"``, ``«Война и мир»``,
etc.) as opaque entities. They get replaced by placeholders before parsing
so the dependency parser doesn't try to dissect the content; the original
text is restored after extraction.

NOTE: this does NOT preserve character offsets (placeholder lengths differ
from original quoted strings). Use it only when an extraction rule wants to
treat a title as a single token — the parse-time pipeline currently does
NOT call this; the old extractor used it sentence-locally for specific
constructions. Kept here as a deliberate, opt-in helper for the rules that
need it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict


_QUOTE_PATTERNS = [
    r'"([^"]+)"',           # straight double
    r"'([^']+)'",           # straight single
    r"«([^»]+)»",           # guillemets
    r"“([^”]+)”",  # smart double
    r"‘([^’]+)’",  # smart single
]


@dataclass(frozen=True)
class QuoteMapping:
    """The result of masking quoted entities.

    ``processed_text`` may differ in length from the original; downstream code
    operating on offsets MUST account for this.
    """

    processed_text: str
    mapping: Dict[str, str]


def mask_quoted_entities(text: str) -> QuoteMapping:
    """Replace quoted spans with ``__ENTITY_<n>__`` placeholders."""
    entities: Dict[str, str] = {}
    counter = [0]

    def _replace(match: re.Match) -> str:
        entity = match.group(1)
        placeholder = f"__ENTITY_{counter[0]}__"
        entities[placeholder] = entity
        counter[0] += 1
        return placeholder

    processed = text
    for pattern in _QUOTE_PATTERNS:
        processed = re.sub(pattern, _replace, processed)

    return QuoteMapping(processed_text=processed, mapping=entities)


def restore_quoted_entities(text: str, mapping: Dict[str, str]) -> str:
    """Inverse of :func:`mask_quoted_entities`."""
    result = text
    for placeholder, entity in mapping.items():
        result = result.replace(placeholder, entity)
    return result
