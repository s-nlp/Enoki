"""Drop degenerate arguments: bare percentages, section headers, context-free years."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..config import FilterConfig

if TYPE_CHECKING:
    from ..models import Triplet


_SECTION_HEADERS = frozenset({
    "early career", "history", "geography", "biography", "personal life",
    "career", "education", "background", "overview", "summary",
    "moves", "transfers", "achievements", "awards", "legacy",
    "early life", "later life", "death", "works", "publications",
})
_BARE_PCT = re.compile(r"^\d+(?:\.\d+)?%$")
_BARE_YEAR = re.compile(r"^\d{4}$")


def is_meaningful_argument(triplet: "Triplet", cfg: FilterConfig) -> bool:
    """Return True if the argument is *not* a degenerate fragment."""
    if not cfg.drop_fragment_args:
        return True
    if triplet.argument is None:
        return True

    arg_text = triplet.argument.span.text.strip().lower()
    if not arg_text:
        return False

    if cfg.drop_section_headers and arg_text in _SECTION_HEADERS:
        return False

    if _BARE_PCT.match(arg_text):
        return False

    if cfg.drop_year_only_args and _BARE_YEAR.match(arg_text):
        # A bare year is kept when a temporal preposition or an establishment
        # verb gives it context. The prep may sit on the Argument rather than
        # in the predicate surface.
        pred = triplet.predicate_surface.lower()
        if triplet.argument and triplet.argument.prep:
            pred = f"{pred} {triplet.argument.prep.lower()}".strip()
        ends_in_temporal_prep = any(
            pred.endswith(f" {prep}") or pred == prep
            for prep in ("in", "on", "at", "during", "since", "until", "before", "after")
        )
        has_establishment_verb = any(
            kw in pred for kw in (
                "born", "died", "founded", "established", "built", "opened",
                "begun", "started", "created", "formed", "married", "released",
            )
        )
        if not (ends_in_temporal_prep or has_establishment_verb):
            return False

    return True
