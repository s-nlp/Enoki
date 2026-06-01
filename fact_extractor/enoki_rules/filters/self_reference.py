"""Self-reference filter.

Reject triplets where the subject and the argument refer to the same entity,
unless the argument is a possessive construction (``"John is known for John's
looks"`` is valid).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import FilterConfig

if TYPE_CHECKING:
    from ..models import Triplet


def is_self_reference(triplet: "Triplet", cfg: FilterConfig) -> bool:
    """Return True if the triplet should be dropped as self-referential."""
    if not cfg.drop_self_reference:
        return False
    if triplet.argument is None:
        return False

    subj = triplet.subject
    arg = triplet.argument.span

    if subj.start == arg.start and subj.end == arg.end:
        return True

    has_possessive = any(t.dep_ == "poss" for t in arg) or "'s" in arg.text.lower()
    if has_possessive:
        return False

    subj_range = range(subj.start, subj.end)
    arg_range = range(arg.start, arg.end)
    overlap = set(subj_range) & set(arg_range)
    if overlap:
        ratio = len(overlap) / max(1, min(len(subj_range), len(arg_range)))
        if ratio > cfg.self_reference_overlap:
            return True

    subj_text = subj.text.lower().strip()
    arg_text = arg.text.lower().strip()
    if subj_text == arg_text:
        return True
    return False
