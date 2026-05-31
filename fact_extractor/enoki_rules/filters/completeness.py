"""Completeness filter — reject triplets missing required slots."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import FilterConfig

if TYPE_CHECKING:
    from ..models import Triplet


_COPULA_AUX_LEMMAS = {"be", "have"}


def is_complete(triplet: "Triplet", cfg: FilterConfig) -> bool:
    """Return True if the triplet has the required slots filled."""
    if cfg.require_subject and (triplet.subject is None or len(triplet.subject) == 0):
        return False
    if cfg.require_predicate and (
        triplet.predicate is None or len(triplet.predicate) == 0
    ):
        return False

    # An empty argument is fine for true intransitives. But a bare copula
    # like "X is" with no complement is nonsense — reject it.
    if triplet.argument is None and len(triplet.predicate) > 0:
        root = triplet.predicate.root
        if root.lemma_.lower() in _COPULA_AUX_LEMMAS:
            return False
    return True
