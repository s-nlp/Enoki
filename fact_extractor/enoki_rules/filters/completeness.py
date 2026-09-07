"""Reject triplets that are missing a required slot."""

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

    # Intransitives may lack an argument, but a bare copula ("X is") may not.
    if triplet.argument is None and len(triplet.predicate) > 0:
        root = triplet.predicate.root
        if root.lemma_.lower() in _COPULA_AUX_LEMMAS:
            return False
    return True
