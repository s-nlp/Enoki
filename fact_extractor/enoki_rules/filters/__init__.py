"""Triplet validation filters.

Each filter exposes a single boolean toggle in :class:`FilterConfig` and a
single pure function that takes a candidate triplet and returns whether it
should be kept.
"""

from .completeness import is_complete
from .dedup import dedup_triplets
from .fragments import is_meaningful_argument
from .self_reference import is_self_reference

__all__ = [
    "is_complete",
    "dedup_triplets",
    "is_meaningful_argument",
    "is_self_reference",
]
