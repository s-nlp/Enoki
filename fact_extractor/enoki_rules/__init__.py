"""Enoki rules fact extractor.

Rule-based Open Information Extraction over spaCy dependency parses.
Output is a flat list of (subject, predicate, argument) Triplets per sentence.

The rule catalogue lives under :mod:`fact_extractor.enoki_rules.rules`.
"""

from .models import Argument, Candidate, Clause, Triplet
from .config import ExtractionConfig

__all__ = [
    "Argument",
    "Candidate",
    "Clause",
    "ExtractionConfig",
    "Triplet",
]
