"""Enoki Fact Extractor v2.

Rule-based Open Information Extraction over spaCy dependency parses.
Output is a flat list of (subject, predicate, argument) Triplets per sentence.

The rule catalogue lives under :mod:`fact_extractor.enoki_rules.rules`.
See PLAN.md and the 2026-05-23 refactor history (the original 16-rule
``rules`` package was deleted and ``rules_new`` was renamed in its
place).
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
