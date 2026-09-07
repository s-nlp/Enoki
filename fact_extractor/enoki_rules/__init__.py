"""Rule-based Open Information Extraction over spaCy dependency parses.

The pipeline emits a flat list of (subject, predicate, argument) triplets per
sentence. The rule catalogue lives under :mod:`fact_extractor.enoki_rules.rules`.
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
