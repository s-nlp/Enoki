"""Deterministic span-shaping policies.

These take a :class:`Candidate` (heads only) and turn it into a concrete
``(subject_span, predicate_span, argument_span)`` triple ready for filtering
and emission. The shaping policies are calibration targets (numeric thresholds
in :class:`fact_extractor.enoki_rules.config.ShapeConfig`) but are not authored
per-construction.
"""

from .boundaries import expand_to_entity, trim_trailing_punct
from .object import shape_argument
from .predicate import shape_predicate
from .subject import shape_subject

__all__ = [
    "shape_subject",
    "shape_predicate",
    "shape_argument",
    "expand_to_entity",
    "trim_trailing_punct",
]
