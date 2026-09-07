"""Deterministic span shaping: turn candidate head tokens into subject, predicate
and argument spans, governed by :class:`~fact_extractor.enoki_rules.config.ShapeConfig`.
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
