"""Clause segmentation and subject-coordination resolution."""

from .clauses import segment_into_clauses
from .coordination import distribute_subjects

__all__ = ["segment_into_clauses", "distribute_subjects"]
