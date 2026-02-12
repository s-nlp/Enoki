"""
Evaluation module for Enoki hallucination detection.

Provides unified evaluation functions for:
- Sentence-level evaluation (FELM, FactCheckBench)
- Entity-level evaluation (HalluEntity)
- Span-level evaluation (PsiloQA, Mushroom, RAGTruth)
"""

__version__ = "0.1.0"

from evaluation.sentence import run_sentence_evaluation
from evaluation.entity import run_entity_evaluation
from evaluation.span import run_span_evaluation
from evaluation.metrics import (
    calculate_entity_metrics,
    calculate_sentence_metrics,
    calculate_span_f1,
    print_entity_metrics_summary,
    print_sentence_metrics_summary,
    print_span_metrics_summary,
)
from evaluation.span_metrics import (
    span_coverage_micro,
    span_coverage_macro,
    SpanCoveragePRF,
)

__all__ = [
    # Evaluation runners
    'run_sentence_evaluation',
    'run_entity_evaluation',
    'run_span_evaluation',
    # Metric calculators
    'calculate_entity_metrics',
    'calculate_sentence_metrics',
    'calculate_span_f1',
    'span_coverage_micro',
    'span_coverage_macro',
    'SpanCoveragePRF',
    # Metric printers
    'print_entity_metrics_summary',
    'print_sentence_metrics_summary',
    'print_span_metrics_summary',
]
