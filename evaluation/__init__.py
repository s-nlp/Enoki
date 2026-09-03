"""Evaluation API for Enoki hallucination detection.

Exports are resolved lazily so lightweight metric helpers do not import model
and dataset dependencies.
"""

from importlib import import_module

__version__ = "0.1.0"

_LAZY_EXPORTS = {
    "run_sentence_evaluation": (".sentence", "run_sentence_evaluation"),
    "run_entity_evaluation": (".entity", "run_entity_evaluation"),
    "run_span_evaluation": (".span", "run_span_evaluation"),
    "calculate_entity_metrics": (".metrics", "calculate_entity_metrics"),
    "calculate_sentence_metrics": (".metrics", "calculate_sentence_metrics"),
    "calculate_span_f1": (".metrics", "calculate_span_f1"),
    "print_entity_metrics_summary": (".metrics", "print_entity_metrics_summary"),
    "print_sentence_metrics_summary": (".metrics", "print_sentence_metrics_summary"),
    "print_span_metrics_summary": (".metrics", "print_span_metrics_summary"),
    "span_coverage_micro": (".span_metrics", "span_coverage_micro"),
    "span_coverage_macro": (".span_metrics", "span_coverage_macro"),
    "span_iou_one": (".span_metrics", "span_iou_one"),
    "span_iou_macro": (".span_metrics", "span_iou_macro"),
    "SpanCoveragePRF": (".span_metrics", "SpanCoveragePRF"),
}


def __getattr__(name):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = list(_LAZY_EXPORTS)
