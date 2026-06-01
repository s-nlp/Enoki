"""Evaluation harness: gold conversion, metrics, runner, ablation, regression."""

from .metrics import EvalReport, score_against_gold
from .qasrl_to_spo import GoldTriplet, QASRLConversionResult, convert_sentence

__all__ = [
    "EvalReport",
    "GoldTriplet",
    "QASRLConversionResult",
    "convert_sentence",
    "score_against_gold",
]
