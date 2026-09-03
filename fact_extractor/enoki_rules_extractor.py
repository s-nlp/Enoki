"""Fact extraction backed by the rules engine (fact_extractor.enoki_rules.Pipeline)."""

from __future__ import annotations

from typing import List, Optional

from .models import Fact, IncrementalFactGroup


class EnokiRulesFactExtractor:
    """Fact extractor that wraps the rules-based Pipeline.

    Each ``Triplet`` emitted by the engine becomes a single-fact
    ``IncrementalFactGroup`` (delta = argument span), matching the interface
    expected by decontextualizer → fact_alignment → NLI.
    """

    def __str__(self) -> str:
        return "EnokiRules"

    def __init__(
        self,
        nlp=None,
        incremental: bool = True,
        config=None,
    ) -> None:
        """
        Args:
            nlp: Pre-loaded spaCy model. When provided, it is injected into
                the engine's Parser so both share the same loaded instance.
            incremental: Accepted for interface compatibility; not used.
            config: ``ExtractionConfig`` for the engine Pipeline. Defaults to
                ``ExtractionConfig()`` (all rules, en_core_web_trf, no GLiNER).
        """
        from .enoki_rules.pipeline import Pipeline

        self._pipeline = Pipeline(config=config)
        if nlp is not None:
            self._pipeline.parser._nlp = nlp

    def extract_granular_facts(self, text: str) -> List[IncrementalFactGroup]:
        """Extract facts and return them as ``IncrementalFactGroup`` objects.

        Each engine ``Triplet`` maps to one single-fact group. The delta is
        the argument span when present, otherwise an empty list.
        """
        if not text or not text.strip():
            return []

        triplets = self._pipeline.extract(text)

        groups: List[IncrementalFactGroup] = []
        for triplet in triplets:
            arg_span = triplet.argument.span if triplet.argument is not None else None
            prep = triplet.argument.prep if triplet.argument is not None else None

            fact = Fact(
                subject=triplet.subject,
                predicate=triplet.predicate,
                argument=arg_span,
                prep=prep,
                predicate_text=triplet.predicate_text,
            )
            group = IncrementalFactGroup(
                facts=[fact],
                deltas=[arg_span] if arg_span is not None else [],
                confidence=triplet.confidence,
            )
            groups.append(group)

        return groups
