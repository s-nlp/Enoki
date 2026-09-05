"""Fact extraction backed by the rules engine (fact_extractor.enoki_rules.Pipeline)."""

from __future__ import annotations

from typing import List, Optional

from .models import IncrementalFactGroup


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
            from enoki.inference import _rule_triple
            from .anchored import fact_group
            triple = _rule_triple(triplet)
            triple["sentence_start"] = triplet.subject.sent.start_char
            triple["sentence_end"] = triplet.subject.sent.end_char
            group = fact_group(triplet.subject.doc, triple)
            if group is not None:
                groups.append(group)

        return groups
