"""Fact extraction backed by the rules engine (fact_extractor.enoki_rules.Pipeline)."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .models import Fact, IncrementalFactGroup


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


class EnokiRulesFactExtractor:
    """Fact extractor that wraps the rules-based Pipeline.

    The engine emits several ``Triplet`` objects per logical proposition at
    different span widths (``incremental_minimal_arg``,
    ``incremental_maximal_arg``, ``incremental_maximal_subject``,
    ``coord_*_granularity``, ...). Those variants are clustered by
    ``(subject, predicate, argument-head lemma)`` and packed into one
    ``IncrementalFactGroup`` each:

    - ``facts`` is sorted by argument-span length (narrow -> wide);
    - ``deltas[0]`` is the narrowest argument span, ``deltas[i]`` for i > 0 is
      the contiguous bracket of tokens NEW at step i.

    Sentence-level evaluation verifies only ``facts[-1]`` of each group, while
    span-level consumers keep the per-step deltas for localisation.
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

        Multi-granularity emissions of the same proposition are clustered
        into one group (see class docstring); every other triplet becomes a
        single-fact group.
        """
        if not text or not text.strip():
            return []

        triplets = list(self._pipeline.extract(text))

        # Group by (subject text, predicate surface, argument-head lemma).
        # The head lemma clusters the granularity variants of one proposition
        # ("implication" / "ethical implication" / "ethical implication of ...").
        clusters: Dict[Tuple[str, str, str], list] = defaultdict(list)
        for triplet in triplets:
            subject = getattr(triplet, "subject", None)
            argument = getattr(triplet, "argument", None)
            arg_span = getattr(argument, "span", None) if argument is not None else None
            arg_head = ""
            if arg_span is not None:
                root = getattr(arg_span, "root", None)
                arg_head = (getattr(root, "lemma_", "") or getattr(root, "text", "")).lower()
            key = (
                _norm(subject.text if subject is not None else ""),
                _norm(triplet.predicate_surface),
                arg_head,
            )
            clusters[key].append(triplet)

        groups: List[IncrementalFactGroup] = []
        for members in clusters.values():
            # Narrow -> wide by argument span length; argument-less triplets
            # have length 0 and sort first.
            members.sort(key=lambda t: len(t.argument.span) if (t.argument is not None and t.argument.span is not None) else 0)
            facts: List[Fact] = []
            deltas: list = []
            prev_token_indices: set = set()
            confidence: Optional[float] = None
            for triplet in members:
                fact = _fact_from_triplet(triplet)
                if fact is None:
                    continue
                facts.append(fact)
                if confidence is None:
                    confidence = getattr(triplet, "confidence", None)
                arg_span = fact.argument
                if arg_span is None:
                    # No argument tokens to mark: the delta is the predicate.
                    deltas.append(fact.predicate)
                    continue
                cur_indices = {tok.i for tok in arg_span}
                new_indices = cur_indices - prev_token_indices
                if not new_indices:
                    deltas.append(arg_span)
                else:
                    lo, hi = min(new_indices), max(new_indices) + 1
                    deltas.append(arg_span.doc[lo:hi])
                prev_token_indices = cur_indices
            if facts:
                groups.append(
                    IncrementalFactGroup(
                        facts=facts,
                        deltas=deltas,
                        confidence=confidence if confidence is not None else 0.0,
                    )
                )

        return groups


def _fact_from_triplet(triplet) -> Optional[Fact]:
    """Adapt one engine ``Triplet`` to a span-anchored ``Fact``."""
    from enoki.inference import _rule_triple
    from .anchored import fact_group

    triple = _rule_triple(triplet)
    triple["sentence_start"] = triplet.subject.sent.start_char
    triple["sentence_end"] = triplet.subject.sent.end_char
    group = fact_group(triplet.subject.doc, triple)
    if group is None:
        return None
    return group.facts[0]
