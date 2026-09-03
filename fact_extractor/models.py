"""Data models for fact extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from spacy.tokens import Doc, Span, Token


TokOrSpan = Union[Token, Span]


@dataclass
class QuotedEntityMapping:
    """Mapping between placeholders and original quoted entities"""
    processed_text: str
    mapping: Dict[str, str]


@dataclass
class ContrastiveParse:
    """Parsed contrastive construction (X on A, B but not on C, D)"""
    base_phrase: str
    preposition: str
    positive_items: List[str]
    negative_items: List[str]


@dataclass
class ListParse:
    """Parsed list/enumeration"""
    context: str
    items: List[str]


@dataclass(frozen=True)
class Fact:
    """
    Represents a single fact extracted from text.

    A fact consists of:
    - subject: The entity the fact is about
    - predicate: The action/relation (verb or copula)
    - argument: The object/complement (optional)
    - prep: Preposition stored separately from argument (optional)
    - predicate_text: Surface override for predicates absent from source text
      (optional)

    DESIGN GOAL: GRANULAR HALLUCINATION DETECTION WITHOUT LLM

    The fact extraction system is designed for precise hallucination detection:

    1. Facts are checked sequentially via NLI against source context
    2. When a fact contradicts the context, the ARGUMENT is marked as hallucination
    3. To enable GRANULAR detection, each modifier must have its own incremental fact

    Example:
        Generated: "C# is a modern multi-paradigm programming language"
        Context: "C# is an old programming language"

        Without incremental facts:
            Fact: C# | is | modern multi-paradigm programming language
            NLI: CONTRADICTION → marks entire "modern multi-paradigm programming language"
            Problem: Can't pinpoint that only "modern" is wrong

        With incremental facts:
            Fact 1: C# | is | language (delta: "language")
            Fact 2: C# | is | programming language (delta: "programming")
            Fact 3: C# | is | multi-paradigm programming language (delta: "multi-paradigm")
            Fact 4: C# | is | modern multi-paradigm programming language (delta: "modern")

            NLI checks sequentially:
            - Facts 1-3: ENTAILMENT ✓
            - Fact 4: CONTRADICTION ✗ → mark delta "modern" as hallucination

            Result: Precise detection - only "modern" is marked, not entire phrase

    This is why incremental facts are crucial: they enable precise span-level
    hallucination detection by testing each added modifier independently.
    """
    subject: Span
    predicate: Span
    argument: Optional[Span] = None
    prep: Optional[str] = None
    # Some rule-based extractions have no predicate span in the source text
    # (e.g. appositive identity: "Marie Curie, a physicist" -> "is").
    # Keep the original span for offset-aware consumers, while allowing those
    # extractors to supply the surface form used to render the fact.
    predicate_text: Optional[str] = None

    def __str__(self) -> str:
        def _pretty(span: Span) -> str:
            doc = span.doc
            if span.start > 0 and span.end < len(doc):
                prev_tok = doc[span.start - 1]
                next_tok = doc[span.end]
                if prev_tok.text in {'"', "'", "``", "\u201c", "\u2018"} and next_tok.text in {'"', "'", "''", "\u201d", "\u2019"}:
                    return f"{prev_tok.text}{span.text}{next_tok.text}"
            return span.text

        sub = _pretty(self.subject)
        pred = self.predicate_text or _pretty(self.predicate)

        if self.argument is None:
            return f"{sub} {pred}"

        arg = _pretty(self.argument)
        if self.prep:
            return f"{sub} {pred} {arg} {self.prep}"
        return f"{sub} {pred} {arg}"


@dataclass
class IncrementalFactGroup:
    """
    Group of incremental facts with shared subject and predicate.

    ``clause_type`` carries the extractor-specific clause classification
    (e.g. claucy's SVO/SVC/SVA/…) and is stored for post-analysis only —
    the NLI pipeline does not use it.

    Each fact adds one more modifier/chunk to the argument, enabling
    GRANULAR HALLUCINATION DETECTION through sequential NLI checking.

    Structure:
    - facts: List of facts ordered from simplest to most complex
    - deltas: List of spans representing ONLY the new chunk added at each step

    CRITICAL: Deltas contain ONLY the new addition, not the accumulated text.
    This ensures that when NLI detects a contradiction, we mark ONLY the
    problematic modifier/chunk, not the entire accumulated phrase.

    Example 1 - Nested prepositions:
        Text: "Duckenfield is a suburb of Reading in the United Kingdom"

        Facts:
          1. Duckenfield | is | a suburb
          2. Duckenfield | is | a suburb of Reading
          3. Duckenfield | is | a suburb of Reading in the United Kingdom

        Deltas:
          1. "a suburb" (full argument)
          2. "of Reading" (only the prepositional phrase added)
          3. "in the United Kingdom" (only the new prepositional phrase)

        If NLI detects contradiction at Fact 3, we mark ONLY delta[2]
        ("in the United Kingdom") as the hallucination, not the entire argument.

    Example 2 - NP modifiers:
        Text: "C# is a modern multi-paradigm programming language"

        Facts:
          1. C# | is | language
          2. C# | is | programming language
          3. C# | is | multi-paradigm programming language
          4. C# | is | modern multi-paradigm programming language

        Deltas:
          1. "language" (base)
          2. "programming" (only the added modifier)
          3. "multi-paradigm" (only the added modifier)
          4. "modern" (only the added modifier)

        If NLI detects contradiction at Fact 4, we mark ONLY delta[3]
        ("modern") as the hallucination.

    Example 3 - Conjuncts/lists:
        Text: "Symptoms include fever, cough, and toxins"
        Context: "Symptoms include fever and cough" (no mention of toxins)

        Facts:
          1. [subject] | include | fever
          2. [subject] | include | cough
          3. [subject] | include | toxins

        Deltas:
          1. "fever"
          2. "cough"
          3. "toxins"

        NLI checks:
        - Fact 1: ENTAILMENT ✓
        - Fact 2: ENTAILMENT ✓
        - Fact 3: CONTRADICTION ✗ → mark delta[2] ("toxins")

        Result: Mark "toxins" as hallucination - the other items are correct.

    This design enables LLM-free hallucination detection with span-level precision,
    identifying exactly which modifier, preposition, or list item is incorrect.
    """
    facts: List[Fact]
    deltas: List[Span]
    clause_type: Optional[str] = None
    confidence: float = 0.0
    # Per-fact hal_probs for incremental mode (hal head only).
    # When set, fact i uses per_fact_confidences[i] instead of group.confidence.
    per_fact_confidences: Optional[List[float]] = None

    def __str__(self) -> str:
        if len(self.facts) == 1:
            return str(self.facts[0])
        return f"IncrementalGroup({len(self.facts)} facts)"

    def __iter__(self):
        return iter(self.facts)

    def __len__(self):
        return len(self.facts)

    def __getitem__(self, idx):
        return self.facts[idx]
