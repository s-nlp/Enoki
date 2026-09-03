"""Core data types for the rules fact extractor.

These types are part of the public contract and are protected from agent
edits (see PLAN.md §7, "Protected from the agent"). Changing them requires a
human change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional, Tuple

if TYPE_CHECKING:
    from spacy.tokens import Span, Token


ArgRole = Literal[
    "object",
    "time",
    "location",
    "manner",
    "cause",
    "instrument",
    "recipient",
    "purpose",
    "other",
]


@dataclass(frozen=True)
class Argument:
    """The argument slot of a Triplet, with its semantic role and surface prep."""

    span: Span
    role: ArgRole
    prep: Optional[str] = None


@dataclass(frozen=True)
class Triplet:
    """A flat SPO fact emitted by the pipeline.

    The predicate span includes verb head, attached particle, and any
    semantically-bound preposition (e.g. ``"born in"``). Negation and modality
    are factored out as boolean / string flags so consumers can render them
    however they like.

    ``argument`` is ``None`` only for true intransitives (e.g. ``"the sun rose"``).
    Multi-argument verbs are emitted as multiple Triplets sharing subject and
    predicate.

    ``predicate_text`` is an optional string that overrides ``predicate.text``
    when rendering. It exists for "no-verb" constructions like appositives
    (``"Marie Curie, a physicist"`` → ``"Marie Curie | is | physicist"``)
    where there is no copula token in the parse to anchor on. For ordinary
    verb-based triplets ``predicate_text`` is ``None`` and ``str(triplet)``
    uses ``predicate.text`` as usual.
    """

    subject: Span
    predicate: Span
    argument: Optional[Argument]
    negated: bool = False
    modality: Optional[str] = None
    source_rule: str = ""
    confidence: float = 1.0
    predicate_text: Optional[str] = None

    @property
    def predicate_surface(self) -> str:
        """The string used to render this triplet's predicate."""
        return self.predicate_text if self.predicate_text else self.predicate.text

    def __str__(self) -> str:
        s = self.subject.text
        p = self.predicate_surface
        if self.negated:
            p = f"NOT {p}"
        if self.argument is None:
            return f"{s} | {p}"
        if self.argument.prep:
            return f"{s} | {p} | {self.argument.span.text} (prep: {self.argument.prep}, role: {self.argument.role})"
        return f"{s} | {p} | {self.argument.span.text} (role: {self.argument.role})"


@dataclass(frozen=True)
class Candidate:
    """What rules emit. Heads only; the shape stage materializes spans.

    A rule's job is to identify the *head tokens* of subject/predicate/argument
    based on the dependency parse, plus the semantic role. Span boundaries
    (including compound modifiers, attached PPs, det stripping, NER expansion,
    etc.) are computed deterministically downstream by :mod:`fact_extractor.enoki_rules.shape`.

    ``synthesized_predicate_text`` lets a rule override the predicate surface
    when there is no copula or verb token in the parse to anchor on (e.g.
    appositive identification yields synthesized predicate ``"is"``). When
    set, the shape stage emits a degenerate one-token predicate span and the
    Triplet carries the synthesized string as ``predicate_text``.
    """

    subject_head: Token
    predicate_head: Token
    arg_head: Optional[Token]
    role: ArgRole
    prep: Optional[str] = None
    negated: bool = False
    modality: Optional[str] = None
    source_rule: str = ""
    confidence: float = 1.0
    synthesized_predicate_text: Optional[str] = None
    arg_span_subtree: bool = False
    """When True, the shape stage emits the full (contiguous, trimmed)
    subtree of ``arg_head`` as the argument span instead of the noun-style
    head expansion. Set for clausal arguments whose head is a verb/aux
    (advcl, ccomp, relcl) — the natural span is the embedded clause."""
    arg_minimal_only: bool = False
    """When True, the shape stage emits ONLY the ``arg_head`` token (one-
    token span, no modifier expansion). Used for multi-granularity gold
    convention (EnokiQA-style) where the same (s, p) is credited at
    multiple object-span widths; the minimal-head emission complements
    the standard noun-style expansion via the dedup key."""
    subj_minimal_only: bool = False
    """Subject-side counterpart of ``arg_minimal_only`` — emit only the
    ``subject_head`` token as the subject span."""
    subj_span_subtree: bool = False
    """Subject-side counterpart of ``arg_span_subtree`` — emit the full
    ``subject_head.subtree`` (trimmed at comma/punct) as the subject
    span. Useful for absorbing appositive named entities ("Dayton,
    Montana")."""


@dataclass(frozen=True)
class Clause:
    """A segmented clause carrying its own subject candidates.

    Rules operate on Clauses, not raw Sents — this keeps subject inheritance
    explicit (relative clauses, reduced clauses, conjoined verbs) instead of
    hidden inside each rule.
    """

    span: Span
    root: Token
    subject_candidates: Tuple[Token, ...] = field(default_factory=tuple)
