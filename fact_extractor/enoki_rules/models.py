"""Core data types shared by rules, shaping, filters and the pipeline."""

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
    """A flat (subject, predicate, argument) fact emitted by the pipeline.

    The predicate span covers the verb head, its auxiliaries and particle, and
    any preposition bound into the predicate (``"born in"``). Negation and
    modality are carried as flags. ``argument`` is ``None`` only for
    intransitives; a verb with several arguments yields several triplets.

    ``predicate_text`` overrides ``predicate.text`` for constructions with no
    verb token to anchor on, such as appositives rendered with a synthesized
    ``"is"``.
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
    """What a rule emits: head tokens plus role; the shape stage builds spans.

    ``synthesized_predicate_text`` overrides the predicate surface when there
    is no verb token to anchor on (appositives yield ``"is"``); the shape stage
    then emits a one-token predicate span and the triplet carries the string as
    ``predicate_text``.
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
    """Emit the full trimmed subtree of ``arg_head`` as the argument span.
    Used for clausal arguments (advcl, ccomp, relcl)."""
    arg_minimal_only: bool = False
    """Emit only the ``arg_head`` token as the argument span, with no
    modifier expansion. Used for multi-granularity emissions."""
    subj_minimal_only: bool = False
    """Emit only the ``subject_head`` token as the subject span."""
    subj_span_subtree: bool = False
    """Emit the full trimmed subtree of ``subject_head`` as the subject span."""


@dataclass(frozen=True)
class Clause:
    """A segmented clause with the subject tokens available to its rules.

    Subject inheritance for relative, reduced and conjoined clauses is resolved
    during segmentation, so rules never have to walk the parse for a subject.
    """

    span: Span
    root: Token
    subject_candidates: Tuple[Token, ...] = field(default_factory=tuple)
