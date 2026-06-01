"""Split a parsed sentence into independent :class:`Clause` units.

A clause is centered on a *clausal head* — a verb (or copular construction)
that carries its own subject. The function discovers clausal heads via
dependency relations and emits one ``Clause`` per head, with that head's
subtree as the clause span.

Heuristics, deliberately conservative:

- ``token.dep_ == "ROOT"`` always produces a clause.
- ``token.dep_ in {"conj", "ccomp", "advcl", "relcl", "acl", "xcomp"}`` and
  ``token.pos_ in {"VERB", "AUX"}`` produces a clause.
- Copular clauses where the predicate is a noun/adjective and the copula is
  attached via ``cop`` produce a clause rooted at the predicate token.
- Subjects are taken from ``nsubj`` / ``nsubjpass`` / ``csubj`` children;
  coordinated subjects are expanded one level deep.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

from ..models import Clause

if TYPE_CHECKING:
    from spacy.tokens import Doc, Span, Token


_CLAUSAL_DEPS = {"ROOT", "conj", "ccomp", "advcl", "relcl", "acl", "xcomp"}
_SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass"}


def _is_clausal_head(token: "Token") -> bool:
    if token.dep_ == "ROOT":
        return True
    if token.dep_ in _CLAUSAL_DEPS and token.pos_ in {"VERB", "AUX"}:
        return True
    # Copular predicate noun/adj: "He is happy" -> happy is the head, 'is' has dep_=cop
    has_cop_child = any(c.dep_ == "cop" for c in token.children)
    if has_cop_child:
        return True
    return False


def _gather_subjects(head: "Token") -> Tuple["Token", ...]:
    subs: List["Token"] = []
    for child in head.children:
        if child.dep_ in _SUBJECT_DEPS:
            subs.append(child)
            # Expand "X and Y" via conj
            for grand in child.children:
                if grand.dep_ == "conj":
                    subs.append(grand)
    return tuple(subs)


def _clause_span(head: "Token") -> "Span":
    # The subtree of the clausal head, trimmed to a contiguous span. spaCy's
    # token.subtree yields tokens in arbitrary order so we sort and bracket.
    tokens = sorted(head.subtree, key=lambda t: t.i)
    start = tokens[0].i
    end = tokens[-1].i + 1
    return head.doc[start:end]


def segment_into_clauses(doc: "Doc") -> List[Clause]:
    """Return one :class:`Clause` per clausal head in ``doc``."""
    clauses: List[Clause] = []
    for token in doc:
        if _is_clausal_head(token):
            clauses.append(
                Clause(
                    span=_clause_span(token),
                    root=token,
                    subject_candidates=_gather_subjects(token),
                )
            )
    return clauses
