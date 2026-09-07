"""Split a parsed document into :class:`Clause` units.

A clause is rooted at a clausal head: the ``ROOT`` token, a verb or auxiliary
attached as ``conj``/``ccomp``/``advcl``/``relcl``/``acl``/``xcomp``, or a
copular predicate with a ``cop`` child. The clause span is the head's subtree
and its subjects are the head's ``nsubj``/``nsubjpass``/``csubj`` children,
with coordinated subjects expanded one level.
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
    has_cop_child = any(c.dep_ == "cop" for c in token.children)
    if has_cop_child:
        return True
    return False


def _gather_subjects(head: "Token") -> Tuple["Token", ...]:
    subs: List["Token"] = []
    for child in head.children:
        if child.dep_ in _SUBJECT_DEPS:
            subs.append(child)
            for grand in child.children:
                if grand.dep_ == "conj":
                    subs.append(grand)
    return tuple(subs)


def _clause_span(head: "Token") -> "Span":
    # token.subtree is unordered; bracket it by index.
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
