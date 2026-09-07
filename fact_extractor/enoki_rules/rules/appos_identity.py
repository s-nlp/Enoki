"""Appositive identity: "X, a Y, ..." -> (X, is, Y).

Fires only when head and appositive are both NOUN/PROPN, a comma separates
them, the appositive is a full noun phrase (has a det/poss/amod/compound/
nummod child or is a multi-word entity), and the head lies in the clause
root's subtree. The predicate is the synthesized string "is".
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


class ApposIdentity(Rule):
    NAME = "appos_identity"
    PRIORITY = 85
    TARGETS = (
        "Appositive identity: 'X, a Y, did Z' -> (X, is, Y). Requires a "
        "clause-root anchor, a comma, and a full-NP appositive; the "
        "predicate is the synthesized string 'is'."
    )
    EXAMPLES = [
        ("Marie Curie, a physicist, won the Nobel Prize.",
         [("Marie Curie", "is", "physicist")]),
        ("Tokyo, the capital of Japan, is bustling.",
         [("Tokyo", "is", "capital")]),
        ("Einstein, a German-born scientist, developed relativity.",
         [("Einstein", "is", "scientist")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        for token in clause.span:
            if token.pos_ not in {"NOUN", "PROPN"}:
                continue
            if token.dep_ == "appos":
                continue
            if not _is_in_clause_root_subtree(token, clause):
                continue
            for child in token.children:
                if child.dep_ != "appos":
                    continue
                if child.pos_ not in {"NOUN", "PROPN"}:
                    continue
                if not _has_comma_between(token, child):
                    continue
                if not _looks_like_full_np(child):
                    continue
                yield Candidate(
                    subject_head=token,
                    predicate_head=child,
                    arg_head=child,
                    role="object",
                    prep=None,
                    source_rule=self.NAME,
                    synthesized_predicate_text="is",
                )


def _is_in_clause_root_subtree(token, clause: Clause) -> bool:
    cur = token
    for _ in range(64):
        if cur.i == clause.root.i:
            return True
        if cur.head.i == cur.i:
            return False
        cur = cur.head
    return False


def _has_comma_between(a, b) -> bool:
    lo, hi = sorted([a.i, b.i])
    for t in a.doc[lo + 1: hi]:
        if t.text == ",":
            return True
    return False


def _looks_like_full_np(token) -> bool:
    for c in token.children:
        if c.dep_ in {"det", "poss", "amod", "compound", "nummod"}:
            return True
    if token.ent_type_ and len(list(token.subtree)) > 1:
        return True
    return False
