"""Coordinated objects: "X verb Y and Z" -> (X, verb, Z).

For each ``conj`` sibling of a direct object or prepositional object of a
VERB root, emits a separate triplet with that conjunct as the argument.
Adjunct prepositions and passive by-agents are skipped as in ``prep_object``.
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


_ADJUNCT_PREPS = frozenset({
    "as", "like", "unlike", "without", "despite", "because", "due",
    "amid", "amidst", "versus", "vs", "notwithstanding",
})


class CoordObject(Rule):
    NAME = "coord_object"
    PRIORITY = 12
    TARGETS = (
        "Coordinated dobj/pobj distribution: for each conj child of "
        "the dobj or pobj, emit a separate triplet with that conj as "
        "the arg. 'X verb Y and Z' -> (X, verb, Y), (X, verb, Z)."
    )
    EXAMPLES = [
        ("She bought apples and oranges.",
         [("She", "bought", "oranges")]),
        ("They visited Paris and London.",
         [("They", "visited", "London")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if verb.pos_ != "VERB":
            return
        if not clause.subject_candidates:
            return

        dobj = next(
            (c for c in verb.children if c.dep_ == "dobj"),
            None,
        )
        if dobj is not None:
            for conj in dobj.children:
                if conj.dep_ != "conj":
                    continue
                for subj in clause.subject_candidates:
                    if _bad_subj(subj):
                        continue
                    yield Candidate(
                        subject_head=subj,
                        predicate_head=verb,
                        arg_head=conj,
                        role="object",
                        prep=None,
                        source_rule=self.NAME,
                    )

        for prep_tok in verb.children:
            if prep_tok.dep_ != "prep":
                continue
            if prep_tok.lower_ in _ADJUNCT_PREPS:
                continue
            if prep_tok.lower_ == "by" and any(
                c.dep_ == "nsubjpass" for c in verb.children
            ):
                continue
            pobj = next(
                (g for g in prep_tok.children if g.dep_ == "pobj"),
                None,
            )
            if pobj is None:
                continue
            for conj in pobj.children:
                if conj.dep_ != "conj":
                    continue
                for subj in clause.subject_candidates:
                    if _bad_subj(subj):
                        continue
                    yield Candidate(
                        subject_head=subj,
                        predicate_head=verb,
                        arg_head=conj,
                        role="other",
                        prep=prep_tok.lower_,
                        source_rule=self.NAME,
                    )


def _bad_subj(subj) -> bool:
    return subj.tag_ in {"WDT", "WP", "WP$"} or subj.lower_ in {
        "who", "which", "that", "whom", "whose"
    }
