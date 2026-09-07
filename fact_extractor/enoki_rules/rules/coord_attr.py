"""Coordinated copular complements: "X was A or B" -> (X, was, B).

``copula_be`` emits the first attr/acomp only; this rule emits one triplet
per ``conj`` sibling of that complement. Counterpart of ``coord_object`` for
the be-copula.
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


class CoordAttr(Rule):
    NAME = "coord_attr"
    PRIORITY = 12
    TARGETS = (
        "Coordinated attr/acomp distribution for be-copula: for each "
        "conj child of an attr/acomp under be, emit a separate triplet. "
        "'X was A or B' -> (X, was, A), (X, was, B)."
    )
    EXAMPLES = [
        ("Her name was Anne or Annie.",
         [("name", "was", "Annie")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if verb.lemma_ != "be":
            return
        if not clause.subject_candidates:
            return
        for attr in verb.children:
            if attr.dep_ not in {"attr", "acomp"}:
                continue
            for conj in attr.children:
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


def _bad_subj(subj) -> bool:
    return subj.tag_ in {"WDT", "WP", "WP$"} or subj.lower_ in {
        "who", "which", "that", "whom", "whose"
    }
