"""L0 — attr complement of a non-copular verb.

Captures constructions like "He became chairman" / "She stayed president"
where the root is a non-be VERB and the complement is attached via `attr`.
Copular `be` is reserved for L1 so that identity/classification facts are
handled by the dedicated copula rule.
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


class CoreAttr(Rule):
    NAME = "core_attr"
    PRIORITY = 10
    TARGETS = (
        "L0 attr complement of a non-be verb: root VERB (lemma != 'be') "
        "with a nominal subject and an attr child. "
        "'He became chairman' -> (He, became, chairman)."
    )
    EXAMPLES = [
        ("He became chairman.", [("He", "became", "chairman")]),
        ("The mixture remained a liquid.", [("mixture", "remained", "liquid")]),
        ("She stayed president.", [("She", "stayed", "president")]),
        ("It turned a dark color.", [("It", "turned", "dark color")]),
        ("He grew a man of means.", [("He", "grew", "man")]),
        ("The economy became a global concern.", [("economy", "became", "global concern")]),
        ("He remained chairman.", [("He", "remained", "chairman")]),
        ("They became partners.", [("They", "became", "partners")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if verb.pos_ != "VERB":
            return
        if verb.lemma_ == "be":
            return
        if not clause.subject_candidates:
            return
        arg = next(
            (c for c in verb.children if c.dep_ == "attr"),
            None,
        )
        if arg is None:
            return
        for subj in clause.subject_candidates:
            yield Candidate(
                subject_head=subj,
                predicate_head=verb,
                arg_head=arg,
                role="object",
                prep=None,
                source_rule=self.NAME,
            )
