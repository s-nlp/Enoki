"""Core subject-verb-object: "Alice signed the contract".

The broadest rule: a clause whose root is a VERB emits
(subject, verb, dobj|acomp). Passive, copular, coordinated, clausal, and
nominal constructions are handled by the dedicated rules.
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


class CoreSVO(Rule):
    NAME = "core_svo"
    PRIORITY = 10
    TARGETS = (
        "Core predicate-argument: root VERB with a nominal subject "
        "and a direct object or adjectival complement. "
        "'Alice signed the contract' -> (Alice, signed, the contract)."
    )
    EXAMPLES = [
        ("Alice signed the contract.", [("Alice", "signed", "contract")]),
        ("The committee approved the budget.",
         [("committee", "approved", "budget")]),
        ("Geologists study rocks.", [("Geologists", "study", "rocks")]),
        ("The dog chased the cat.", [("dog", "chased", "cat")]),
        ("Engineers built the bridge.",
         [("Engineers", "built", "bridge")]),
        ("She wrote a novel.", [("She", "wrote", "novel")]),
        ("The company launched a product.",
         [("company", "launched", "product")]),
        ("Workers repaired the road.",
         [("Workers", "repaired", "road")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if verb.pos_ != "VERB":
            return
        if not clause.subject_candidates:
            return
        arg = next(
            (c for c in verb.children if c.dep_ in {"dobj", "acomp"}),
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
