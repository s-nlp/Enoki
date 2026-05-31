"""L4 — clausal complement argument (ccomp).

A VERB root with a ``ccomp`` child (clausal complement with its own
subject: "He said [she left]").  Emits (subject, verb, ccomp-clause)
with ``arg_span_subtree=True`` so the pipeline materialises the whole
embedded clause as the argument span.
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from .base import Rule


class CcompArg(Rule):
    NAME = "ccomp_arg"
    PRIORITY = 50
    TARGETS = (
        "L4 clausal complement: root VERB + nsubj + ccomp child. "
        "Emits (subject, verb, embedded-clause) with arg_span_subtree=True. "
        "'He said she left' -> (He, said, she left)."
    )
    EXAMPLES = [
        ("He said she left.", [("He", "said", "she left")]),
        ("Scientists believe the universe is expanding.",
         [("Scientists", "believe", "the universe is expanding")]),
        ("She thinks he is wrong.", [("She", "thinks", "he is wrong")]),
        ("They reported that the bridge collapsed.",
         [("They", "reported", "the bridge collapsed")]),
        ("The study shows that exercise reduces stress.",
         [("study", "shows", "exercise reduces stress")]),
        ("Experts claim the policy will fail.",
         [("Experts", "claim", "the policy will fail")]),
        ("The report states that emissions increased.",
         [("report", "states", "emissions increased")]),
        ("She noticed that the door was open.",
         [("She", "noticed", "the door was open")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if verb.pos_ != "VERB":
            return
        if not clause.subject_candidates:
            return
        cc = next(
            (c for c in verb.children if c.dep_ == "ccomp"),
            None,
        )
        if cc is None:
            return
        for subj in clause.subject_candidates:
            yield Candidate(
                subject_head=subj,
                predicate_head=verb,
                arg_head=cc,
                role="other",
                prep=None,
                source_rule=self.NAME,
                arg_span_subtree=True,
            )
