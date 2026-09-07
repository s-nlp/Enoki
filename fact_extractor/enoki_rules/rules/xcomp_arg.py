"""Open clausal complement argument (``xcomp``).

A verb root with an ``xcomp`` child (open clausal complement with a shared
subject: "She wants [to leave]"). Emits (subject, verb, xcomp-clause) with
``arg_span_subtree=True`` so the whole embedded clause becomes the
argument span.
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


class XcompArg(Rule):
    NAME = "xcomp_arg"
    PRIORITY = 50
    TARGETS = (
        "Open clausal complement: root VERB + nsubj + xcomp child. "
        "Emits (subject, verb, embedded-clause) with arg_span_subtree=True. "
        "Skips passive to-infinitival xcomp (VBN+to). "
        "'She wants to leave' -> (She, wants, to leave)."
    )
    EXAMPLES = [
        ("She wants to leave.", [("She", "wants", "to leave")]),
        ("They tried to open the door.",
         [("They", "tried", "to open the door")]),
        ("He decided to stay.", [("He", "decided", "to stay")]),
        ("She began to sing.", [("She", "began", "to sing")]),
        ("They expected to win the match.",
         [("They", "expected", "to win the match")]),
        ("The company agreed to pay the fine.",
         [("company", "agreed", "to pay the fine")]),
        ("He promised to return.", [("He", "promised", "to return")]),
        ("She refused to sign the contract.",
         [("She", "refused", "to sign the contract")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if verb.pos_ != "VERB":
            return
        if not clause.subject_candidates:
            return
        xc = next(
            (c for c in verb.children if c.dep_ == "xcomp"),
            None,
        )
        if xc is None:
            return
        # A passive to-infinitival xcomp ("... to be seen") is not an argument.
        if xc.tag_ == "VBN" and any(
            c.lower_ == "to" and c.dep_ in {"aux", "mark"}
            for c in xc.children
        ):
            return
        for subj in clause.subject_candidates:
            yield Candidate(
                subject_head=subj,
                predicate_head=verb,
                arg_head=xc,
                role="other",
                prep=None,
                source_rule=self.NAME,
                arg_span_subtree=True,
            )
