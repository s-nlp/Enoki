"""L1 — copular predication with 'be'.

Root lemma 'be' with an attr or acomp child -> (subject, <be form>, predicate-nominal/adj).
Deliberately complements core_attr which requires root VERB with lemma != 'be'.
Example: 'Paris is the capital of France.' -> (Paris, is, capital).
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


class CopulaBe(Rule):
    NAME = "copula_be"
    PRIORITY = 20
    TARGETS = (
        "L1 copular predication: root lemma 'be' with attr or acomp child "
        "-> (subject, <be surface form>, predicate-nominal or adjective). "
        "'Paris is the capital of France.' -> (Paris, is, capital)."
    )
    EXAMPLES = [
        ("Paris is the capital of France.",
         [("Paris", "is", "capital")]),
        ("The sky is blue.",
         [("sky", "is", "blue")]),
        ("She was a teacher.",
         [("She", "was", "teacher")]),
        ("Water is a liquid.",
         [("Water", "is", "liquid")]),
        ("The results were surprising.",
         [("results", "were", "surprising")]),
        ("He is the director of the company.",
         [("He", "is", "director")]),
        ("The meeting was productive.",
         [("meeting", "was", "productive")]),
        ("Dogs are loyal animals.",
         [("Dogs", "are", "animals")]),
        # Q5: catenative/raising adjective predications skipped (no triplet)
        ("She is able to swim.", []),
        ("He is likely to win.", []),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if verb.lemma_ != "be":
            return
        if not clause.subject_candidates:
            return
        arg = next(
            (c for c in verb.children if c.dep_ in {"attr", "acomp"}),
            None,
        )
        if arg is None:
            return
        # Q5 precision: skip catenative/raising adjective predications
        # ("X is able/likely/ready/going TO ...").  When the attr/acomp
        # carries its own xcomp (a to-infinitival complement), the
        # informative predication is the infinitival, not (subj, is,
        # able); the bare copular triplet is gold-uncredited
        # (acomp/attr-with-xcomp: ~25 FP / ~2 TP on dev).
        if any(c.dep_ == "xcomp" for c in arg.children):
            return
        for subj in clause.subject_candidates:
            # Guard 1: skip relativizer subjects (relative-clause gap)
            if subj.tag_ in {"WDT", "WP", "WP$"} or subj.lower_ in {
                "who", "which", "that", "whom", "whose"
            }:
                continue
            # Guard 2: skip pleonastic 'it' with extraposed clause
            if subj.lower_ == "it" and any(
                c.dep_ in {"ccomp", "csubj", "csubjpass"} for c in verb.children
            ):
                continue
            yield Candidate(
                subject_head=subj,
                predicate_head=verb,
                arg_head=arg,
                role="object",
                prep=None,
                source_rule=self.NAME,
            )
