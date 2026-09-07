"""Copular predication: "X is Y".

Root lemma ``be`` with an ``attr`` or ``acomp`` child emits
(subject, <be form>, complement). Complements ``core_attr``, which handles
non-``be`` verbs. Complements that carry their own to-infinitival ``xcomp``
("is able to swim") are skipped, since the informative predication is the
infinitival.

    "Paris is the capital of France." -> (Paris, is, capital)
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


class CopulaBe(Rule):
    NAME = "copula_be"
    PRIORITY = 20
    TARGETS = (
        "Copular predication: root lemma 'be' with attr or acomp child "
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
        # Catenative/raising predications ("is able to swim"): the
        # informative predication is the infinitival, not (subj, is, able).
        if any(c.dep_ == "xcomp" for c in arg.children):
            return
        for subj in clause.subject_candidates:
            if subj.tag_ in {"WDT", "WP", "WP$"} or subj.lower_ in {
                "who", "which", "that", "whom", "whose"
            }:
                continue
            # Pleonastic 'it' with an extraposed clause.
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
