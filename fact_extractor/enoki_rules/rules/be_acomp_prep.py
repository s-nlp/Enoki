"""Copula + adjectival complement + preposition: "X is rich with Y".

Root lemma ``be`` with an ADJ ``acomp`` that carries a prep + pobj. Emits
(subj, acomp, pobj) with the preposition bound into the predicate, alongside
the plain (subj, is, acomp) that ``copula_be`` emits. VBN complements are left
to ``svo_passive``.

    "The concept was crucial for measurements."
    -> (concept, crucial for, measurements)
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


class BeAcompPrep(Rule):
    NAME = "be_acomp_prep"
    PRIORITY = 25
    TARGETS = (
        "Copula be + ADJ acomp + prep -> (subj, acomp prep, pobj). "
        "Covers composite predicates ('is rich with', 'is crucial for', "
        "'is suitable for') that copula_be misses by emitting only "
        "(subj, is, acomp)."
    )
    EXAMPLES = [
        ("The metaphor is rich with symbolism.",
         [("metaphor", "rich with", "symbolism")]),
        ("The concept was crucial for measurements.",
         [("concept", "crucial for", "measurements")]),
        ("The plan is suitable for beginners.",
         [("plan", "suitable for", "beginners")]),
        ("The book is full of insights.",
         [("book", "full of", "insights")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if verb.lemma_ != "be":
            return
        if not clause.subject_candidates:
            return
        acomp = next(
            (c for c in verb.children if c.dep_ == "acomp"),
            None,
        )
        if acomp is None:
            return
        if acomp.pos_ != "ADJ":
            return
        for prep_tok in acomp.children:
            if prep_tok.dep_ != "prep":
                continue
            pobj = next(
                (g for g in prep_tok.children if g.dep_ == "pobj"),
                None,
            )
            if pobj is None:
                continue
            for subj in clause.subject_candidates:
                if subj.tag_ in {"WDT", "WP", "WP$"} or subj.lower_ in {
                    "who", "which", "that", "whom", "whose"
                }:
                    continue
                yield Candidate(
                    subject_head=subj,
                    predicate_head=acomp,
                    arg_head=pobj,
                    role="object",
                    prep=prep_tok.lower_,
                    source_rule=self.NAME,
                )
