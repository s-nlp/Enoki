"""L5 — be + ADJ acomp + prep → (subj, is-acomp-prep, pobj).

EnokiQA gold treats composite copular-adjective+preposition phrases as
single predicates flattened onto the matrix subject:

    "The ladder metaphor is rich with symbolism."
    -> (The ladder metaphor, is rich with, symbolism)

    "The concept was crucial for measurements."
    -> (concept, was crucial for, measurements)

copula_be emits ``(subj, is, acomp)`` for the simple copular layer but
misses the prep+pobj extension. This sibling rule emits the composite
predicate keeping the be-subject, the acomp ADJ, and the prep-pobj
argument — non-deduped with copula_be (different arg_head / predicate
surface).

Tight conditions:
  - root lemma 'be' + nsubj
  - acomp child whose POS is ADJ (not VBN; VBN passives are
    svo_passive's domain)
  - the acomp has a prep child with pobj
  - emit (subj, acomp, pobj, prep=prep_lower)

Gated on the EnokiQA val sample.
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
        # ADJ acomp only — VBN passives are svo_passive's territory.
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
