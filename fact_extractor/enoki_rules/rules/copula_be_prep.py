"""Copula + prepositional complement: "X is in Y".

Root lemma ``be`` whose complement is a prep + pobj rather than an
attr/acomp (those belong to ``copula_be``). Emits (subject, <be form> prep,
pobj). Relativizer subjects and pleonastic ``it`` are skipped.

    "Paris is in France." -> (Paris, is in, France)
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


class CopulaBePrep(Rule):
    NAME = "copula_be_prep"
    PRIORITY = 15
    TARGETS = (
        "Copular BE + nsubj + prep child whose pobj is the argument. "
        "Complements copula_be (which handles attr/acomp) by targeting "
        "locative/directional complements. "
        "e.g. 'Paris is in France' -> (Paris, is in, France)."
    )
    EXAMPLES = [
        ("Paris is in France.",
         [("Paris", "is in", "France")]),
        ("The meeting is at noon.",
         [("meeting", "is at", "noon")]),
        ("The book is on the shelf.",
         [("book", "is on", "shelf")]),
        ("She is from Brazil.",
         [("She", "is from", "Brazil")]),
        ("Germany is in Europe.",
         [("Germany", "is in", "Europe")]),
        ("He is in the hospital.",
         [("He", "is in", "hospital")]),
        ("They are at the airport.",
         [("They", "are at", "airport")]),
        ("She was in Paris.",
         [("She", "was in", "Paris")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if verb.lemma_ != "be":
            return
        if not clause.subject_candidates:
            return

        children = list(verb.children)

        subjs = [
            s for s in clause.subject_candidates
            if not (
                s.tag_ in {"WDT", "WP", "WP$"}
                or s.lower_ in {"who", "which", "that", "whom", "whose"}
            )
        ]
        if not subjs:
            return

        # Pleonastic 'it' with an extraposed clause.
        subjs = [
            s for s in subjs
            if not (
                s.lower_ == "it"
                and any(c.dep_ in {"ccomp", "csubj", "csubjpass"} for c in children)
            )
        ]
        if not subjs:
            return

        has_attr_acomp = any(c.dep_ in {"attr", "acomp"} for c in children)
        if has_attr_acomp:
            return

        for prep_tok in children:
            if prep_tok.dep_ != "prep":
                continue
            pobj = next(
                (gc for gc in prep_tok.children if gc.dep_ == "pobj"),
                None,
            )
            if pobj is None:
                continue
            for subj in subjs:
                yield Candidate(
                    subject_head=subj,
                    predicate_head=verb,
                    arg_head=pobj,
                    role="other",
                    prep=prep_tok.lower_,
                    source_rule=self.NAME,
                )
