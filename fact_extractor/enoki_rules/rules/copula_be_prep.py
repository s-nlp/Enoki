"""Refinement R12 — copular BE (AUX) + prepositional locative/directional complement.

Handles copular be + prep + pobj constructions, e.g. 'Paris is in France'.
The root is a form of 'be' (pos=AUX) and the complement is a prepositional
phrase rather than an attr/acomp (which copula_be already handles).  Examples:
  "Paris is in France."         -> (Paris, is in, France)
  "She was in Paris."           -> (She, was in, Paris)
  "Germany is in Europe."       -> (Germany, is in, Europe)

The predicate root is the copula (lemma='be', pos='AUX'), so this
complements copula_be without overlapping it.

Precision guard: require an explicit nsubj (skip existential/expletive
constructions without a true topic subject).
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


class CopulaBePrep(Rule):
    NAME = "copula_be_prep"
    PRIORITY = 15
    TARGETS = (
        "Refinement R12 recall: copular BE (AUX) + nsubj + prep child whose "
        "pobj is the argument (be + prep + pobj). Complements copula_be (which "
        "handles attr/acomp) by targeting locative/directional complements. "
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
        # Target copular 'be' (any surface form: is, was, are, were, been)
        if verb.lemma_ != "be":
            return
        if not clause.subject_candidates:
            return

        children = list(verb.children)

        # Guard: skip relativizer subjects (same as copula_be)
        subjs = [
            s for s in clause.subject_candidates
            if not (
                s.tag_ in {"WDT", "WP", "WP$"}
                or s.lower_ in {"who", "which", "that", "whom", "whose"}
            )
        ]
        if not subjs:
            return

        # Guard: skip pleonastic 'it' with extraposed clause
        subjs = [
            s for s in subjs
            if not (
                s.lower_ == "it"
                and any(c.dep_ in {"ccomp", "csubj", "csubjpass"} for c in children)
            )
        ]
        if not subjs:
            return

        # Only fire for prep + pobj (not attr/acomp — those go to copula_be)
        # Skip if there is already an attr or acomp (copula_be covers that)
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
