"""L5 — granularity variants for copula_be_prep pattern.

copula_be_prep emits ``(be-subj, be prep, pobj)`` for "Paris is in
France" type sentences (be + nsubj + prep + pobj, no attr/acomp).

N14/N15/N17/N18 cover the canonical SVO patterns (core_svo /
copula_be / core_attr / core_oprd / prep_object), but their VERB-only
filter skips the be-AUX root that copula_be_prep targets.

This rule adds granularity variants for that gap:
  - (subj_med, arg_min)  — head-only pobj
  - (subj_med, arg_max)  — full pobj subtree
  - (subj_max, arg_med)  — subj subtree, medium pobj
  - (subj_max, arg_max)  — both maximal
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from .base import Rule


class CopulaBePrepGranularity(Rule):
    NAME = "copula_be_prep_granularity"
    PRIORITY = 9
    TARGETS = (
        "Granularity variants for copula_be_prep: be + nsubj + prep + "
        "pobj (no attr/acomp) -> emit minimal-arg, maximal-arg, "
        "maximal-subj+medium-arg, maximal-subj+maximal-arg variants."
    )
    EXAMPLES = [
        ("Paris is in France.",
         [("Paris", "is in", "France")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if verb.lemma_ != "be":
            return
        if not clause.subject_candidates:
            return
        # Mirror copula_be_prep's guards: no attr/acomp on the be root.
        if any(c.dep_ in {"attr", "acomp"} for c in verb.children):
            return
        for prep_tok in verb.children:
            if prep_tok.dep_ != "prep":
                continue
            pobj = next(
                (g for g in prep_tok.children if g.dep_ == "pobj"),
                None,
            )
            if pobj is None:
                continue
            if pobj.tag_ in {"WDT", "WP", "WP$"} or pobj.lower_ in {
                "who", "which", "that", "whom", "where"
            }:
                continue
            for subj in clause.subject_candidates:
                if _bad_subj(subj):
                    continue
                # Skip pleonastic 'it' with ccomp extraposition (mirror
                # copula_be guard).
                if subj.lower_ == "it" and any(
                    c.dep_ in {"ccomp", "csubj", "csubjpass"}
                    for c in verb.children
                ):
                    continue
                # (subj_med, arg_min)
                yield Candidate(
                    subject_head=subj,
                    predicate_head=verb,
                    arg_head=pobj,
                    role="object",
                    prep=prep_tok.lower_,
                    source_rule=self.NAME,
                    arg_minimal_only=True,
                )
                # (subj_med, arg_max)
                yield Candidate(
                    subject_head=subj,
                    predicate_head=verb,
                    arg_head=pobj,
                    role="object",
                    prep=prep_tok.lower_,
                    source_rule=self.NAME,
                    arg_span_subtree=True,
                )
                # (subj_max, arg_med)
                yield Candidate(
                    subject_head=subj,
                    predicate_head=verb,
                    arg_head=pobj,
                    role="object",
                    prep=prep_tok.lower_,
                    source_rule=self.NAME,
                    subj_span_subtree=True,
                )
                # (subj_max, arg_max)
                yield Candidate(
                    subject_head=subj,
                    predicate_head=verb,
                    arg_head=pobj,
                    role="object",
                    prep=prep_tok.lower_,
                    source_rule=self.NAME,
                    subj_span_subtree=True,
                    arg_span_subtree=True,
                )


def _bad_subj(subj) -> bool:
    return subj.tag_ in {"WDT", "WP", "WP$"} or subj.lower_ in {
        "who", "which", "that", "whom", "whose"
    }
