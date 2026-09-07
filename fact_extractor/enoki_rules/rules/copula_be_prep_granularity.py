"""Subject/argument width variants of ``copula_be_prep``.

For the same be + prep + pobj construction, emits four extra candidates:
head-only argument, full-subtree argument, full-subtree subject, and both
subtrees. The ``incremental_*`` rules skip ``be`` roots, so this fills that
gap.
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


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
                # Pleonastic 'it' with an extraposed clause.
                if subj.lower_ == "it" and any(
                    c.dep_ in {"ccomp", "csubj", "csubjpass"}
                    for c in verb.children
                ):
                    continue
                yield Candidate(
                    subject_head=subj,
                    predicate_head=verb,
                    arg_head=pobj,
                    role="object",
                    prep=prep_tok.lower_,
                    source_rule=self.NAME,
                    arg_minimal_only=True,
                )
                yield Candidate(
                    subject_head=subj,
                    predicate_head=verb,
                    arg_head=pobj,
                    role="object",
                    prep=prep_tok.lower_,
                    source_rule=self.NAME,
                    arg_span_subtree=True,
                )
                yield Candidate(
                    subject_head=subj,
                    predicate_head=verb,
                    arg_head=pobj,
                    role="object",
                    prep=prep_tok.lower_,
                    source_rule=self.NAME,
                    subj_span_subtree=True,
                )
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
