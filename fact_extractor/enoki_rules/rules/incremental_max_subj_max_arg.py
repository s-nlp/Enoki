"""L5 — cross-product (max-subtree subject × max-subtree arg) variant.

Combines the two accepted maximal expansions:
  - subj_span_subtree=True (N17: appositive NER absorption etc.)
  - arg_span_subtree=True  (N15: full arg modifier subtree)

EnokiQA gold's widest gold-variant entries pair the maximal subject
with the maximal arg ("Dayton, Montana" + "small settlement located
in Chouteau County, near the eastern border of the state"). Existing
4 width combinations from N14/N15/N17/source rules don't hit this
specific (max, max) cross-product.

Fires on the same canonical SVO patterns (core_svo / copula_be /
core_attr / core_oprd / prep_object).
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


_ADJUNCT_PREPS = frozenset({
    "as", "like", "unlike", "without", "despite", "because", "due",
    "amid", "amidst", "versus", "vs", "notwithstanding",
})


class IncrementalMaxSubjMaxArg(Rule):
    NAME = "incremental_max_subj_max_arg"
    PRIORITY = 9
    TARGETS = (
        "Cross-product (max-subject × max-arg) variant for the canonical "
        "SVO patterns. Both subject and arg use full subtree span."
    )
    EXAMPLES = [
        ("Alice signed the contract.",
         [("Alice", "signed", "the contract")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if not clause.subject_candidates:
            return
        is_be_root = verb.lemma_ == "be"

        for c in verb.children:
            if c.dep_ in {"attr", "acomp", "oprd"}:
                if c.dep_ in {"attr", "acomp"}:
                    if is_be_root:
                        pass
                    elif verb.pos_ == "VERB" and c.dep_ == "attr":
                        pass
                    else:
                        continue
                elif c.dep_ == "oprd":
                    if verb.pos_ != "VERB":
                        continue
                for subj in clause.subject_candidates:
                    if _bad_subj(subj):
                        continue
                    yield Candidate(
                        subject_head=subj,
                        predicate_head=verb,
                        arg_head=c,
                        role="object",
                        prep=None,
                        source_rule=self.NAME,
                        subj_span_subtree=True,
                        arg_span_subtree=True,
                    )

        if verb.pos_ == "VERB":
            for c in verb.children:
                if c.dep_ == "dobj":
                    for subj in clause.subject_candidates:
                        if _bad_subj(subj):
                            continue
                        yield Candidate(
                            subject_head=subj,
                            predicate_head=verb,
                            arg_head=c,
                            role="object",
                            prep=None,
                            source_rule=self.NAME,
                            subj_span_subtree=True,
                            arg_span_subtree=True,
                        )

        if verb.pos_ == "VERB":
            for prep_tok in verb.children:
                if prep_tok.dep_ != "prep":
                    continue
                if prep_tok.lower_ in _ADJUNCT_PREPS:
                    continue
                if prep_tok.lower_ == "by" and any(
                    c.dep_ == "nsubjpass" for c in verb.children
                ):
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
                    yield Candidate(
                        subject_head=subj,
                        predicate_head=verb,
                        arg_head=pobj,
                        role="other",
                        prep=prep_tok.lower_,
                        source_rule=self.NAME,
                        subj_span_subtree=True,
                        arg_span_subtree=True,
                    )


def _bad_subj(subj) -> bool:
    return subj.tag_ in {"WDT", "WP", "WP$"} or subj.lower_ in {
        "who", "which", "that", "whom", "whose"
    }
