"""Maximal-argument variant of the canonical SVO patterns.

Re-emits the copula/attr/oprd, direct-object and prepositional-object
candidates with ``arg_span_subtree=True``, so the argument is the full
comma-trimmed subtree of its head (head plus all modifiers and attached
PPs) instead of the standard noun-phrase expansion. Complements
``incremental_minimal_arg`` at the other end of the granularity range.
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


_ADJUNCT_PREPS = frozenset({
    "as", "like", "unlike", "without", "despite", "because", "due",
    "amid", "amidst", "versus", "vs", "notwithstanding",
})


class IncrementalMaximalArg(Rule):
    NAME = "incremental_maximal_arg"
    PRIORITY = 9
    TARGETS = (
        "Maximal-granularity argument variants of the canonical SVO patterns "
        "(core_svo / copula_be / core_attr / core_oprd / prep_object): same "
        "subject and predicate, argument span is the full comma-trimmed "
        "subtree of the argument head."
    )
    EXAMPLES = [
        ("Alice signed the contract carefully.",
         [("Alice", "signed", "the contract")]),
        ("Paris is the capital of France.",
         [("Paris", "is", "the capital of France")]),
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
                        arg_span_subtree=True,
                    )


def _bad_subj(subj) -> bool:
    return subj.tag_ in {"WDT", "WP", "WP$"} or subj.lower_ in {
        "who", "which", "that", "whom", "whose"
    }
