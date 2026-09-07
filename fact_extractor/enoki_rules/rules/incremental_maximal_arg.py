"""L5 — incremental maximal-arg variants for the canonical SVO patterns.

Companion to ``incremental_minimal_arg`` (N14) at the OTHER end of the
multi-granularity spectrum: the full ``arg_head.subtree`` (head plus
*all* transitive modifiers + PPs + acls), trimmed at comma/punct.

EnokiQA gold credits up to 3 object granularities per (s, p):
  - minimal   → head token alone (N14)
  - medium    → head + det/amod/compound (existing rules)
  - maximal   → head + transitive modifiers + PPs (THIS rule)

Same firing logic as N14 (mirrors core_svo / copula_be / core_attr /
core_oprd / prep_object).  Uses the existing ``arg_span_subtree=True``
mechanism (the pipeline's ``_subtree_span`` already handles bracketing
+ comma trim).

Survives dedup alongside N14 + the source rule via the full lemmatized
arg span text in the dedup key.
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
        "Maximal-granularity arg variants for the canonical SVO "
        "patterns (core_svo / copula_be / core_attr / core_oprd / "
        "prep_object).  Same (s, p) as the source rule; arg span is "
        "the full arg_head subtree (head + all modifiers + PPs, "
        "trimmed at comma).  Survives dedup alongside the minimal "
        "(N14) and medium-NP (source-rule) variants."
    )
    EXAMPLES = [
        # The pipeline emits the trimmed-subtree variant; gold may
        # credit longer-span variants of the medium-NP cases.
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
