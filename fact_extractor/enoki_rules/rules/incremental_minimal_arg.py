"""Head-only argument variant of the canonical SVO patterns.

Re-emits the candidates of ``core_svo``, ``copula_be``, ``core_attr``,
``core_oprd`` and ``prep_object`` with ``arg_minimal_only=True``, so the
argument span is the head token alone ("implication" alongside
"ethical implication"). Subject, predicate, role and preposition are
unchanged; both granularities survive dedup because the key uses the full
argument text.
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


# Same adjunct prepositions that prep_object skips.
_ADJUNCT_PREPS = frozenset({
    "as", "like", "unlike", "without", "despite", "because", "due",
    "amid", "amidst", "versus", "vs", "notwithstanding",
})


class IncrementalMinimalArg(Rule):
    NAME = "incremental_minimal_arg"
    PRIORITY = 9
    TARGETS = (
        "Head-only minimal-granularity argument variants of the canonical SVO "
        "patterns (core_svo / copula_be / core_attr / core_oprd / "
        "prep_object): same subject and predicate, argument span is the "
        "head token alone."
    )
    EXAMPLES = [
        ("Alice signed the contract.",
         [("Alice", "signed", "contract")]),
        ("She wrote a novel.",
         [("She", "wrote", "novel")]),
        ("Paris is the capital of France.",
         [("Paris", "is", "capital")]),
        ("They elected her president.",
         [("They", "elected", "president")]),
        ("She lives in Paris.",
         [("She", "lives in", "Paris")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if not clause.subject_candidates:
            return
        is_be_root = verb.lemma_ == "be"

        # copula_be / core_attr / core_oprd
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
                        arg_minimal_only=True,
                    )

        # core_svo
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
                            arg_minimal_only=True,
                        )

        # prep_object
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
                        arg_minimal_only=True,
                    )


def _bad_subj(subj) -> bool:
    return subj.tag_ in {"WDT", "WP", "WP$"} or subj.lower_ in {
        "who", "which", "that", "whom", "whose"
    }
