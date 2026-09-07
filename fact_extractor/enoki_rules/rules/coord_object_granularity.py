"""L5 — granularity variants for coord_object (min + max).

Sample 2 pattern: "Each rung represents a specific challenge or
temptation that must be overcome."
  8 gold variants = 2 conjuncts × 4 granularities (head-only,
  head+amod, head+relcl, head+amod+relcl).

coord_object (N20) emits each conj at the medium-NP granularity.
This rule emits the minimal (head-only) and maximal (full subtree
incl. relcl) granularity variants per conj — completing the
cross-product.
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


_ADJUNCT_PREPS = frozenset({
    "as", "like", "unlike", "without", "despite", "because", "due",
    "amid", "amidst", "versus", "vs", "notwithstanding",
})


class CoordObjectGranularity(Rule):
    NAME = "coord_object_granularity"
    PRIORITY = 12
    TARGETS = (
        "Granularity variants for coord_object: for each conj child of "
        "a dobj or pobj, emit BOTH minimal (head-only) and maximal "
        "(full subtree, comma-trimmed) arg span variants."
    )
    EXAMPLES = [
        ("She bought apples and red oranges.",
         [("She", "bought", "oranges")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if verb.pos_ != "VERB":
            return
        if not clause.subject_candidates:
            return

        # dobj conjuncts
        dobj = next(
            (c for c in verb.children if c.dep_ == "dobj"),
            None,
        )
        if dobj is not None:
            for conj in dobj.children:
                if conj.dep_ != "conj":
                    continue
                for subj in clause.subject_candidates:
                    if _bad_subj(subj):
                        continue
                    # minimal (head-only)
                    yield Candidate(
                        subject_head=subj,
                        predicate_head=verb,
                        arg_head=conj,
                        role="object",
                        prep=None,
                        source_rule=self.NAME,
                        arg_minimal_only=True,
                    )
                    # maximal (subtree)
                    yield Candidate(
                        subject_head=subj,
                        predicate_head=verb,
                        arg_head=conj,
                        role="object",
                        prep=None,
                        source_rule=self.NAME,
                        arg_span_subtree=True,
                    )

        # prep+pobj conjuncts
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
            for conj in pobj.children:
                if conj.dep_ != "conj":
                    continue
                for subj in clause.subject_candidates:
                    if _bad_subj(subj):
                        continue
                    yield Candidate(
                        subject_head=subj,
                        predicate_head=verb,
                        arg_head=conj,
                        role="other",
                        prep=prep_tok.lower_,
                        source_rule=self.NAME,
                        arg_minimal_only=True,
                    )
                    yield Candidate(
                        subject_head=subj,
                        predicate_head=verb,
                        arg_head=conj,
                        role="other",
                        prep=prep_tok.lower_,
                        source_rule=self.NAME,
                        arg_span_subtree=True,
                    )


def _bad_subj(subj) -> bool:
    return subj.tag_ in {"WDT", "WP", "WP$"} or subj.lower_ in {
        "who", "which", "that", "whom", "whose"
    }
