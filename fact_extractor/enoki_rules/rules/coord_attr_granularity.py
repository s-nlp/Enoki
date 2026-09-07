"""Head-only and full-subtree variants of ``coord_attr``.

For each ``conj`` sibling of a be-copula attr/acomp, emits the conjunct at two
argument widths: the bare head token and the full subtree. Counterpart of
``coord_object_granularity``.
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


class CoordAttrGranularity(Rule):
    NAME = "coord_attr_granularity"
    PRIORITY = 12
    TARGETS = (
        "Granularity variants (head-only + subtree) for be-copula "
        "attr/acomp conjuncts. Companion to coord_attr."
    )
    EXAMPLES = [
        ("Her name was Anne or Annie.",
         [("name", "was", "Annie")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if verb.lemma_ != "be":
            return
        if not clause.subject_candidates:
            return
        for attr in verb.children:
            if attr.dep_ not in {"attr", "acomp"}:
                continue
            for conj in attr.children:
                if conj.dep_ != "conj":
                    continue
                for subj in clause.subject_candidates:
                    if _bad_subj(subj):
                        continue
                    yield Candidate(
                        subject_head=subj,
                        predicate_head=verb,
                        arg_head=conj,
                        role="object",
                        prep=None,
                        source_rule=self.NAME,
                        arg_minimal_only=True,
                    )
                    yield Candidate(
                        subject_head=subj,
                        predicate_head=verb,
                        arg_head=conj,
                        role="object",
                        prep=None,
                        source_rule=self.NAME,
                        arg_span_subtree=True,
                    )


def _bad_subj(subj) -> bool:
    return subj.tag_ in {"WDT", "WP", "WP$"} or subj.lower_ in {
        "who", "which", "that", "whom", "whose"
    }
