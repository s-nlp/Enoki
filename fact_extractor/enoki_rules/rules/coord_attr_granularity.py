"""L5 — granularity variants for coord_attr (min + max arg).

Mirrors ``coord_object_granularity`` (N22) but for coord_attr's
conjuncts under be-copula's attr/acomp.

Sample 4 pattern variants gold may credit at multiple object widths:
  "Her name was Saint John Scholasticus or John Sinaites"
  -> (name, was, Scholasticus) [min]
     (name, was, Saint John Scholasticus) [med — by coord_attr]
     (name, was, Saint John Scholasticus the Climacus) [max if appos]
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from .base import Rule


class CoordAttrGranularity(Rule):
    NAME = "coord_attr_granularity"
    PRIORITY = 12
    TARGETS = (
        "Granularity variants (head-only + subtree) for be-copula "
        "attr/acomp conjuncts. Companion to coord_attr (N22)."
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
                    # minimal (head-only conj)
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


def _bad_subj(subj) -> bool:
    return subj.tag_ in {"WDT", "WP", "WP$"} or subj.lower_ in {
        "who", "which", "that", "whom", "whose"
    }
