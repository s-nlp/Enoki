"""Split a date prepositional phrase into per-component time arguments.

For a temporal ``prep`` on the root verb whose ``pobj`` is a date (month
name, four-digit year, or DATE/TIME entity), emit one
``(subject, verb prep, N)`` candidate for each numeric ``nummod``/``appos``
child of the date head, so "occurred on September 15, 1648" also yields
``15`` and ``1648`` as head-only time arguments.
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


_TEMPORAL_PREPS = frozenset({
    "in", "on", "at", "during", "by", "before", "after",
    "since", "until", "from",
})

_MONTHS = frozenset({
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct",
    "nov", "dec",
})


def _is_date_pobj(tok) -> bool:
    if tok.ent_type_ in {"DATE", "TIME"}:
        return True
    if tok.lower_ in _MONTHS:
        return True
    txt = tok.text
    if txt.isdigit() and len(txt) == 4 and 1500 <= int(txt) <= 2099:
        return True
    return False


class DateComponentSplit(Rule):
    NAME = "date_component_split"
    PRIORITY = 9
    TARGETS = (
        "Split a date-PP's NUM children (day, year) into separate "
        "triplets. 'occurred on September 15, 1648' -> "
        "(X, occurred on, 15) (X, occurred on, 1648)."
    )
    EXAMPLES = [
        ("The event occurred on September 15, 1648.",
         [("event", "occurred on", "1648")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if verb.pos_ != "VERB":
            return
        if not clause.subject_candidates:
            return
        for prep_tok in verb.children:
            if prep_tok.dep_ != "prep":
                continue
            if prep_tok.lower_ not in _TEMPORAL_PREPS:
                continue
            pobj = next(
                (g for g in prep_tok.children if g.dep_ == "pobj"),
                None,
            )
            if pobj is None or not _is_date_pobj(pobj):
                continue
            for child in pobj.children:
                if child.dep_ not in {"nummod", "appos"}:
                    continue
                if not (child.like_num or child.tag_ == "CD"):
                    continue
                for subj in clause.subject_candidates:
                    if _bad_subj(subj):
                        continue
                    yield Candidate(
                        subject_head=subj,
                        predicate_head=verb,
                        arg_head=child,
                        role="time",
                        prep=prep_tok.lower_,
                        source_rule=self.NAME,
                        arg_minimal_only=True,
                    )


def _bad_subj(subj) -> bool:
    return subj.tag_ in {"WDT", "WP", "WP$"} or subj.lower_ in {
        "who", "which", "that", "whom", "whose"
    }
