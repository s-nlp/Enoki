"""Temporal ``npadvmod`` adjunct as a time argument.

Time and duration phrases attached directly to the verb as ``npadvmod``
("He arrived yesterday", "She worked Monday") have no preposition, so
``prep_object`` and ``core_svo`` miss them. Emit
``(subject, verb, npadvmod)`` with role ``time`` when the modifier's lemma
is in a closed set of day, weekday, duration-unit and part-of-day nouns.
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule

_TIME_LEMMAS = frozenset({
    # absolute days
    "today", "yesterday", "tomorrow", "tonight",
    # weekdays
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday",
    # duration units
    "year", "month", "week", "day", "hour", "minute", "second",
    "decade", "century", "millennium",
    # parts of day
    "morning", "afternoon", "evening", "night", "noon", "midnight",
    "dawn", "dusk",
})


class NpAdvmodTime(Rule):
    NAME = "npadvmod_time"
    PRIORITY = 60
    TARGETS = (
        "Temporal npadvmod adjunct: root VERB + active nsubj + "
        "npadvmod child whose lemma is in a closed time/duration set "
        "-> (subj, verb, time_phrase, role='time'). "
        "'He arrived yesterday.' -> (He, arrived, yesterday)."
    )
    EXAMPLES = [
        ("He arrived yesterday.",
         [("He", "arrived", "yesterday")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if verb.pos_ != "VERB":
            return
        if not any(c.dep_ == "nsubj" for c in verb.children):
            return
        for c in verb.children:
            if c.dep_ != "npadvmod":
                continue
            if c.lemma_.lower() not in _TIME_LEMMAS:
                continue
            for subj in clause.subject_candidates:
                if subj.tag_ in {"WDT", "WP", "WP$"} or subj.lower_ in {
                    "who", "which", "that", "whom", "whose"
                }:
                    continue
                yield Candidate(
                    subject_head=subj,
                    predicate_head=verb,
                    arg_head=c,
                    role="time",
                    prep=None,
                    source_rule=self.NAME,
                )
