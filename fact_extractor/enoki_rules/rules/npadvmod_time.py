"""L5 — npadvmod temporal/measure adjunct as time argument.

Some gold-credited time/duration arguments are attached to the verb as
``npadvmod`` (noun-phrase adverbial modifier), not as a prepositional
phrase: "Australia rose four places", "He arrived yesterday", "She
worked Monday".  prep_object (needs prep+pobj) and core_svo (needs
dobj) emit nothing for these — they fall into the GAP_addr FN bucket.

Rule: root VERB + active nsubj + npadvmod child whose lemma is in a
closed temporal/duration/measure noun set → (subj, verb, npadvmod).
Role="time".  Verb's complement structure is *not* otherwise
restricted (the rule emits an additional time-triplet alongside
whatever core/prep_object rule fires).

Closed lexical set protects precision: only canonical time/duration/
measure heads (today/yesterday/year/week/Monday/place/places/year/...)
trigger.  Gated under refinement relaxed bar (ΔS≥0.0005 ∧ ΔP≥−0.0075).
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
