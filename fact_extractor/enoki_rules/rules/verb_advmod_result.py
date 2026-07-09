"""Refinement R11 — verb with ordinal/directional advmod result.

Captures ranking, placement, and directional-movement constructions
where the result is an adverbial modifier (advmod) that is either a
numeral or a member of a closed positional/directional set, and there
is no direct object (so core_svo does not already cover it).

Examples:
  "The team finished fifth."    -> (team, finished, fifth)
  "Brazil moved up."            -> (Brazil, moved, up)
  "The stock closed higher."    -> (stock, closed, higher)
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from .base import Rule

_RESULT_ADVMODS = frozenset({
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
    "up", "down", "out", "ahead", "higher", "lower",
})


class VerbAdvmodResult(Rule):
    NAME = "verb_advmod_result"
    PRIORITY = 15
    TARGETS = (
        "Refinement R11 recall: root VERB + nsubj + advmod whose token "
        "pos_=='NUM' or lower_ in the ordinal/directional set, "
        "and NO dobj child (core_svo would cover that). "
        "Emits (subject, verb, advmod) role='other'. "
        "'The team finished fifth' -> (team, finished, fifth)."
    )
    EXAMPLES = [
        ("The team finished fifth.",
         [("team", "finished", "fifth")]),
        ("He ranked third.",
         [("He", "ranked", "third")]),
        ("They placed second.",
         [("They", "placed", "second")]),
        ("The runner finished ahead.",
         [("runner", "finished", "ahead")]),
        ("The stock closed higher.",
         [("stock", "closed", "higher")]),
        ("She came first.",
         [("She", "came", "first")]),
        ("Brazil moved up.",
         [("Brazil", "moved", "up")]),
        ("The market closed lower.",
         [("market", "closed", "lower")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if verb.pos_ != "VERB":
            return
        if not clause.subject_candidates:
            return

        children = list(verb.children)

        # Guard: skip if there is a direct object (core_svo covers that)
        if any(c.dep_ == "dobj" for c in children):
            return

        for ch in children:
            if ch.dep_ != "advmod":
                continue
            if ch.pos_ != "NUM" and ch.lower_ not in _RESULT_ADVMODS:
                continue
            for subj in clause.subject_candidates:
                yield Candidate(
                    subject_head=subj,
                    predicate_head=verb,
                    arg_head=ch,
                    role="other",
                    prep=None,
                    source_rule=self.NAME,
                )
