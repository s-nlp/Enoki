"""Verb with an ordinal or directional ``advmod`` as its result.

Ranking, placement and movement constructions where the result is an
adverbial modifier that is a numeral or in a closed positional set, and
the verb has no direct object: "The team finished fifth." ->
(team, finished, fifth); "Brazil moved up." -> (Brazil, moved, up).
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule

_RESULT_ADVMODS = frozenset({
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
    "up", "down", "out", "ahead", "higher", "lower",
})


class VerbAdvmodResult(Rule):
    NAME = "verb_advmod_result"
    PRIORITY = 15
    TARGETS = (
        "Root VERB + nsubj + advmod that is a NUM or in the "
        "ordinal/directional set, with no dobj child. "
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

        # A direct object means core_svo covers the clause.
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
