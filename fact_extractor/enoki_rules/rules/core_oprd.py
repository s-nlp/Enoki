"""L0 — object predicate complement (oprd).

Captures constructions like "They elected her president" / "The board named
him chairman" where the root VERB has an open predicate complement (`oprd`)
attached to it.  The `oprd` dependent describes what the object *becomes*
or *is considered to be*, which is the salient argument for fact extraction.
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


class CoreOprd(Rule):
    NAME = "core_oprd"
    PRIORITY = 10
    TARGETS = (
        "L0 object predicate complement: root VERB with a nominal subject "
        "and an oprd child. "
        "'They elected her president' -> (They, elected, president)."
    )
    EXAMPLES = [
        ("The board named him chairman.", [("board", "named", "chairman")]),
        ("They appointed him director.", [("They", "appointed", "director")]),
        ("They painted the house red.", [("They", "painted", "red")]),
        ("People call him a genius.", [("People", "call", "genius")]),
        ("She named him her successor.", [("She", "named", "her successor")]),
        ("They called him a fool.", [("They", "called", "fool")]),
        ("They dubbed him a hero.", [("They", "dubbed", "hero")]),
        ("She appointed him secretary.", [("She", "appointed", "secretary")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if verb.pos_ != "VERB":
            return
        if not clause.subject_candidates:
            return
        arg = next(
            (c for c in verb.children if c.dep_ == "oprd"),
            None,
        )
        if arg is None:
            return
        for subj in clause.subject_candidates:
            yield Candidate(
                subject_head=subj,
                predicate_head=verb,
                arg_head=arg,
                role="object",
                prep=None,
                source_rule=self.NAME,
            )
