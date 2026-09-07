"""Bare intransitive root: ``(subject, verb, None)``.

Fires when the clause root is a verb from a closed set of eventive
intransitives (exist, occur, happen, die, melt, ...) with an active
``nsubj`` and no complement or adjunct child (no dobj/attr/acomp/oprd/
ccomp/xcomp/advcl/prep/agent/dative/npadvmod). The closed verb set and
the strict no-complement guard keep the rule precise.
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule

# Verbs with no transitive reading. Generic motion verbs (come, go, return)
# are excluded because their adjunct PPs usually carry the argument.
_INTRANSITIVE_VERBS = frozenset({
    "exist", "occur", "happen", "die", "melt", "flow", "form", "erode",
    "vanish", "perish", "expire", "persist", "cease", "emerge",
    "survive", "endure", "decline", "deposit",
})

# Any complement or adjunct child means another rule owns the clause.
_BLOCK_CHILD_DEPS = frozenset({
    "dobj", "attr", "acomp", "oprd", "ccomp", "xcomp", "advcl",
    "prep", "agent", "dative", "npadvmod",
})


class IntransitiveRoot(Rule):
    NAME = "intransitive_root"
    PRIORITY = 60
    TARGETS = (
        "Bare intransitive: root VERB from a closed eventive set with an "
        "nsubj subject and no complement child -> (subj, verb, None). "
        "'The system exists.' -> (system, exists, None)."
    )
    EXAMPLES = [
        ("The system exists.", [("system", "exists", None)]),
        ("She died.", [("She", "died", None)]),
        ("The earthquake occurred.", [("earthquake", "occurred", None)]),
        ("Markets declined.", [("Markets", "declined", None)]),
        ("The accident happened.", [("accident", "happened", None)]),
        ("Snow melts.", [("Snow", "melts", None)]),
        ("Crystals form.", [("Crystals", "form", None)]),
        ("The river flows.", [("river", "flows", None)]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if verb.pos_ != "VERB":
            return
        if verb.lemma_ not in _INTRANSITIVE_VERBS:
            return
        if not clause.subject_candidates:
            return
        # Active subject only; passives belong to svo_passive.
        if not any(c.dep_ == "nsubj" for c in verb.children):
            return
        for c in verb.children:
            if c.dep_ in _BLOCK_CHILD_DEPS:
                return
        for subj in clause.subject_candidates:
            if subj.tag_ in {"WDT", "WP", "WP$"} or subj.lower_ in {
                "who", "which", "that", "whom", "whose"
            }:
                continue
            yield Candidate(
                subject_head=subj,
                predicate_head=verb,
                arg_head=None,
                role="other",
                prep=None,
                source_rule=self.NAME,
            )
