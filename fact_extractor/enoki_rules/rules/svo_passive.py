"""Root passive: ``(patient, was verb, agent | dobj)``.

Fires on a verb root with both ``nsubjpass`` and ``auxpass`` children. The
argument is the ``by``-agent's object ("The bill was signed by the
president" -> (bill, was signed, president)) or, failing that, a
ditransitive ``dobj`` ("She was given a prize"). Passives with neither
are skipped, so bare and preposition-only passives are never emitted.
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


class SvoPassive(Rule):
    NAME = "svo_passive"
    PRIORITY = 15
    TARGETS = (
        "Root passive: root VERB with nsubjpass (patient) and auxpass child. "
        "Emits (patient, verb, agent|dobj) role=object; requires an agent "
        "pobj or a dobj, bare/prep-only passives are skipped. "
        "'The bill was signed by the president' -> (bill, was signed, president)."
    )
    EXAMPLES = [
        ("The bill was signed by the president.",
         [("bill", "signed", "president")]),
        ("Hamlet was written by Shakespeare.",
         [("Hamlet", "written", "Shakespeare")]),
        ("She was awarded a prize.",
         [("She", "awarded", "prize")]),
        ("The law was passed by parliament.",
         [("law", "passed", "parliament")]),
        ("The students were given homework.",
         [("students", "given", "homework")]),
        ("The suspects were questioned by police.",
         [("suspects", "questioned", "police")]),
        ("The book was translated by a professor.",
         [("book", "translated", "professor")]),
        ("The children were shown a movie.",
         [("children", "shown", "movie")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if verb.pos_ != "VERB":
            return

        children = list(verb.children)

        has_nsubjpass = any(c.dep_ == "nsubjpass" for c in children)
        has_auxpass = any(c.dep_ == "auxpass" for c in children)
        if not (has_nsubjpass and has_auxpass):
            return

        patient = next((c for c in children if c.dep_ == "nsubjpass"), None)
        if patient is None:
            return

        arg_head = None
        role = "other"

        # by-agent object
        agent_prep = next((c for c in children if c.dep_ == "agent"), None)
        if agent_prep is not None:
            pobj = next(
                (gc for gc in agent_prep.children if gc.dep_ == "pobj"),
                None,
            )
            if pobj is not None:
                arg_head = pobj
                role = "object"

        # ditransitive dobj
        if arg_head is None:
            dobj = next((c for c in children if c.dep_ == "dobj"), None)
            if dobj is not None:
                arg_head = dobj
                role = "object"

        # No agent or dobj: skip rather than fall back to a bare passive.
        if arg_head is None:
            return

        yield Candidate(
            subject_head=patient,
            predicate_head=verb,
            arg_head=arg_head,
            role=role,
            prep=None,
            source_rule=self.NAME,
        )
