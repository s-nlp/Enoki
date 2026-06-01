"""Refinement R2 — ROOT passive (nsubjpass + auxpass).

Handles sentences where the grammatical subject is the semantic patient:
"The bill was signed by the president" -> (bill, was signed, president).

The auxpass guard is the precision shield: we only fire when spaCy has
already marked an auxiliary as a passive auxiliary (dep_==auxpass),
which prevents active-voice false fires on past-participle adjectives.

Argument requirement (precision-phase Q2):
  Only emit when the passive verb has a STRONG argument:
  (a) agent by-phrase: prep(dep_==agent) -> pobj  (e.g. "by the president")
      emit (patient, verb, agent-pobj) role="object"
  (b) ditransitive dobj (e.g. "She was given a prize")
      emit (patient, verb, dobj) role="object"
  If NEITHER an agent-pobj NOR a dobj exists, SKIP entirely.
  Bare passives (no arg) and prep-only passives (e.g. "in the war") are
  dropped to reduce false positives against LSOIE+OpenIE4 gold.
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from .base import Rule


class SvoPassive(Rule):
    NAME = "svo_passive"
    PRIORITY = 15
    TARGETS = (
        "Refinement R2 ROOT passive: root VERB with nsubjpass (patient) "
        "and auxpass child. Emits (patient, verb[+auxpass], agent|dobj) role=object. "
        "Requires agent-pobj OR dobj; bare/prep-only passives are skipped. "
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

        # Precision guard: require both nsubjpass and auxpass
        has_nsubjpass = any(c.dep_ == "nsubjpass" for c in children)
        has_auxpass = any(c.dep_ == "auxpass" for c in children)
        if not (has_nsubjpass and has_auxpass):
            return

        patient = next((c for c in children if c.dep_ == "nsubjpass"), None)
        if patient is None:
            return

        # Determine argument: require a STRONG argument (agent-pobj or dobj).
        # Bare passives and prep-only passives are skipped for precision.
        arg_head = None
        role = "other"

        # (a) by-agent: dep_==agent child whose own child has dep_==pobj
        agent_prep = next((c for c in children if c.dep_ == "agent"), None)
        if agent_prep is not None:
            pobj = next(
                (gc for gc in agent_prep.children if gc.dep_ == "pobj"),
                None,
            )
            if pobj is not None:
                arg_head = pobj
                role = "object"

        # (b) ditransitive dobj (e.g. "She was given a prize")
        if arg_head is None:
            dobj = next((c for c in children if c.dep_ == "dobj"), None)
            if dobj is not None:
                arg_head = dobj
                role = "object"

        # PRECISION GUARD: if no strong argument found, skip entirely.
        # Do NOT fall back to prep-pobj-only or None (bare passive).
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
