"""Refinement R10 — post-nominal passive participial modifier.

A noun N heads a reduced/full passive participial relative clause:
the segmenter gives the participle its own Clause with N as the
inherited subject.  We fire when the clause root is an acl/relcl
past participle (VBN) and the argument is the by-agent pobj (not
already covered by prep_object).

Only the agent case is emitted here.  The plain-prep case is already
covered by prep_object (which fires on the same clause root).

Examples:
  "Documents signed by the CEO were filed."
      -> (Documents, signed, CEO)
  "A book written by Orwell sold millions."
      -> (book, written, Orwell)
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


class AclPassiveParticiple(Rule):
    NAME = "acl_passive_participle"
    PRIORITY = 15
    TARGETS = (
        "Refinement R10 recall: acl/relcl past-participle clause (VBN) "
        "whose head noun is its subject (inherited) and whose by-agent "
        "child provides the argument. "
        "Emits (head_noun, participle, agent_pobj). "
        "'Documents signed by the CEO were filed' -> (Documents, signed, CEO)."
    )
    EXAMPLES = [
        ("Documents signed by the CEO were filed.",
         [("Documents", "signed", "CEO")]),
        ("A book written by Orwell sold millions.",
         [("book", "written", "Orwell")]),
        ("Funds raised by the charity helped many.",
         [("Funds", "raised", "charity")]),
        ("A law passed by Congress changed everything.",
         [("law", "passed", "Congress")]),
        ("The novel written by Hemingway won the prize.",
         [("novel", "written", "Hemingway")]),
        ("Policies drafted by the committee were approved.",
         [("Policies", "drafted", "committee")]),
        ("An essay written by the student earned an award.",
         [("essay", "written", "student")]),
        ("The film directed by Spielberg won awards.",
         [("film", "directed", "Spielberg")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        # Must be an acl or relcl past participle
        if verb.dep_ not in {"acl", "relcl"}:
            return
        if verb.tag_ != "VBN":
            return
        if not clause.subject_candidates:
            return

        children = list(verb.children)

        # Look for by-agent: dep_=="agent" child with pobj grandchild
        agent_prep = next((c for c in children if c.dep_ == "agent"), None)
        if agent_prep is None:
            return
        pobj = next((gc for gc in agent_prep.children if gc.dep_ == "pobj"), None)
        if pobj is None:
            return

        for subj in clause.subject_candidates:
            yield Candidate(
                subject_head=subj,
                predicate_head=verb,
                arg_head=pobj,
                role="object",
                prep=None,
                source_rule=self.NAME,
            )
