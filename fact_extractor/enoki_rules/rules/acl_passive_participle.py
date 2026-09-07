"""Post-nominal passive participle with a by-agent.

Fires on a clause whose root is an ``acl``/``relcl`` past participle (VBN)
with the modified noun as inherited subject, and emits the by-agent pobj as
the argument. Plain prepositional arguments of the same participle are left
to ``prep_object``.

    "Documents signed by the CEO were filed." -> (Documents, signed, CEO)
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


class AclPassiveParticiple(Rule):
    NAME = "acl_passive_participle"
    PRIORITY = 15
    TARGETS = (
        "acl/relcl past-participle clause (VBN) whose head noun is its "
        "inherited subject and whose by-agent child provides the argument. "
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
        if verb.dep_ not in {"acl", "relcl"}:
            return
        if verb.tag_ != "VBN":
            return
        if not clause.subject_candidates:
            return

        children = list(verb.children)

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
