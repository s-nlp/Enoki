"""N37 — active V + as + NP → (subj, V as, NP).

Active sibling to lexical_passive_as. Closed lexical set of verbs
whose canonical complement is an `as` PP encoding role/function:

    "He served as president."   -> (He, served as, president)
    "It functions as a tool."    -> (It, functions as, tool)
    "She acted as a mediator."   -> (She, acted as, mediator)

Distinct from prep_object (which skips ``as`` to avoid generic-prep
FPs) and from svo_passive (which requires nsubjpass).
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


_ACTIVE_AS_VERBS = frozenset({
    "serve", "function", "act", "work", "operate",
    "double", "pose", "qualify", "masquerade",
    "register", "stand",   # "stands as a testament"
    "rank",                # "ranks as the best"
    "emerge",              # "emerged as a leader"
})


class LexicalActiveAs(Rule):
    NAME = "lexical_active_as"
    PRIORITY = 56
    TARGETS = (
        "Active V + as + NP (closed set: serve/function/act/work/operate/"
        "double/pose/qualify/register/stand/rank/emerge). "
        "'He served as president.' -> (He, served as, president)."
    )
    EXAMPLES = [
        ("He served as president.",
         [("He", "served as", "president")]),
        ("It functions as a key tool.",
         [("It", "functions as", "tool")]),
        ("She acted as a mediator.",
         [("She", "acted as", "mediator")]),
        ("The bridge stands as a testament to engineering.",
         [("bridge", "stands as", "testament")]),
        ("He emerged as a leader.",
         [("He", "emerged as", "leader")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if verb.pos_ != "VERB":
            return
        # Active only — auxpass means passive, owned by lexical_passive_as.
        if any(c.dep_ == "auxpass" for c in verb.children):
            return
        if verb.lemma_ not in _ACTIVE_AS_VERBS:
            return
        if not clause.subject_candidates:
            return
        as_prep = next(
            (c for c in verb.children
             if c.dep_ == "prep" and c.lower_ == "as"),
            None,
        )
        if as_prep is None:
            return
        obj = next(
            (g for g in as_prep.children if g.dep_ in {"pobj", "pcomp"}),
            None,
        )
        if obj is None:
            return
        for subj in clause.subject_candidates:
            if subj.tag_ in {"WDT", "WP", "WP$"} or subj.lower_ in {
                "who", "which", "that", "whom", "whose"
            }:
                continue
            yield Candidate(
                subject_head=subj,
                predicate_head=verb,
                arg_head=obj,
                role="object",
                prep="as",
                source_rule=self.NAME,
            )
