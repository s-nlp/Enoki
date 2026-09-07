"""Copula + noun complement + preposition: "X is home to Y".

Noun-headed counterpart of ``be_acomp_prep``: root lemma ``be`` with a NOUN
``attr`` whose (lemma, preposition) pair is in a closed set of composite
predicates ("home to", "part of", "result of", ...). The closed set keeps
ordinary "X is a doctor in Y" from producing an "is doctor in" predicate.

    "She is part of the team." -> (She, part of, team)
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


# Closed (noun lemma, preposition) pairs whose prep argument is the semantic
# object of the predicate rather than an adjunct.
_NOUN_PREP_PAIRS = frozenset({
    ("home", "to"),
    ("part", "of"),
    ("testament", "to"),
    ("symbol", "of"),
    ("subject", "to"),
    ("one", "of"),
    ("member", "of"),
    ("source", "of"),
    ("cause", "of"),
    ("result", "of"),
    ("type", "of"),
    ("kind", "of"),
    ("form", "of"),
    ("sign", "of"),
    ("matter", "of"),
    ("way", "of"),
    ("means", "of"),
    ("victim", "of"),
    ("witness", "to"),
    ("friend", "of"),
    ("ally", "of"),
    ("enemy", "of"),
    ("father", "of"),
    ("mother", "of"),
    ("author", "of"),
    ("creator", "of"),
    ("founder", "of"),
    ("owner", "of"),
    ("president", "of"),
    ("leader", "of"),
    ("head", "of"),
    ("director", "of"),
    ("teacher", "of"),
    ("example", "of"),
    ("instance", "of"),
    ("case", "of"),
    ("piece", "of"),
    ("champion", "of"),
    ("pioneer", "of"),
    ("advocate", "of"),
    ("advocate", "for"),
    ("supporter", "of"),
    ("supporter", "for"),
    ("guardian", "of"),
    ("expression", "of"),
    ("manifestation", "of"),
    ("reflection", "of"),
    ("embodiment", "of"),
    ("version", "of"),
    ("variant", "of"),
    ("variation", "of"),
    ("descendant", "of"),
    ("ancestor", "of"),
    ("native", "of"),
    ("resident", "of"),
    ("citizen", "of"),
    ("winner", "of"),
    ("recipient", "of"),
    ("hero", "of"),
    ("center", "of"),
    ("hub", "of"),
    ("epitome", "of"),
})


class BeAttrNounPrep(Rule):
    NAME = "be_attr_noun_prep"
    PRIORITY = 26
    TARGETS = (
        "Copula be + NOUN attr + prep -> (subj, attr prep, pobj). "
        "Closed lexical set of noun-headed composite copular predicates "
        "('is home to', 'is part of', 'is member of') that be_acomp_prep "
        "rejects because the head is NOUN not ADJ."
    )
    EXAMPLES = [
        ("Ellesmere Port is home to businesses.",
         [("Ellesmere Port", "home to", "businesses")]),
        ("She is part of the team.",
         [("She", "part of", "team")]),
        ("The illness was a result of poor diet.",
         [("illness", "result of", "poor diet")]),
        ("Aristotle was the teacher of Alexander.",
         [("Aristotle", "teacher of", "Alexander")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if verb.lemma_ != "be":
            return
        if not clause.subject_candidates:
            return
        attr = next(
            (c for c in verb.children if c.dep_ == "attr"),
            None,
        )
        if attr is None:
            return
        if attr.pos_ != "NOUN":
            return
        attr_lemma = attr.lemma_.lower()
        for prep_tok in attr.children:
            if prep_tok.dep_ != "prep":
                continue
            prep_lower = prep_tok.lower_
            if (attr_lemma, prep_lower) not in _NOUN_PREP_PAIRS:
                continue
            pobj = next(
                (g for g in prep_tok.children if g.dep_ == "pobj"),
                None,
            )
            if pobj is None:
                continue
            for subj in clause.subject_candidates:
                if subj.tag_ in {"WDT", "WP", "WP$"} or subj.lower_ in {
                    "who", "which", "that", "whom", "whose"
                }:
                    continue
                yield Candidate(
                    subject_head=subj,
                    predicate_head=attr,
                    arg_head=pobj,
                    role="object",
                    prep=prep_lower,
                    source_rule=self.NAME,
                )
