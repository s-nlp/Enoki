"""N36 — be + NOUN attr + prep → (subj, is-attr-prep, pobj).

Companion to be_acomp_prep, but for the NOUN-headed attr case that the
ADJ-restricted sibling rejects. EnokiQA gold flattens noun-headed
composite predicates onto the matrix subject:

    "Ellesmere Port is home to businesses."
    -> (Ellesmere Port, is home to, businesses)

    "She is part of the team."
    -> (She, is part of, the team)

    "The illness was a result of poor diet."
    -> (illness, was result of, poor diet)

spaCy tags many copular complement nouns as ``attr`` (NOUN), not
``acomp`` (ADJ). The lemma set is intentionally CLOSED to nouns that
genuinely form composite predicates with a fixed preposition — this
keeps false positives down (we don't want every "X is doctor in Y" to
emit a "is doctor in" predicate).
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


# Closed (noun-lemma, prep-lemma) pairs: composite copular predicates.
# Each is a phrase the language treats as a unit ("home to ⟨place⟩",
# "part of ⟨whole⟩"). Membership requires that the prep argument is
# the semantic object of the predicate, not just an adjunct.
_NOUN_PREP_PAIRS = frozenset({
    # High-frequency in EnokiQA gold (from val triplet mining)
    ("home", "to"),         # ~57
    ("part", "of"),         # ~72
    ("testament", "to"),    # ~55
    ("symbol", "of"),       # ~18
    ("subject", "to"),      # ~17
    ("one", "of"),          # ~19  "is one of the…"
    # Composite role/identity nouns
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
