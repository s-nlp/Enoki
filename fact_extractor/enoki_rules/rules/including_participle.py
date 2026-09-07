"""Inclusion facts from participial prepositions ("X, including Y").

spaCy attaches ``including``/``involving``/``featuring``/``containing`` as a
``prep`` on the noun they modify, with the included items as ``pobj``
children. Emit ``(noun, participle, item)`` for the pobj and each of its
conjuncts: "The countries, including France and Spain, ..." ->
(countries, including, France), (countries, including, Spain).
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


_INCLUSION_PARTICIPLES = frozenset({
    "including", "involving", "featuring", "containing",
})


class IncludingParticiple(Rule):
    NAME = "including_participle"
    PRIORITY = 28
    TARGETS = (
        "Participial-prep inclusion modifier: token whose lemma is in "
        "{include, involve, feature, contain}, dep=prep, modifying a "
        "noun head, with pobj children. Emit (noun_head, participle, pobj)."
    )
    EXAMPLES = [
        ("The countries, including France and Spain, signed the treaty.",
         [("countries", "including", "France"),
          ("countries", "including", "Spain")]),
        ("The book features stories involving dragons.",
         [("book", "features", "stories"),
          ("stories", "involving", "dragons")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        for tok in clause.span:
            if tok.lower_ not in _INCLUSION_PARTICIPLES:
                continue
            if tok.dep_ != "prep":
                continue
            head = tok.head
            if head is None:
                continue
            if head.pos_ not in {"NOUN", "PROPN", "PRON"}:
                continue
            for child in tok.children:
                if child.dep_ != "pobj":
                    continue
                for arg in [child, *(c for c in child.children
                                     if c.dep_ == "conj")]:
                    if arg.tag_ in {"WDT", "WP", "WP$"} or arg.lower_ in {
                        "who", "which", "that", "whom", "where"
                    }:
                        continue
                    yield Candidate(
                        subject_head=head,
                        predicate_head=tok,
                        arg_head=arg,
                        role="object",
                        prep=None,
                        source_rule=self.NAME,
                    )
