"""N38 — "X, including Y" / "X, such as Y" participial-prep inclusion.

spaCy parses present-participle ``including`` (and ``involving``,
``featuring``) as dep=prep modifying a noun head, with the included
items as pobj children. We surface these as inclusion facts:

    "The countries, including France and Spain, signed the treaty."
    -> (countries, includes, France)
    -> (countries, includes, Spain)

    "Tools featuring lasers"
    -> (Tools, featuring, lasers)

Companion to the bootstrap rules — they miss this because the parse
attaches the participle as a *prep on the noun*, not as a verb root
or clause. The participle's lemma is lifted to predicate head; the
predicate surface ends up as the surface form ("including", "featuring",
"involving"), which the 0.5-overlap matcher tolerates against gold
"includes"/"include" / "involves" / "features".
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


# Participles that double as inclusion-prep modifiers in spaCy parses.
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
        # Walk every token in the clause's span — the participial prep
        # may live anywhere on the dependency tree.
        for tok in clause.span:
            if tok.lower_ not in _INCLUSION_PARTICIPLES:
                continue
            if tok.dep_ != "prep":
                continue
            # The head is the noun being augmented.
            head = tok.head
            if head is None:
                continue
            if head.pos_ not in {"NOUN", "PROPN", "PRON"}:
                continue
            for child in tok.children:
                if child.dep_ != "pobj":
                    continue
                # Emit pobj plus its conjuncts ("France and Spain" →
                # two facts).
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
