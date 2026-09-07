"""L5 — coordinated dobj/pobj distribution (EnokiQA-targeted).

For each coordinated direct object or prepositional object, emit a
separate triplet so each conjunct gets its own (s, p, conj-token).

Was rejected at L3/iter1 on LSOIE (ΔS=−0.0051, precision hurt) but
EnokiQA gold credits coordinated NPs separately (multi-granularity
structure), so the trade may now favor it.

Patterns:
  - root VERB + nsubj + dobj that has ``conj`` children -> emit
    (subj, verb, conj) per conjunct
  - root VERB + nsubj + prep + pobj that has ``conj`` children ->
    emit (subj, verb prep, conj) per conjunct
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


_ADJUNCT_PREPS = frozenset({
    "as", "like", "unlike", "without", "despite", "because", "due",
    "amid", "amidst", "versus", "vs", "notwithstanding",
})


class CoordObject(Rule):
    NAME = "coord_object"
    PRIORITY = 12
    TARGETS = (
        "Coordinated dobj/pobj distribution: for each conj child of "
        "the dobj or pobj, emit a separate triplet with that conj as "
        "the arg. 'X verb Y and Z' -> (X, verb, Y), (X, verb, Z)."
    )
    EXAMPLES = [
        ("She bought apples and oranges.",
         [("She", "bought", "oranges")]),
        ("They visited Paris and London.",
         [("They", "visited", "London")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if verb.pos_ != "VERB":
            return
        if not clause.subject_candidates:
            return

        # dobj conjuncts
        dobj = next(
            (c for c in verb.children if c.dep_ == "dobj"),
            None,
        )
        if dobj is not None:
            for conj in dobj.children:
                if conj.dep_ != "conj":
                    continue
                for subj in clause.subject_candidates:
                    if _bad_subj(subj):
                        continue
                    yield Candidate(
                        subject_head=subj,
                        predicate_head=verb,
                        arg_head=conj,
                        role="object",
                        prep=None,
                        source_rule=self.NAME,
                    )

        # prep+pobj conjuncts
        for prep_tok in verb.children:
            if prep_tok.dep_ != "prep":
                continue
            if prep_tok.lower_ in _ADJUNCT_PREPS:
                continue
            if prep_tok.lower_ == "by" and any(
                c.dep_ == "nsubjpass" for c in verb.children
            ):
                continue
            pobj = next(
                (g for g in prep_tok.children if g.dep_ == "pobj"),
                None,
            )
            if pobj is None:
                continue
            for conj in pobj.children:
                if conj.dep_ != "conj":
                    continue
                for subj in clause.subject_candidates:
                    if _bad_subj(subj):
                        continue
                    yield Candidate(
                        subject_head=subj,
                        predicate_head=verb,
                        arg_head=conj,
                        role="other",
                        prep=prep_tok.lower_,
                        source_rule=self.NAME,
                    )


def _bad_subj(subj) -> bool:
    return subj.tag_ in {"WDT", "WP", "WP$"} or subj.lower_ in {
        "who", "which", "that", "whom", "whose"
    }
