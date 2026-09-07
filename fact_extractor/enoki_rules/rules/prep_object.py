"""Prepositional object argument: ``(subject, verb prep, pobj)``.

A verb root with a ``prep`` child whose ``pobj`` is the argument; the
preposition is absorbed into the predicate. Passive-agent ``by`` phrases
are left to the passive rules, and adjunct prepositions (as, like,
without, despite, because, ...) are skipped. When a ``prep`` carries a
``pcomp`` gerund clause instead of a ``pobj`` ("resulted in anyone being
convicted"), the clause is emitted as a subtree argument.
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule

# Prepositions that head adjunct phrases rather than verb arguments.
_ADJUNCT_PREPS = frozenset({
    "as", "like", "unlike", "without", "despite", "because", "due",
    "amid", "amidst", "versus", "vs", "notwithstanding",
})


class PrepObject(Rule):
    NAME = "prep_object"
    PRIORITY = 30
    TARGETS = (
        "Prepositional object: root VERB + nsubj + prep child whose pobj is "
        "the argument. Passive-agent 'by' and adjunct prepositions "
        "(as/like/without/despite/because/due/...) are skipped. "
        "'She lives in Paris' -> (She, lives in, Paris)."
    )
    EXAMPLES = [
        ("She lives in Paris.", [("She", "lives in", "Paris")]),
        ("He works at Google.", [("He", "works at", "Google")]),
        ("They traveled to Spain.", [("They", "traveled to", "Spain")]),
        ("The book sits on the shelf.", [("book", "sits on", "shelf")]),
        ("She runs for office.", [("She", "runs for", "office")]),
        ("He relies on his team.", [("He", "relies on", "his team")]),
        ("The company depends on exports.", [("company", "depends on", "exports")]),
        ("She agreed with the decision.", [("She", "agreed with", "decision")]),
        ("He works as a consultant.", []),
        ("They marched despite the storm.", []),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if verb.pos_ != "VERB":
            return
        if not clause.subject_candidates:
            return
        for prep_tok in verb.children:
            if prep_tok.dep_ != "prep":
                continue
            if prep_tok.lower_ in _ADJUNCT_PREPS:
                continue
            pobj = next(
                (g for g in prep_tok.children if g.dep_ == "pobj"),
                None,
            )
            if pobj is not None:
                # Passive-agent 'by' belongs to the passive rules.
                if prep_tok.lower_ == "by" and any(
                    c.dep_ == "nsubjpass" for c in verb.children
                ):
                    continue
                # Relative pronouns as pobj are relative-clause artefacts.
                if pobj.tag_ in {"WDT", "WP", "WP$"} or pobj.lower_ in {
                    "who", "which", "that", "whom", "where"
                }:
                    continue
                for subj in clause.subject_candidates:
                    if subj.tag_ in {"WDT", "WP", "WP$"} or subj.lower_ in {
                        "who", "which", "that", "whom", "whose"
                    }:
                        continue
                    yield Candidate(
                        subject_head=subj,
                        predicate_head=verb,
                        arg_head=pobj,
                        role="other",
                        prep=prep_tok.lower_,
                        source_rule=self.NAME,
                    )
            else:
                # Gerund-clause object when there is no pobj.
                pcomp = next(
                    (g for g in prep_tok.children if g.dep_ == "pcomp"),
                    None,
                )
                if pcomp is None:
                    continue
                if prep_tok.lower_ == "by" and any(
                    c.dep_ == "nsubjpass" for c in verb.children
                ):
                    continue
                for subj in clause.subject_candidates:
                    if subj.tag_ in {"WDT", "WP", "WP$"} or subj.lower_ in {
                        "who", "which", "that", "whom", "whose"
                    }:
                        continue
                    yield Candidate(
                        subject_head=subj,
                        predicate_head=verb,
                        arg_head=pcomp,
                        role="other",
                        prep=prep_tok.lower_,
                        source_rule=self.NAME,
                        arg_span_subtree=True,
                    )
