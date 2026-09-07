"""L2 — prepositional object argument.

A VERB root with a prepositional phrase child: a ``prep`` dependent whose
own child has ``dep_=="pobj"``.  Emits (subject, verb, pobj) with the
preposition absorbed into the predicate span (via ``prep=``).

Passive-agent ``by``-phrases are skipped (claimed by passive rules).

Additional (pcomp widening): when a prep child has no ``pobj`` but has a
``pcomp`` child (gerund-clause object, e.g. "resulted in anyone being
convicted"), the ``pcomp`` token is used as the argument with
``arg_span_subtree=True`` so the whole gerund clause becomes the span.
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule

# Q4 precision: prepositions that head comparative/concessive/causal/
# manner ADJUNCT phrases, never a core verb oblique argument.  On dev
# each is heavily FP-dominated as a prep_object argument:
#   as P=0.39 (77 FP)  like P=0.29  without P=0.00  despite P=0.00
#   because P=0.38  due P=0.25  (unlike/amid/versus: same family).
# Argument-bearing preps (in/on/at/to/with/for/from/into/of/...) are
# deliberately NOT listed — they carry real obliques.
_ADJUNCT_PREPS = frozenset({
    "as", "like", "unlike", "without", "despite", "because", "due",
    "amid", "amidst", "versus", "vs", "notwithstanding",
})


class PrepObject(Rule):
    NAME = "prep_object"
    PRIORITY = 30
    TARGETS = (
        "L2 prepositional object: root VERB + nsubj + prep child whose pobj "
        "child is the argument.  Passive-agent 'by' skipped; adjunct/"
        "subordinating preps (as/like/without/despite/because/due/...) "
        "skipped.  'She lives in Paris' -> (She, lives in, Paris)."
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
        # Q4: adjunct/subordinating preps skipped (no triplet)
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
            # Q4 precision: skip adjunct/subordinating prepositions —
            # comparative/concessive/causal/manner PPs are not verb
            # arguments (LSOIE+OpenIE4 gold does not credit them).
            if prep_tok.lower_ in _ADJUNCT_PREPS:
                continue
            pobj = next(
                (g for g in prep_tok.children if g.dep_ == "pobj"),
                None,
            )
            if pobj is not None:
                # Skip passive-agent 'by': that's the territory of passive rules.
                if prep_tok.lower_ == "by" and any(
                    c.dep_ == "nsubjpass" for c in verb.children
                ):
                    continue
                # Skip relative pronouns as pobj — these are spurious emissions
                # from relative clause context (e.g. "talks to whom", "refers to
                # which", "given to who"). Mirrors the existing subject-side guard.
                if pobj.tag_ in {"WDT", "WP", "WP$"} or pobj.lower_ in {
                    "who", "which", "that", "whom", "where"
                }:
                    continue
                for subj in clause.subject_candidates:
                    # Skip bare relative pronouns as subjects — these are
                    # spurious emissions from relcl context (e.g. "who", "which",
                    # "that" as nsubj inside a relative clause).
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
                # pcomp widening: gerund-clause object when pobj is absent.
                # e.g. "resulted in anyone being convicted",
                #      "disappeared after leaving the bar"
                pcomp = next(
                    (g for g in prep_tok.children if g.dep_ == "pcomp"),
                    None,
                )
                if pcomp is None:
                    continue
                # Skip passive-agent 'by' (pcomp case, defensive)
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
