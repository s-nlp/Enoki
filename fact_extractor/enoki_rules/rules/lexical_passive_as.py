"""L5 — lexical passive perception/designation with `as`-complement.

"X is described/known/regarded/seen/viewed/defined as Y"
    -> (X, <verb> as, Y)

This recovers a coverage gap the precision phase *created*: Q2 tightened
``svo_passive`` to require an agent/dobj (so a bare nsubjpass passive
with only an ``as``-PP is skipped) and Q4 made ``prep_object`` skip the
preposition ``as`` (FP-dominated in the generic case).  But the FN
bucketization shows ``X is described as Y`` is a GAP_addr cluster —
LSOIE+OpenIE4 gold *does* credit it (subj=X, pred=<verb>, arg="as Y").

Scope is a closed set of perception/designation verbs in the passive,
which keeps precision high (unlike a generic ``as``-PP rule).  Root is a
passive VBN (auxpass + nsubjpass) whose lemma is in the set, with a
``prep`` child ``as`` carrying a ``pobj``/``pcomp``.  Gated under the
post-precision-phase refinement relaxed bar (ΔS≥0.0005 ∧ ΔP≥−0.0075).
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from .base import Rule

_AS_VERBS = frozenset({
    "describe", "know", "regard", "see", "view", "define",
    "characterize", "classify", "treat", "depict", "portray",
    "cite", "list", "recognize", "perceive", "identify", "label",
    "categorize", "brand", "interpret", "render",
})


class LexicalPassiveAs(Rule):
    NAME = "lexical_passive_as"
    PRIORITY = 55
    TARGETS = (
        "Lexical passive perception/designation with as-complement: "
        "passive VBN (auxpass+nsubjpass) whose lemma is a "
        "perception/designation verb + prep 'as' + pobj/pcomp. "
        "'Webster is described as a boyfriend.' "
        "-> (Webster, described as, boyfriend)."
    )
    EXAMPLES = [
        ("He was described as a hero.",
         [("He", "described as", "hero")]),
        ("The site is known as Silicon Valley.",
         [("site", "known as", "Silicon Valley")]),
        ("She is regarded as an expert.",
         [("She", "regarded as", "expert")]),
        ("The film was seen as a masterpiece.",
         [("film", "seen as", "masterpiece")]),
        ("The treaty was defined as a milestone.",
         [("treaty", "defined as", "milestone")]),
        ("The region is viewed as a hub.",
         [("region", "viewed as", "hub")]),
        ("The painting was depicted as a forgery.",
         [("painting", "depicted as", "forgery")]),
        ("The compound is classified as a toxin.",
         [("compound", "classified as", "toxin")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if verb.pos_ != "VERB" or verb.tag_ != "VBN":
            return
        if verb.lemma_ not in _AS_VERBS:
            return
        if not any(c.dep_ == "auxpass" for c in verb.children):
            return
        subj = next(
            (c for c in verb.children if c.dep_ == "nsubjpass"),
            None,
        )
        if subj is None:
            return
        if subj.tag_ in {"WDT", "WP", "WP$"} or subj.lower_ in {
            "who", "which", "that", "whom", "whose"
        }:
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
        yield Candidate(
            subject_head=subj,
            predicate_head=verb,
            arg_head=obj,
            role="object",
            prep="as",
            source_rule=self.NAME,
        )
