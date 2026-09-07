"""Light-verb constructions: verb + object noun + preposition as one predicate.

Idiomatic ``V (det) N prep`` phrases from a closed set of
``(verb, noun, prep)`` lemma tuples are emitted as a single composite
predicate with the preposition's object as the argument:
"The reform paved the way for new policies." ->
(reform, paved the way for, new policies). The predicate text is
synthesised from the actual token sequence and carried in
``predicate_text``.
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


# (verb_lemma, dobj_lemma, prep_lemma)
_LVC_PATTERNS = frozenset({
    ("play", "role", "in"),
    ("set", "stage", "for"),
    ("set", "precedent", "for"),
    ("lay", "groundwork", "for"),
    ("lay", "foundation", "for"),
    ("pave", "way", "for"),
    ("pave", "way", "to"),
    ("bridge", "gap", "between"),
    ("bridge", "gap", "in"),
    ("have", "impact", "on"),
    ("make", "impact", "on"),
    ("have", "influence", "on"),
    ("make", "contribution", "to"),
    ("have", "effect", "on"),
    ("place", "emphasis", "on"),
    ("put", "emphasis", "on"),
    ("place", "focus", "on"),
    ("draw", "comparison", "between"),
    ("draw", "comparison", "with"),
    ("make", "comparison", "between"),
    ("gain", "recognition", "for"),
    ("receive", "recognition", "for"),
    ("take", "place", "in"),
    ("find", "place", "in"),
})


class LightVerbConstruction(Rule):
    NAME = "light_verb_construction"
    PRIORITY = 27
    TARGETS = (
        "Light-verb construction: closed (verb, dobj, prep) tuples like "
        "(play, role, in), (set, stage, for), (lay, groundwork, for), "
        "(pave, way, for/to), (bridge, gap, between/in). Composite "
        "predicate 'V (the/a) DOBJ PREP'; arg = pobj of prep."
    )
    EXAMPLES = [
        ("The reform paved the way for new policies.",
         [("reform", "paved the way for", "new policies")]),
        ("Her research played a crucial role in the discovery.",
         [("research", "played a role in", "discovery")]),
        ("The talks set the stage for negotiations.",
         [("talks", "set the stage for", "negotiations")]),
        ("Their efforts laid the foundation for modern science.",
         [("efforts", "laid the foundation for", "modern science")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if verb.pos_ != "VERB":
            return
        if not clause.subject_candidates:
            return
        verb_lemma = verb.lemma_.lower()
        # The prep may attach to the object noun or to the verb after it.
        for dobj in verb.children:
            if dobj.dep_ != "dobj":
                continue
            if dobj.pos_ not in {"NOUN", "PROPN"}:
                continue
            dobj_lemma = dobj.lemma_.lower()
            prep_candidates = [c for c in dobj.children if c.dep_ == "prep"]
            prep_candidates += [
                c for c in verb.children
                if c.dep_ == "prep" and c.i > dobj.i
            ]
            for prep_tok in prep_candidates:
                prep_lower = prep_tok.lower_
                if (verb_lemma, dobj_lemma, prep_lower) not in _LVC_PATTERNS:
                    continue
                pobj = next(
                    (g for g in prep_tok.children if g.dep_ == "pobj"),
                    None,
                )
                if pobj is None:
                    continue
                indices = sorted(
                    [verb.i, dobj.i, prep_tok.i]
                    + [c.i for c in dobj.children
                       if c.dep_ in {"det", "amod", "compound", "nummod"}]
                )
                pred_text = verb.doc[indices[0]:indices[-1] + 1].text
                for subj in clause.subject_candidates:
                    if subj.tag_ in {"WDT", "WP", "WP$"} or subj.lower_ in {
                        "who", "which", "that", "whom", "whose"
                    }:
                        continue
                    yield Candidate(
                        subject_head=subj,
                        predicate_head=verb,
                        arg_head=pobj,
                        role="object",
                        prep=prep_lower,
                        source_rule=self.NAME,
                        synthesized_predicate_text=pred_text,
                    )
