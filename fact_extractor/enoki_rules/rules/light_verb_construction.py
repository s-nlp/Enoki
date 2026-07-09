"""N42 — light-verb construction (LVC): V + dobj-N + prep → composite pred.

EnokiQA gold treats idiomatic V+det+N+prep phrases as single composite
predicates flattened onto the matrix subject + the prep's pobj as
object:

    "The reform paved the way for new policies."
    -> (reform, paved the way for, new policies)

    "His research played a crucial role in the discovery."
    -> (research, played a crucial role in, discovery)

    "The talks set the stage for negotiations."
    -> (talks, set the stage for, negotiations)

The bootstrap pipeline emits these as (subj, V, dobj) via core_svo and
ignores the prep-PP, or emits (subj, V prep, pobj) via prep_object
dropping the dobj. Neither matches the gold composite predicate.

Closed (verb_lemma, dobj_lemma, prep_lemma) tuples — high gold
frequency in EnokiQA val, low collision risk with the bootstrap rules
(they emit narrower-predicate variants that survive dedup alongside).

Composite predicate is synthesized as a string: ``"paved the way
for"`` etc. The predicate token span is degenerate (V only), but
``predicate_text`` carries the full surface for the matcher.
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


# (verb_lemma, dobj_lemma, prep_lemma) — verified high-frequency in
# EnokiQA val triplet mining (≥10 occurrences in 10k sentences).
_LVC_PATTERNS = frozenset({
    # "play / played / plays a role in"
    ("play", "role", "in"),
    # "set a / the stage / precedent for"
    ("set", "stage", "for"),
    ("set", "precedent", "for"),
    # "lay the groundwork / foundation for"
    ("lay", "groundwork", "for"),
    ("lay", "foundation", "for"),
    # "pave the way for / to"
    ("pave", "way", "for"),
    ("pave", "way", "to"),
    # "bridge the gap between / in"
    ("bridge", "gap", "between"),
    ("bridge", "gap", "in"),
    # "have / make an impact on, influence on, contribution to"
    ("have", "impact", "on"),
    ("make", "impact", "on"),
    ("have", "influence", "on"),
    ("make", "contribution", "to"),
    ("have", "effect", "on"),
    # "take / put emphasis / focus on"
    ("place", "emphasis", "on"),
    ("put", "emphasis", "on"),
    ("place", "focus", "on"),
    # "draw / make a comparison between/to/with"
    ("draw", "comparison", "between"),
    ("draw", "comparison", "with"),
    ("make", "comparison", "between"),
    # "gain / receive recognition for/as"
    ("gain", "recognition", "for"),
    ("receive", "recognition", "for"),
    # "find / take a place in"
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
        # Find dobj children and candidate prep tokens. The prep can
        # attach to either (a) the dobj itself ("the way for X") or
        # (b) the verb as a sibling of the dobj ("set the stage for X"
        # — the more common English LVC parse).
        for dobj in verb.children:
            if dobj.dep_ != "dobj":
                continue
            if dobj.pos_ not in {"NOUN", "PROPN"}:
                continue
            dobj_lemma = dobj.lemma_.lower()
            prep_candidates = [c for c in dobj.children if c.dep_ == "prep"]
            prep_candidates += [
                c for c in verb.children
                if c.dep_ == "prep" and c.i > dobj.i  # appears after dobj
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
                # Build the composite predicate surface from the actual
                # token sequence: verb [det/amod...] dobj prep.
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
