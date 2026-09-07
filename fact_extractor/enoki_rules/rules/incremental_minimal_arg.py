"""L5 — incremental minimal-arg variants for the canonical SVO patterns.

EnokiQA gold credits the same (subject, predicate) at multiple object-
span widths — minimal (head-only), medium (head + det/amod/compound),
and maximal (head + all modifiers/PPs).  Existing rules (core_svo /
copula_be / prep_object / core_attr / core_oprd) emit the medium-NP
granularity via shape_argument.  This rule emits the *minimal* head-
only granularity, surviving dedup alongside the medium variant because
the dedup key uses the full lemmatized arg span text (so "implication"
and "ethical implication" are distinct keys).

Patterns covered (fires whenever the named source rule would fire):
  - core_svo:     root VERB + nsubj + dobj/acomp        → minimal dobj
  - copula_be:    root be + nsubj + attr/acomp          → minimal attr/acomp
  - core_attr:    root VERB!=be + nsubj + attr          → minimal attr
  - core_oprd:    root VERB + nsubj + oprd              → minimal oprd
  - prep_object:  root VERB + nsubj + prep + pobj       → minimal pobj

Predicate / subject / role / prep are unchanged from the source rule
— only the argument span granularity differs.  Targets the ~38%
multi_granularity FN bucket on EnokiQA val.
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


# Closed adjunct-prep set: same exclusions as prep_object (Q4) so this
# rule never emits where prep_object deliberately skips.
_ADJUNCT_PREPS = frozenset({
    "as", "like", "unlike", "without", "despite", "because", "due",
    "amid", "amidst", "versus", "vs", "notwithstanding",
})


class IncrementalMinimalArg(Rule):
    NAME = "incremental_minimal_arg"
    PRIORITY = 9  # one notch below core_svo (10) — pure additive
    TARGETS = (
        "Head-only minimal-granularity arg variants for the canonical "
        "SVO patterns (core_svo / copula_be / core_attr / core_oprd / "
        "prep_object).  Same (s, p) as the source rule; the arg span is "
        "the head token alone.  Survives dedup via the full-span lemma "
        "key so it complements the medium-NP emission to match EnokiQA's "
        "multi-granularity gold."
    )
    EXAMPLES = [
        # Each example shows the head-only arg only; the source rule
        # emits the medium-NP variant via its own EXAMPLES.
        ("Alice signed the contract.",
         [("Alice", "signed", "contract")]),
        ("She wrote a novel.",
         [("She", "wrote", "novel")]),
        ("Paris is the capital of France.",
         [("Paris", "is", "capital")]),
        ("They elected her president.",
         [("They", "elected", "president")]),
        ("She lives in Paris.",
         [("She", "lives in", "Paris")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if not clause.subject_candidates:
            return
        is_be_root = verb.lemma_ == "be"

        # --- copula_be / core_attr / core_oprd: attr / acomp / oprd ---
        for c in verb.children:
            if c.dep_ in {"attr", "acomp", "oprd"}:
                # copula_be: root must be 'be' AND child attr/acomp.
                # core_attr: root VERB lemma!=be AND child attr.
                # core_oprd: root VERB AND child oprd.
                if c.dep_ in {"attr", "acomp"}:
                    if is_be_root:
                        pass  # copula_be territory
                    elif verb.pos_ == "VERB" and c.dep_ == "attr":
                        pass  # core_attr territory (non-be VERB)
                    else:
                        continue
                elif c.dep_ == "oprd":
                    if verb.pos_ != "VERB":
                        continue
                for subj in clause.subject_candidates:
                    if _bad_subj(subj):
                        continue
                    yield Candidate(
                        subject_head=subj,
                        predicate_head=verb,
                        arg_head=c,
                        role="object",
                        prep=None,
                        source_rule=self.NAME,
                        arg_minimal_only=True,
                    )

        # --- core_svo: dobj/acomp on VERB root (acomp already done above) ---
        if verb.pos_ == "VERB":
            for c in verb.children:
                if c.dep_ == "dobj":
                    for subj in clause.subject_candidates:
                        if _bad_subj(subj):
                            continue
                        yield Candidate(
                            subject_head=subj,
                            predicate_head=verb,
                            arg_head=c,
                            role="object",
                            prep=None,
                            source_rule=self.NAME,
                            arg_minimal_only=True,
                        )

        # --- prep_object: VERB root + prep child + pobj (Q4 adjunct skip) ---
        if verb.pos_ == "VERB":
            for prep_tok in verb.children:
                if prep_tok.dep_ != "prep":
                    continue
                if prep_tok.lower_ in _ADJUNCT_PREPS:
                    continue
                # Passive-agent 'by': defer to passive rules.
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
                # Skip relativizer pobj (mirrors prep_object).
                if pobj.tag_ in {"WDT", "WP", "WP$"} or pobj.lower_ in {
                    "who", "which", "that", "whom", "where"
                }:
                    continue
                for subj in clause.subject_candidates:
                    if _bad_subj(subj):
                        continue
                    yield Candidate(
                        subject_head=subj,
                        predicate_head=verb,
                        arg_head=pobj,
                        role="other",
                        prep=prep_tok.lower_,
                        source_rule=self.NAME,
                        arg_minimal_only=True,
                    )


def _bad_subj(subj) -> bool:
    return subj.tag_ in {"WDT", "WP", "WP$"} or subj.lower_ in {
        "who", "which", "that", "whom", "whose"
    }
