"""L5 — granularity (min + max) arg variants for the lexical passive rules.

Wraps four lexical-passive rules whose existing emission is
medium-NP-only:
  - acl_passive_participle:  acl/relcl VBN + by-agent pobj
  - acl_passive_as (N5):     acl/relcl perception VBN + as + pobj
  - be_acl_passive_locative (N10): be + attr/acomp + locative-state acl
                                   + prep + pobj
  - be_acomp_prep (N11):     be + ADJ acomp + prep + pobj

For each, emit minimal (head-only pobj) + maximal (full pobj subtree)
variants — completing the granularity coverage like N14/N15 do for
the canonical SVO patterns.
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule


_AS_VERBS = frozenset({
    "describe", "know", "regard", "see", "view", "define",
    "characterize", "classify", "treat", "depict", "portray",
    "cite", "list", "recognize", "perceive", "identify", "label",
    "categorize", "brand", "interpret", "render",
})

_LOCATIVE_STATE_VERBS = frozenset({
    "located", "situated", "based", "headquartered", "perched",
    "positioned", "nestled", "housed", "set", "found",
})


def _bad_subj(t) -> bool:
    return t.tag_ in {"WDT", "WP", "WP$"} or t.lower_ in {
        "who", "which", "that", "whom", "whose"
    }


def _emit_min_max(subj_head, pred_head, arg_head, prep_lower):
    for flag in ("min", "max"):
        yield Candidate(
            subject_head=subj_head,
            predicate_head=pred_head,
            arg_head=arg_head,
            role="object",
            prep=prep_lower,
            source_rule="lexical_passive_arg_granularity",
            arg_minimal_only=(flag == "min"),
            arg_span_subtree=(flag == "max"),
        )


class LexicalPassiveArgGranularity(Rule):
    NAME = "lexical_passive_arg_granularity"
    PRIORITY = 9
    TARGETS = (
        "Granularity (min head-only + max subtree) arg variants for the "
        "lexical passive rules: acl_passive_participle, acl_passive_as, "
        "be_acl_passive_locative, be_acomp_prep."
    )
    EXAMPLES = [
        ("The site known as Silicon Valley grew rapidly.",
         [("site", "known as", "Silicon Valley")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        # -------- acl_passive_participle (acl/relcl VBN + by-agent) --------
        for tok in clause.span:
            if tok.pos_ != "VERB" or tok.tag_ != "VBN":
                continue
            if tok.dep_ not in {"acl", "relcl"}:
                continue
            head_noun = tok.head
            if head_noun.pos_ not in {"NOUN", "PROPN"}:
                continue
            # by-agent
            agent = next(
                (c for c in tok.children if c.dep_ == "agent"),
                None,
            )
            if agent is not None:
                pobj = next(
                    (g for g in agent.children if g.dep_ == "pobj"),
                    None,
                )
                if pobj is not None:
                    yield from _emit_min_max(
                        head_noun, tok, pobj, "by",
                    )
            # acl_passive_as (perception/designation verb + as prep + pobj)
            if (tok.lemma_ in _AS_VERBS
                    and not any(c.dep_ in {"auxpass", "nsubjpass"}
                                for c in tok.children)):
                as_prep = next(
                    (c for c in tok.children
                     if c.dep_ == "prep" and c.lower_ == "as"),
                    None,
                )
                if as_prep is not None:
                    pobj_as = next(
                        (g for g in as_prep.children
                         if g.dep_ in {"pobj", "pcomp"}),
                        None,
                    )
                    if pobj_as is not None:
                        yield from _emit_min_max(
                            head_noun, tok, pobj_as, "as",
                        )

        # -------- be-root: be_acl_passive_locative + be_acomp_prep --------
        verb = clause.root
        if verb.lemma_ != "be":
            return
        if not clause.subject_candidates:
            return
        attr = next(
            (c for c in verb.children if c.dep_ in {"attr", "acomp"}),
            None,
        )
        if attr is None:
            return
        # be_acomp_prep: ADJ acomp + prep + pobj
        if attr.pos_ == "ADJ" and attr.dep_ == "acomp":
            for prep_tok in attr.children:
                if prep_tok.dep_ != "prep":
                    continue
                pobj = next(
                    (g for g in prep_tok.children if g.dep_ == "pobj"),
                    None,
                )
                if pobj is None:
                    continue
                for subj in clause.subject_candidates:
                    if _bad_subj(subj):
                        continue
                    yield from _emit_min_max(subj, attr, pobj,
                                             prep_tok.lower_)
        # be_acl_passive_locative: attr/acomp's acl-VBN locative + prep + pobj
        for acl in attr.children:
            if acl.dep_ not in {"acl", "relcl"}:
                continue
            if acl.tag_ != "VBN" or acl.lower_ not in _LOCATIVE_STATE_VERBS:
                continue
            if any(c.dep_ in {"nsubj", "nsubjpass"} for c in acl.children):
                continue
            for prep_tok in acl.children:
                if prep_tok.dep_ != "prep":
                    continue
                pobj = next(
                    (g for g in prep_tok.children if g.dep_ == "pobj"),
                    None,
                )
                if pobj is None:
                    continue
                for subj in clause.subject_candidates:
                    if _bad_subj(subj):
                        continue
                    yield from _emit_min_max(subj, acl, pobj,
                                             prep_tok.lower_)
