"""Reduced passive perception/designation verb with an ``as``-complement.

Non-root counterpart of ``lexical_passive_as``: a participial ``acl``/``relcl``
VBN from the same closed verb set, with ``as`` + pobj/pcomp. The subject is
the modified head noun.

    "the site known as Silicon Valley" -> (site, known as, Silicon Valley)
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule

# Same closed set as lexical_passive_as.
_AS_VERBS = frozenset({
    "describe", "know", "regard", "see", "view", "define",
    "characterize", "classify", "treat", "depict", "portray",
    "cite", "list", "recognize", "perceive", "identify", "label",
    "categorize", "brand", "interpret", "render",
})


class AclPassiveAs(Rule):
    NAME = "acl_passive_as"
    PRIORITY = 55
    TARGETS = (
        "Reduced passive perception/designation verb with as-complement: "
        "acl/relcl VBN from the closed verb set + prep 'as' + pobj/pcomp; "
        "subject = modified head noun. "
        "'the site known as Silicon Valley' -> (site, known as, Silicon Valley)."
    )
    EXAMPLES = [
        ("The site known as Silicon Valley grew rapidly.",
         [("site", "known as", "Silicon Valley")]),
        ("An artist regarded as a pioneer arrived.",
         [("artist", "regarded as", "pioneer")]),
        ("A film classified as a masterpiece won awards.",
         [("film", "classified as", "masterpiece")]),
        ("A man known as the chief left.",
         [("man", "known as", "chief")]),
        ("A region described as a hub thrived.",
         [("region", "described as", "hub")]),
        ("A method labeled as a standard was adopted.",
         [("method", "labeled as", "standard")]),
        ("A book cited as a source was banned.",
         [("book", "cited as", "source")]),
        ("A person identified as a hero received praise.",
         [("person", "identified as", "hero")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        for tok in clause.span:
            if tok.pos_ != "VERB" or tok.tag_ != "VBN":
                continue
            if tok.dep_ not in {"acl", "relcl"}:
                continue
            if tok.lemma_ not in _AS_VERBS:
                continue
            # A finite relcl (auxpass/nsubjpass present) is covered by
            # lexical_passive_as at its own clause root.
            kids = {c.dep_ for c in tok.children}
            if "auxpass" in kids or "nsubjpass" in kids:
                continue
            as_prep = next(
                (c for c in tok.children
                 if c.dep_ == "prep" and c.lower_ == "as"),
                None,
            )
            if as_prep is None:
                continue
            obj = next(
                (g for g in as_prep.children
                 if g.dep_ in {"pobj", "pcomp"}),
                None,
            )
            if obj is None:
                continue
            head_noun = tok.head
            if head_noun.pos_ not in {"NOUN", "PROPN"}:
                continue
            yield Candidate(
                subject_head=head_noun,
                predicate_head=tok,
                arg_head=obj,
                role="object",
                prep="as",
                source_rule=self.NAME,
            )
