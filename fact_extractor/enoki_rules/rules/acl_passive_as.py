"""L5 — reduced passive perception/designation with `as`-complement (acl/relcl).

Companion to ``lexical_passive_as`` (N3) extending the same closed
perception/designation verb set to *non-root* positions: reduced
relative ``acl`` / finite relative ``relcl`` participial VBN heads.

    "the site known as Silicon Valley"   -> (site, known as, Silicon Valley)
    "a substance described as toxic"     -> (substance, described as, toxic)
    "an artist regarded as a pioneer"    -> (artist, regarded as, pioneer)

The subject of the relation is the modified head noun (``acl.head`` /
``relcl.head``), not an nsubjpass — these reduced forms drop the
auxiliary and explicit subject.  Verb set is identical to N3; closed
lexical scope keeps precision high.  Gated under refinement relaxed
bar (ΔS≥0.0005 ∧ ΔP≥−0.0075).
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from .base import Rule

# Same closed set as lexical_passive_as (N3) — perception/designation
# verbs that take an ``as``-NP/AdjP complement in passive.
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
        "Reduced passive perception/designation with as-complement: "
        "acl/relcl VBN whose lemma is in the N3 perception/designation "
        "set + prep 'as' + pobj/pcomp; subject = modified head noun. "
        "'the site known as Silicon Valley' "
        "-> (site, known as, Silicon Valley). Companion to "
        "lexical_passive_as (N3) for non-root positions."
    )
    EXAMPLES = [
        # NP / PROPN pobj only — ADJ-pobj parses inconsistently.
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
            # Reduced relatives lack auxpass + nsubjpass; if these are
            # present this is a *finite* relcl which N3 would already
            # cover at its own clause root.  Keeping this branch focused
            # on the reduced form avoids double-emission.
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
