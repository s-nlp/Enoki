"""Adverbial clause argument: "She volunteered [to help the community]".

A VERB root with an ``advcl`` child emits (subject, verb, embedded clause)
with ``arg_span_subtree=True``. Subordinate clauses that are context rather
than arguments (conditional, temporal, concessive, causal, comparative ``as``,
absolutive ``with``), bare gerund clauses, and bare reduced past participles
misattached as ``advcl`` are skipped. Purpose to-infinitivals are kept.
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule

_SUBORDINATORS = frozenset({
    "if", "when", "while", "although", "because", "though", "since",
    "after", "before", "until", "unless", "whereas",
    "as", "whilst", "with",
})
_RELATIVIZERS_LOWER = frozenset({"who", "which", "that"})
_RELATIVIZER_TAGS = frozenset({"WDT", "WP", "WP$"})


def _is_contextual_advcl(ac) -> bool:
    """Return True if ``ac`` is a contextual/subordinate clause to be skipped."""
    subtree_tokens = list(ac.subtree)
    if not subtree_tokens:
        return False
    first = min(subtree_tokens, key=lambda t: t.i)

    if first.lower_ in _SUBORDINATORS:
        return True

    if first.tag_ in _RELATIVIZER_TAGS or first.lower_ in _RELATIVIZERS_LOWER:
        return True

    if any(
        c.dep_ == "mark" and c.lower_ in _SUBORDINATORS
        for c in ac.children
    ):
        return True

    # Bare gerund clause with no subject of its own.
    if ac.tag_ == "VBG" and not any(c.dep_ == "nsubj" for c in ac.children):
        return True

    # Reduced participial clause ("the film directed by X") that the parser
    # attached as advcl; never an argument of the matrix verb.
    if ac.tag_ == "VBN" and not any(
        c.dep_ in {"mark", "nsubj", "nsubjpass"} for c in ac.children
    ):
        return True

    return False


class AdvclClause(Rule):
    NAME = "advcl_clause"
    PRIORITY = 50
    TARGETS = (
        "Adverbial clause: root VERB + nsubj + advcl child. "
        "Emits (subject, verb, advcl-clause) with arg_span_subtree=True. "
        "Skips contextual/subordinate clauses (if/when/while/because/as/"
        "whilst/with/... mark, relativizer, bare VBG gerund, bare VBN "
        "reduced participle). Keeps purpose to-infinitival advcls. "
        "'She volunteered to help the community' -> (She, volunteered, to help the community)."
    )
    EXAMPLES = [
        ("She volunteered to help the community.",
         [("She", "volunteered", "to help the community")]),
        ("He smiled as she entered the room.",
         []),
        ("She worked whilst he slept.",
         []),
        ("She smiled when she saw him.",
         []),
        ("He left because it was late.",
         []),
        ("They stayed home although it was sunny.",
         []),
        ("They moved abroad after the company closed.",
         []),
        ("He would go if she asked.",
         []),
        ("She called before she arrived.",
         []),
        ("He worked while she slept.",
         []),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if verb.pos_ != "VERB":
            return
        if not clause.subject_candidates:
            return
        ac = next(
            (c for c in verb.children if c.dep_ == "advcl"),
            None,
        )
        if ac is None:
            return
        if _is_contextual_advcl(ac):
            return
        for subj in clause.subject_candidates:
            yield Candidate(
                subject_head=subj,
                predicate_head=verb,
                arg_head=ac,
                role="other",
                prep=None,
                source_rule=self.NAME,
                arg_span_subtree=True,
            )
