"""L4 — adverbial clause argument (advcl).

A VERB root with an ``advcl`` child (adverbial clause modifier, typically
introduced by a subordinating conjunction: "She smiled [when she saw him]").
Emits (subject, verb, advcl-clause) with ``arg_span_subtree=True`` so the
pipeline materialises the whole embedded clause as the argument span.

Precision guard: SKIP contextual/subordinate clauses that are not true
arguments (conditional, temporal, concessive, causal, comparative-``as``,
absolutive-``with``, etc.), bare participial gerund clauses, and bare
reduced past-participle (VBN) clauses misattached as advcl.  Purpose
to-infinitival advcls (no mark, TO aux) are preserved.
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from .base import Rule

_SUBORDINATORS = frozenset({
    "if", "when", "while", "although", "because", "though", "since",
    "after", "before", "until", "unless", "whereas",
    # Q3 precision: comparative/manner ``as``, ``whilst`` (=while), and
    # absolutive ``with`` advcls are parenthetical, not verb arguments
    # (as-mark: 33 FP / 10 TP; whilst/with: ~6 FP / 0 TP on dev).
    "as", "whilst", "with",
})
_RELATIVIZERS_LOWER = frozenset({"who", "which", "that"})
_RELATIVIZER_TAGS = frozenset({"WDT", "WP", "WP$"})


def _is_contextual_advcl(ac) -> bool:
    """Return True if ``ac`` is a contextual/subordinate clause to be skipped."""
    # Find the first content token of the advcl subtree (lowest .i)
    subtree_tokens = list(ac.subtree)
    if not subtree_tokens:
        return False
    first = min(subtree_tokens, key=lambda t: t.i)

    # (i) first content token is a subordinating conjunction
    if first.lower_ in _SUBORDINATORS:
        return True

    # (ii) first token is a relativizer
    if first.tag_ in _RELATIVIZER_TAGS or first.lower_ in _RELATIVIZERS_LOWER:
        return True

    # (iii) ac has a mark child whose lower_ is in the subordinator set
    if any(
        c.dep_ == "mark" and c.lower_ in _SUBORDINATORS
        for c in ac.children
    ):
        return True

    # (iv) bare participial gerund clause (VBG head, no nsubj child)
    if ac.tag_ == "VBG" and not any(c.dep_ == "nsubj" for c in ac.children):
        return True

    # (v) bare reduced past-participle clause (VBN head, no mark and no
    # nsubj/nsubjpass child).  These are reduced relative/participial
    # clauses ("the film directed by X", "the report titled Y") that the
    # parser misattaches as advcl; they are never the matrix verb's
    # argument (25 FP / 8 TP on dev).
    if ac.tag_ == "VBN" and not any(
        c.dep_ in {"mark", "nsubj", "nsubjpass"} for c in ac.children
    ):
        return True

    return False


class AdvclClause(Rule):
    NAME = "advcl_clause"
    PRIORITY = 50
    TARGETS = (
        "L4 adverbial clause: root VERB + nsubj + advcl child. "
        "Emits (subject, verb, advcl-clause) with arg_span_subtree=True. "
        "Skips contextual/subordinate clauses (if/when/while/because/as/"
        "whilst/with/... mark, relativizer, bare VBG gerund, bare VBN "
        "reduced participle). Keeps purpose to-infinitival advcls. "
        "'She volunteered to help the community' -> (She, volunteered, to help the community)."
    )
    EXAMPLES = [
        # Purpose to-infinitival — must still fire (no mark, TO aux)
        ("She volunteered to help the community.",
         [("She", "volunteered", "to help the community")]),
        # Skipped: comparative/manner "as" (Q3)
        ("He smiled as she entered the room.",
         []),
        # Skipped: "whilst" (= while) (Q3)
        ("She worked whilst he slept.",
         []),
        # Skipped: temporal "when"
        ("She smiled when she saw him.",
         []),
        # Skipped: causal "because"
        ("He left because it was late.",
         []),
        # Skipped: concessive "although"
        ("They stayed home although it was sunny.",
         []),
        # Skipped: temporal "after"
        ("They moved abroad after the company closed.",
         []),
        # Skipped: conditional "if"
        ("He would go if she asked.",
         []),
        # Skipped: temporal "before"
        ("She called before she arrived.",
         []),
        # Skipped: mark "while"
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
        # Precision guard: skip contextual/subordinate advcl clauses
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
