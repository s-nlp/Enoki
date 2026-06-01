"""N39 — appos_identity with maximal-subtree appositive arg.

Companion to ``appos_identity``: same firing conditions (X, NP, …)
appositive, NP is a full noun phrase, head + appos both NOUN/PROPN,
comma-separated. But emits the **full appositive subtree** (head +
all amod/compound/det/poss/nummod + prep PPs + acl modifiers + nested
of-PPs), trimmed at the comma. That covers EnokiQA's maximal-arg gold
for appositives:

    "The parotid gland, the largest salivary gland in the human body,
     produces saliva."
    -> (parotid gland, is, largest salivary gland in the human body)

The base appos_identity rule emits (parotid gland, is, largest
salivary gland) — head+amods only. This rule adds the full-subtree
variant; both survive dedup because the arg spans differ.
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule
from .appos_identity import (
    _has_comma_between,
    _is_in_clause_root_subtree,
    _looks_like_full_np,
)


class ApposIdentityMaximal(Rule):
    NAME = "appos_identity_maximal"
    PRIORITY = 84
    TARGETS = (
        "Maximal-subtree variant of appos_identity. Same X-comma-NP "
        "guards; arg is the full appositive subtree (head + amods + "
        "compounds + prep PPs + acl), trimmed at comma. Fills "
        "EnokiQA's maximal-arg gold for appositives like 'the largest "
        "salivary gland in the human body'."
    )
    EXAMPLES = [
        ("The parotid gland, the largest salivary gland in the human body, "
         "produces saliva.",
         [("parotid gland", "is", "largest salivary gland in the human body")]),
        ("Marie Curie, a physicist from Poland, won the Nobel Prize.",
         [("Marie Curie", "is", "physicist from Poland")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        for token in clause.span:
            if token.pos_ not in {"NOUN", "PROPN"}:
                continue
            if token.dep_ == "appos":
                continue
            if not _is_in_clause_root_subtree(token, clause):
                continue
            for child in token.children:
                if child.dep_ != "appos":
                    continue
                if child.pos_ not in {"NOUN", "PROPN"}:
                    continue
                if not _has_comma_between(token, child):
                    continue
                if not _looks_like_full_np(child):
                    continue
                yield Candidate(
                    subject_head=token,
                    predicate_head=child,
                    arg_head=child,
                    role="object",
                    prep=None,
                    source_rule=self.NAME,
                    synthesized_predicate_text="is",
                    arg_span_subtree=True,
                )
