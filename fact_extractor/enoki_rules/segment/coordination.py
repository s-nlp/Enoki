"""Subject inheritance for clauses that have no explicit subject of their own.

In ``"Alice signed the bill and announced the change"`` the clause rooted at
``announced`` has no ``nsubj`` child; it inherits ``Alice`` by walking up the
``conj``/``xcomp`` chain. Relative and participial clauses additionally
receive their antecedent noun as a subject candidate.
"""

from __future__ import annotations

from typing import List

from ..models import Clause


def distribute_subjects(clauses: List[Clause]) -> List[Clause]:
    """Return the clauses with inherited and antecedent subjects filled in."""
    by_root_idx = {c.root.i: c for c in clauses}
    patched: List[Clause] = []
    for clause in clauses:
        subs = clause.subject_candidates
        if not subs:
            inherited = _inherit_subject(clause, by_root_idx)
            if inherited:
                subs = inherited

        # Relative clause: add the antecedent when the subject is a wh-word or
        # missing (subject-gap relative).
        if _is_relcl_root(clause.root):
            antecedent = clause.root.head
            if antecedent.pos_ in {"NOUN", "PROPN", "PRON"}:
                has_wh_subj = any(
                    s.lower_ in {"who", "which", "that", "whom", "whose"}
                    for s in subs
                )
                if has_wh_subj or not subs:
                    subs = tuple(list(subs) + [antecedent])

        # Participial clause: the antecedent is the subject of a VBG
        # ("the man building the house") and the patient of a VBN
        # ("the bill signed by Alice").
        if clause.root.dep_ == "acl":
            antecedent = clause.root.head
            if antecedent.pos_ in {"NOUN", "PROPN", "PRON"}:
                if clause.root.tag_ == "VBG" and not subs:
                    subs = (antecedent,)
                elif clause.root.tag_ == "VBN":
                    if antecedent.i not in {s.i for s in subs}:
                        subs = tuple(list(subs) + [antecedent])

        patched.append(
            Clause(
                span=clause.span,
                root=clause.root,
                subject_candidates=subs,
            )
        )
    return patched


def _is_relcl_root(token) -> bool:
    return token.dep_ == "relcl"


def _inherit_subject(clause: Clause, by_root_idx):
    walker = clause.root
    visited = set()
    # Object control: in "She asked him to leave" the matrix dobj is the
    # subject of the xcomp.
    if walker.dep_ == "xcomp":
        head = walker.head
        if head is not None and head.i != walker.i:
            dobjs = tuple(c for c in head.children if c.dep_ == "dobj")
            if dobjs:
                return dobjs
    # Only reduced (participial or infinitival) advcls inherit the matrix
    # subject; tensed ones carry their own.
    inheritable = {"conj", "xcomp"}
    if walker.dep_ == "advcl" and walker.tag_ in {"VBG", "VBN", "VB"}:
        inheritable = {"conj", "xcomp", "advcl"}
    while walker is not None and walker.dep_ in inheritable:
        head = walker.head
        # A token with no head is its own head; guard the loop.
        if head is None or head.i == walker.i or head.i in visited:
            break
        visited.add(head.i)
        cand = by_root_idx.get(head.i)
        if cand and cand.subject_candidates:
            return cand.subject_candidates
        walker = head
    return ()
