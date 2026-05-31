"""Subject inheritance for coordinated verbs.

When the parse says ``"Alice signed the bill and announced the change"``,
``announced`` is a ``conj`` of ``signed`` and shares the subject ``Alice``.
The default :func:`segment_into_clauses` will see no direct ``nsubj`` child
on ``announced`` and emit a subject-less clause. This module patches that
post-segmentation by walking up the ``conj``/``xcomp`` chain to inherit the
nearest explicit subject.
"""

from __future__ import annotations

from typing import List

from ..models import Clause


def distribute_subjects(clauses: List[Clause]) -> List[Clause]:
    """Return a new list of clauses with inherited subjects filled in.

    Two passes:

    1. Coordination/control inheritance — clauses without any subject pick up
       the nearest explicit subject by walking up the conj/xcomp chain.
    2. Relative-clause antecedent binding — for clauses whose root is a relcl
       of a noun and whose existing subject is a wh-pronoun (who/which/that),
       the antecedent noun is added as an additional subject candidate so
       downstream rules can emit (antecedent, verb, …) alongside the
       wh-pronoun version.
    """
    by_root_idx = {c.root.i: c for c in clauses}
    patched: List[Clause] = []
    for clause in clauses:
        subs = clause.subject_candidates
        if not subs:
            inherited = _inherit_subject(clause, by_root_idx)
            if inherited:
                subs = inherited

        # Relcl antecedent binding: add the modified noun as a subject when
        # the relcl root has a wh-word in subject position OR no explicit
        # nsubj at all (subject-gap relcl).
        if _is_relcl_root(clause.root):
            antecedent = clause.root.head
            if antecedent.pos_ in {"NOUN", "PROPN", "PRON"}:
                has_wh_subj = any(
                    s.lower_ in {"who", "which", "that", "whom", "whose"}
                    for s in subs
                )
                if has_wh_subj or not subs:
                    subs = tuple(list(subs) + [antecedent])

        # Participial acl binding: the antecedent fills the missing
        # grammatical slot. VBG → active subject ("the man building the
        # house"); VBN → passive subject / patient ("the bill signed by
        # Alice"); for VBN we add the antecedent as a *candidate* even if
        # other subjects exist, so downstream rules can also emit
        # (antecedent, V-ed by, agent) alongside any wh-version.
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
    # Object-control xcomp: "She asked him to leave" — leave is xcomp of
    # asked and asked has a dobj 'him'. The dobj of the matrix verb is the
    # implicit subject of the xcomp infinitive ("him leaves"). Take it in
    # preference to the matrix nsubj.
    if walker.dep_ == "xcomp":
        head = walker.head
        if head is not None and head.i != walker.i:
            dobjs = tuple(c for c in head.children if c.dep_ == "dobj")
            if dobjs:
                return dobjs
    # advcl inherits the matrix subject only when the advcl head is a
    # participle (VBG/VBN/VB without explicit nsubj) — i.e., a reduced
    # clause whose subject is controlled by the matrix. Tensed advcls
    # like "if the ice is thick" have their own subject already.
    inheritable = {"conj", "xcomp"}
    if walker.dep_ == "advcl" and walker.tag_ in {"VBG", "VBN", "VB"}:
        inheritable = {"conj", "xcomp", "advcl"}
    while walker is not None and walker.dep_ in inheritable:
        head = walker.head
        # spaCy returns the token itself when there is no head; guard against
        # the resulting infinite loop.
        if head is None or head.i == walker.i or head.i in visited:
            break
        visited.add(head.i)
        cand = by_root_idx.get(head.i)
        if cand and cand.subject_candidates:
            return cand.subject_candidates
        walker = head
    return ()
