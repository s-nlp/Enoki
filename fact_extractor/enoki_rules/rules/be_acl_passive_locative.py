"""L5 — be-predicative + locative acl participle, matrix-subject anchored.

EnokiQA-style gold extracts modifier-level locative relations using the
MATRIX subject of the be-copula, not the modified noun:

    "Dayton, Montana, is a small settlement located in Chouteau County."
    -> (Dayton, Montana, located in, Chouteau County)
      [not  (settlement,    located in, Chouteau County)
            which acl_passive_participle currently emits]

Pattern: root lemma 'be' + nsubj X + attr/acomp Y; Y has an acl/relcl
VBN child whose lemma is in a closed locative-state set (located /
situated / based / headquartered / nestled / perched / positioned /
housed / found / set); that VBN has prep child with pobj Z. Emit
(X, <acl-verb> <prep>, Z) — re-anchoring the subject from Y to X.

The closed verb set keeps precision high. Gated under the EnokiQA
refinement bar (ΔS ≥ 0.0005 on the EnokiQA val sample).
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from ..rule_base import Rule

_LOCATIVE_STATE_VERBS = frozenset({
    "located", "situated", "based", "headquartered", "perched",
    "positioned", "nestled", "housed", "set", "found",
})


class BeAclPassiveLocative(Rule):
    NAME = "be_acl_passive_locative"
    PRIORITY = 55
    TARGETS = (
        "Be-predicative + locative acl participle, matrix-subject "
        "re-anchored: 'X is a Y located in Z' -> (X, located in, Z). "
        "Closed locative-state verb set (located/situated/based/...) + "
        "acl on attr/acomp of be-root + prep + pobj."
    )
    EXAMPLES = [
        ("Dayton is a settlement located in Chouteau County.",
         [("Dayton", "located in", "Chouteau County")]),
        ("Mars is a planet situated in the solar system.",
         [("Mars", "situated in", "solar system")]),
        ("Apple is a company headquartered in Cupertino.",
         [("Apple", "headquartered in", "Cupertino")]),
        ("The chapel is a building perched on a cliff.",
         [("chapel", "perched on", "cliff")]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
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
        for acl in attr.children:
            if acl.dep_ not in {"acl", "relcl"}:
                continue
            if acl.tag_ != "VBN" or acl.lower_ not in _LOCATIVE_STATE_VERBS:
                continue
            # Skip if the acl has its own nsubjpass (finite relcl —
            # different construction, gold handles via main passive).
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
                    if subj.tag_ in {"WDT", "WP", "WP$"} or subj.lower_ in {
                        "who", "which", "that", "whom", "whose"
                    }:
                        continue
                    yield Candidate(
                        subject_head=subj,
                        predicate_head=acl,
                        arg_head=pobj,
                        role="object",
                        prep=prep_tok.lower_,
                        source_rule=self.NAME,
                    )
