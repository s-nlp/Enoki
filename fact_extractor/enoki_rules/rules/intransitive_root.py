"""L5 — bare intransitive root with closed eventive verb set.

Root is a VERB in a closed eventive/state-change set (rise/fall/drop/
decline/come/go/die/live/exist/occur/happen/begin/start/end/move/
return/appear/break/form/flow/melt/erode/...) with a nominal subject
and NO transitive/clausal/copular complement (no dobj/attr/acomp/oprd/
ccomp/xcomp/advcl) — emit (subject, verb, None).

Targets the dominant ``arg_missing`` FN bucket (~254 / 542 = 47%):
LSOIE+OpenIE4 gold credits these as `(subj, verb, None)` triplets.
The completeness filter explicitly allows `argument=None` for non-
copular verbs.  Prep children are permitted (``prep_object`` emits the
parallel `(subj, verb prep, pobj)` triplet) but a bare-intransitive
triplet is still gold-credited separately.

Verb set is closed and semantically homogeneous (eventive,
intransitive, state-change/motion/existence) to control precision.
Gated under refinement relaxed bar (ΔS≥0.0005 ∧ ΔP≥−0.0075).
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, Clause
from .base import Rule

_INTRANSITIVE_VERBS = frozenset({
    # Pruned to verbs that are *dead-set* intransitive in any context
    # (no transitive reading where arg gets credited).  Generic verbs
    # (come/go/leave/return/follow/stay/work/walk/run/...) were dropped
    # — they routinely take adjunct PPs gold credits as args.
    "exist", "occur", "happen", "die", "melt", "flow", "form", "erode",
    "vanish", "perish", "expire", "persist", "cease", "emerge",
    "survive", "endure", "decline", "deposit",
})

_BLOCK_CHILD_DEPS = frozenset({
    # any of these means the verb has an explicit complement; bail.
    # Crucially includes ``prep`` — adjunct PPs like "at home" are
    # frequently gold-credited as the verb's argument.
    "dobj", "attr", "acomp", "oprd", "ccomp", "xcomp", "advcl",
    "prep", "agent", "dative", "npadvmod",
})


class IntransitiveRoot(Rule):
    NAME = "intransitive_root"
    PRIORITY = 60
    TARGETS = (
        "L5 bare intransitive: root VERB in a closed eventive set with "
        "an nsubj subject and NO {dobj,attr,acomp,oprd,ccomp,xcomp,advcl} "
        "child -> (subj, verb, None). "
        "'The system exists.' -> (system, exists, None). "
        "Targets the arg=None FN bucket; closed verb set protects precision."
    )
    EXAMPLES = [
        ("The system exists.", [("system", "exists", None)]),
        ("She died.", [("She", "died", None)]),
        ("The earthquake occurred.", [("earthquake", "occurred", None)]),
        ("Markets declined.", [("Markets", "declined", None)]),
        ("The accident happened.", [("accident", "happened", None)]),
        ("Snow melts.", [("Snow", "melts", None)]),
        ("Crystals form.", [("Crystals", "form", None)]),
        ("The river flows.", [("river", "flows", None)]),
    ]

    def apply(self, clause: Clause) -> Iterable[Candidate]:
        verb = clause.root
        if verb.pos_ != "VERB":
            return
        if verb.lemma_ not in _INTRANSITIVE_VERBS:
            return
        if not clause.subject_candidates:
            return
        # Must have an active nsubj (not passive); passive intransitives
        # like "X is suspected" are a different beast we deliberately
        # leave to svo_passive after Q2 tightening.
        if not any(c.dep_ == "nsubj" for c in verb.children):
            return
        # No explicit complement (covered by other rules then).
        for c in verb.children:
            if c.dep_ in _BLOCK_CHILD_DEPS:
                return
        for subj in clause.subject_candidates:
            # Skip relativizer subjects (relcl gap context).
            if subj.tag_ in {"WDT", "WP", "WP$"} or subj.lower_ in {
                "who", "which", "that", "whom", "whose"
            }:
                continue
            yield Candidate(
                subject_head=subj,
                predicate_head=verb,
                arg_head=None,
                role="other",
                prep=None,
                source_rule=self.NAME,
            )
