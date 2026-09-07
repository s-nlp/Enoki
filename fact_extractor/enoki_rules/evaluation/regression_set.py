"""Frozen regression set used by the agent loop gate.

A hand-curated set of (sentence, expected_triplets) pairs the optimization
loop's CI gate must reproduce identically with any proposed change applied
(PLAN.md §7, step 5). The agent NEVER sees this set; it is loaded only by
:mod:`fact_extractor.enoki_rules.optimize.gates`.

This file is the canonical source. To extend it: add tuples to
``REGRESSION_SET`` and run ``python -m fact_extractor.enoki_rules.evaluation.runner``
once locally to confirm the new entries pass with the current rule set.

NOTE: PLAN.md §10 / M4 calls for 200 entries stratified across the four dev
splits. The initial commit ships with a small seed list bootstrapping from
the seed-rule EXAMPLES; the dev-set curation grows it to 200 in later work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


ExpectedTriplet = Tuple[str, str, Optional[str]]


@dataclass(frozen=True)
class RegressionCase:
    sentence: str
    expected_triplets: List[ExpectedTriplet]
    note: str = ""


# Hand-curated regression seeds. Each entry must pass with all seed rules
# enabled. The agent NEVER reads this list.
REGRESSION_SET: List[RegressionCase] = [
    RegressionCase(
        "Alice signed the bill.",
        [("Alice", "signed", "bill")],
        note="svo_active baseline",
    ),
    RegressionCase(
        "Paris is the capital of France.",
        [("Paris", "is", "capital")],
        note="copula with attr child",
    ),
    RegressionCase(
        "Albert Einstein was a physicist.",
        [("Albert Einstein", "was", "physicist")],
        note="copula with proper-noun subject",
    ),
    RegressionCase(
        "Marie Curie, a physicist, won the Nobel Prize.",
        [("Marie Curie", "is", "physicist"),
         ("Marie Curie", "won", "Nobel Prize")],
        note="appositive + svo_active",
    ),
    RegressionCase(
        "Alice was born in Boston.",
        [("Alice", "was born in", "Boston")],
        note="passive + prep argument",
    ),
    RegressionCase(
        "The bill was signed by the president.",
        [("bill", "was signed by", "president")],
        note="passive with by-agent",
    ),
    RegressionCase(
        "Napoleon died in 1821.",
        [("Napoleon", "died in", "1821")],
        note="temporal prep + bare year (filter must allow)",
    ),
    RegressionCase(
        "They moved to Berlin in 1989.",
        [("They", "moved to", "Berlin"),
         ("They", "moved in", "1989")],
        note="two prep arguments on the same verb",
    ),
    RegressionCase(
        "She studied at Oxford.",
        [("She", "studied at", "Oxford")],
        note="locative prep argument",
    ),
    RegressionCase(
        "The treaty was ratified by Congress.",
        [("treaty", "was ratified by", "Congress")],
        note="passive + by-agent",
    ),
]
