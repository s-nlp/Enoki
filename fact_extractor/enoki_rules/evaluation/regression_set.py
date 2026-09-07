"""Frozen regression set used by the optimisation gates.

Hand-curated (sentence, expected_triplets) pairs that every proposed rule
change must still reproduce. Loaded only by
:mod:`fact_extractor.enoki_rules.optimize.gates`. New entries must pass with
the current rule set before being added.
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
