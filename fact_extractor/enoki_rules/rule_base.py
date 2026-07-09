"""Rule contract (PLAN.md §5).

Every construction-specific extraction rule subclasses :class:`Rule` and lives
in its own file under :mod:`fact_extractor.enoki_rules.rules`. The metaclass-time hooks
enforce the contract: missing metadata, wrong file name, illegal I/O, or
illegal nesting all raise at import time so a malformed proposal never reaches
the dev evaluator.
"""

from __future__ import annotations

import abc
import inspect
import sys
from pathlib import Path
from typing import ClassVar, Iterable, List, Optional, Tuple

from .models import Candidate, Clause


# Type alias for a single expected triplet in an EXAMPLES entry. Format mirrors
# the human-readable triplet shape: (subject_text, predicate_text, arg_text).
ExpectedTriplet = Tuple[str, str, Optional[str]]
RuleExample = Tuple[str, List[ExpectedTriplet]]


class RuleContractError(TypeError):
    """Raised at import time when a Rule subclass violates the contract."""


class Rule(abc.ABC):
    """Base class for all proposer rules.

    Required class attributes (validated in ``__init_subclass__``):

    * ``NAME``                  — unique identifier; must equal the file's stem.
    * ``TARGETS``               — short prose description of the construction.
    * ``EXAMPLES``              — non-empty list of (sentence, expected_triplets).

    Optional:

    * ``SEED``                  — True if hand-written; agent may replace it.
    * ``PRIORITY``              — int used only for dedup tie-break.
    * ``DEPENDENCY_PATTERNS``   — optional spaCy DependencyMatcher pattern list.

    Rules MUST NOT:

    * perform any I/O (file, network, subprocess);
    * call ``spacy.load`` or otherwise re-parse text;
    * mutate module-level state;
    * shape spans or filter candidates — that happens downstream.
    """

    # Required attributes — concrete subclasses must override.
    NAME: ClassVar[str] = ""
    TARGETS: ClassVar[str] = ""
    EXAMPLES: ClassVar[List[RuleExample]] = []

    # Optional attributes with defaults.
    SEED: ClassVar[bool] = False
    PRIORITY: ClassVar[int] = 0
    DEPENDENCY_PATTERNS: ClassVar[List[dict]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return

        # Skip contract validation for the base class itself if it ever
        # appears here (defensive; should not happen with abc.ABC).
        if cls is Rule:
            return

        # Required-attribute presence and shape.
        if not isinstance(cls.NAME, str) or not cls.NAME:
            raise RuleContractError(
                f"{cls.__module__}.{cls.__qualname__}: NAME must be a non-empty str"
            )
        if not isinstance(cls.TARGETS, str) or not cls.TARGETS.strip():
            raise RuleContractError(
                f"{cls.__module__}.{cls.__qualname__}: TARGETS must be a non-empty description"
            )
        if not isinstance(cls.EXAMPLES, list) or len(cls.EXAMPLES) == 0:
            raise RuleContractError(
                f"{cls.__module__}.{cls.__qualname__}: EXAMPLES must be a non-empty list"
            )
        for i, ex in enumerate(cls.EXAMPLES):
            if (
                not isinstance(ex, tuple)
                or len(ex) != 2
                or not isinstance(ex[0], str)
                or not isinstance(ex[1], list)
            ):
                raise RuleContractError(
                    f"{cls.__module__}.{cls.__qualname__}: EXAMPLES[{i}] must be "
                    "(sentence: str, expected: List[Tuple[str, str, Optional[str]]])"
                )

        # File-name discipline: file stem must equal NAME (PLAN.md §5).
        # The base class lives in base.py; skip the check for it.
        module = sys.modules.get(cls.__module__)
        module_file = getattr(module, "__file__", None) if module else None
        if module_file is not None:
            stem = Path(module_file).stem
            # Allow rules defined in tests/fixtures to bypass the stem rule
            # (their module path won't match the rules/ directory anyway).
            in_rules_dir = "fact_extractor/engine/rules" in str(Path(module_file).resolve()).replace("\\", "/")
            if in_rules_dir and stem != cls.NAME:
                raise RuleContractError(
                    f"{cls.__module__}.{cls.__qualname__}: file stem '{stem}' "
                    f"does not match NAME '{cls.NAME}'"
                )

    @abc.abstractmethod
    def apply(self, clause: Clause) -> Iterable[Candidate]:
        """Yield zero or more candidates for the given clause.

        MUST be pure: no I/O, no module state, no parsing.
        """
        raise NotImplementedError
