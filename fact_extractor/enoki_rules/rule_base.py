"""Rule contract.

Every extraction rule subclasses :class:`Rule` and lives in its own file under
:mod:`fact_extractor.enoki_rules.rules`. ``__init_subclass__`` validates the
required metadata and the file-name convention at import time.
"""

from __future__ import annotations

import abc
import inspect
import sys
from pathlib import Path
from typing import ClassVar, Iterable, List, Optional, Tuple

from .models import Candidate, Clause


# One expected triplet in an EXAMPLES entry: (subject, predicate, argument).
ExpectedTriplet = Tuple[str, str, Optional[str]]
RuleExample = Tuple[str, List[ExpectedTriplet]]


class RuleContractError(TypeError):
    """Raised at import time when a Rule subclass violates the contract."""


class Rule(abc.ABC):
    """Base class for extraction rules.

    Required class attributes, validated in ``__init_subclass__``:

    * ``NAME`` — unique identifier; must equal the module's file stem.
    * ``TARGETS`` — prose description of the construction handled.
    * ``EXAMPLES`` — non-empty list of ``(sentence, expected_triplets)``.

    ``SEED``, ``PRIORITY`` and ``DEPENDENCY_PATTERNS`` are reserved metadata;
    nothing in the pipeline reads them.

    Rules must be pure: no I/O, no re-parsing, no module state. They identify
    head tokens only; span shaping and filtering happen downstream.
    """

    NAME: ClassVar[str] = ""
    TARGETS: ClassVar[str] = ""
    EXAMPLES: ClassVar[List[RuleExample]] = []

    SEED: ClassVar[bool] = False
    PRIORITY: ClassVar[int] = 0
    DEPENDENCY_PATTERNS: ClassVar[List[dict]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        if cls is Rule:
            return

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

        # The file stem must equal NAME; only enforced inside the rules package
        # so test fixtures can live elsewhere.
        module = sys.modules.get(cls.__module__)
        module_file = getattr(module, "__file__", None) if module else None
        if module_file is not None:
            stem = Path(module_file).stem
            in_rules_dir = "fact_extractor/enoki_rules/rules" in str(Path(module_file).resolve()).replace("\\", "/")
            if in_rules_dir and stem != cls.NAME:
                raise RuleContractError(
                    f"{cls.__module__}.{cls.__qualname__}: file stem '{stem}' "
                    f"does not match NAME '{cls.NAME}'"
                )

    @abc.abstractmethod
    def apply(self, clause: Clause) -> Iterable[Candidate]:
        """Yield zero or more candidates for ``clause``."""
        raise NotImplementedError
