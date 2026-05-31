"""Discovery for the rules line.

Binds the parameterized ``discover_rules`` / ``get_rule_class`` from
``fact_extractor.enoki_rules.rule_registry`` to the
``fact_extractor.enoki_rules.rules`` package.
"""

from __future__ import annotations

from typing import Dict, Type

from ..rule_base import Rule
from ..rule_registry import discover_rules as _discover
from ..rule_registry import get_rule_class as _get

_PACKAGE = "fact_extractor.enoki_rules.rules"


def discover_rules() -> Dict[str, Type[Rule]]:
    return _discover(_PACKAGE)


def get_rule_class(name: str) -> Type[Rule]:
    return _get(name, _PACKAGE)
