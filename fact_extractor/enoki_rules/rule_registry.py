"""Auto-discovery of Rule subclasses from ``fact_extractor/enoki_rules/rules/*.py``.

Agents only write files; they never edit a registry list. Anything dropped in
this directory whose module-level class subclasses :class:`Rule` and passes
the contract becomes a known rule.

Protected from agent edits (PLAN.md §7).
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, Iterator, Type

from .rule_base import Rule


_RESERVED_MODULE_NAMES = {"base", "_registry", "__init__"}


def discover_rules(
    package: str = "fact_extractor.enoki_rules.rules",
) -> Dict[str, Type[Rule]]:
    """Import every rule module in ``package`` and return ``{NAME: rule_class}``.

    ``package`` defaults to the legacy line so existing callers are
    unaffected. Pass ``"fact_extractor.enoki_rules.rules"`` for the new line.
    """
    rules_pkg = importlib.import_module(package)

    found: Dict[str, Type[Rule]] = {}
    for mod_info in pkgutil.iter_modules(rules_pkg.__path__):
        name = mod_info.name
        if name in _RESERVED_MODULE_NAMES or name.startswith("_"):
            continue
        module = importlib.import_module(f"{package}.{name}")
        for attr in vars(module).values():
            if isinstance(attr, type) and issubclass(attr, Rule) and attr is not Rule:
                if attr.NAME in found and found[attr.NAME] is not attr:
                    raise RuntimeError(
                        f"Duplicate Rule NAME '{attr.NAME}' in modules "
                        f"{found[attr.NAME].__module__} and {attr.__module__}"
                    )
                found[attr.NAME] = attr
    return found


def get_rule_class(
    name: str, package: str = "fact_extractor.enoki_rules.rules"
) -> Type[Rule]:
    """Convenience lookup for a single rule by NAME within ``package``."""
    rules = discover_rules(package)
    if name not in rules:
        raise KeyError(f"Unknown rule: '{name}'")
    return rules[name]


def iter_rules() -> Iterator[Type[Rule]]:
    """Iterate over all discovered rule classes."""
    yield from discover_rules().values()
