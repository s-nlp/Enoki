"""Auto-discovery of :class:`Rule` subclasses in the rules package.

Any module in ``fact_extractor.enoki_rules.rules`` whose module-level class
subclasses :class:`Rule` becomes a known rule; there is no hand-maintained list.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, Type

from .rule_base import Rule


RULES_PACKAGE = "fact_extractor.enoki_rules.rules"


def discover_rules() -> Dict[str, Type[Rule]]:
    """Import every rule module and return ``{NAME: rule_class}``."""
    rules_pkg = importlib.import_module(RULES_PACKAGE)

    found: Dict[str, Type[Rule]] = {}
    for mod_info in pkgutil.iter_modules(rules_pkg.__path__):
        name = mod_info.name
        if name.startswith("_"):
            continue
        module = importlib.import_module(f"{RULES_PACKAGE}.{name}")
        for attr in vars(module).values():
            if isinstance(attr, type) and issubclass(attr, Rule) and attr is not Rule:
                if attr.NAME in found and found[attr.NAME] is not attr:
                    raise RuntimeError(
                        f"Duplicate Rule NAME '{attr.NAME}' in modules "
                        f"{found[attr.NAME].__module__} and {attr.__module__}"
                    )
                found[attr.NAME] = attr
    return found
