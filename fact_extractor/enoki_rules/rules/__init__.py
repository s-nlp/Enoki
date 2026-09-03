"""The active rule catalogue.

Discovery is automatic via :mod:`fact_extractor.enoki_rules.rule_registry`;
never hand-wire a rule list.
"""

from ._registry import discover_rules, get_rule_class

__all__ = ["discover_rules", "get_rule_class"]
