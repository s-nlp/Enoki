"""The active v2 rule catalogue.

Discovery is automatic via :mod:`fact_extractor.enoki_rules.rule_registry`;
never hand-wire a rule list. Origin: this directory began life as
``rules_new`` (the top-down rule-authoring line — see
``docs/superpowers/specs/2026-05-16-rules-new-topdown-line-design.md``)
and was renamed to ``rules`` after the original ``rules`` package was
deleted (2026-05-23 refactor).
"""

from ._registry import discover_rules, get_rule_class

__all__ = ["discover_rules", "get_rule_class"]
