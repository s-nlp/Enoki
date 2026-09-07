"""The active rule catalogue.

Every module in this package that defines a :class:`Rule` subclass is picked
up automatically by :func:`fact_extractor.enoki_rules.rule_registry.discover_rules`;
never hand-wire a rule list.
"""
