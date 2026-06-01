"""Re-export of the shared Rule contract from ``fact_extractor.enoki_rules.rule_base``."""

from __future__ import annotations

from ..rule_base import Rule, RuleContractError, RuleExample, ExpectedTriplet

__all__ = ["Rule", "RuleContractError", "RuleExample", "ExpectedTriplet"]
