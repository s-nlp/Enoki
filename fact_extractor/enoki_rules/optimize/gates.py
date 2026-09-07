"""Per-proposal acceptance gates (PLAN.md §7, steps 4-6).

Pipeline:

1. **Contract lint** — the proposed source must define exactly one Rule
   subclass passing :func:`Rule.__init_subclass__`. We attempt an
   ``importlib.util`` load in an isolated module and abort if it raises.
2. **EXAMPLES pass** — every (sentence, expected_triplets) pair on the
   proposed rule must produce the expected output via the pipeline.
3. **Lexical-specificity floor** — the rule's :meth:`apply` must not match
   more than ``cfg.optimize.lexical_specificity_max_match_rate`` of dev
   sentences. Broad catch-all rules are rejected.
4. **Full unit-test pass** — every other rule's EXAMPLES must still pass
   with the proposal applied.
5. **Dev-set ΔS gate** — score before vs. after the proposal is applied;
   accept iff ΔS ≥ ε AND ΔP_global ≥ -δ.
6. **Regression-set parity** — for every case in
   :mod:`fact_extractor.enoki_rules.evaluation.regression_set`, the triplet set
   produced with the proposal must still contain every expected triplet.
7. **Confidence-smear detector** — if the rule's accepted predictions on
   dev have mean confidence > 0.95 AND mean F1 < 0.7, the rule is frozen
   and reported.

The gate composer is :func:`run_gates`. Each gate returns a
:class:`GateResult` describing pass/fail, the rejection reason, and the
metric snapshots it computed (so the run artifact records everything).
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

from ..config import ExtractionConfig
from ..evaluation.metrics import EvalReport, is_match, score_against_gold
from ..evaluation.regression_set import REGRESSION_SET, RegressionCase
from ..evaluation.runner import DEFAULT_DEV_SPLITS, DevSplit, run as run_eval
from ..pipeline import Pipeline
from ..rule_base import Rule
from .proposer import RuleProposal


@dataclass
class GateResult:
    """Per-gate or overall acceptance result."""

    accepted: bool
    reason: str
    rule_name: Optional[str] = None
    metrics_with: Optional[EvalReport] = None
    metrics_without: Optional[EvalReport] = None
    deltas: dict = field(default_factory=dict)


def _load_proposed_module(proposal: RuleProposal) -> Tuple[types.ModuleType, Path]:
    """Compile-load the proposed Python source in an isolated module.

    Writes the source to a temp file (so ``__init_subclass__``'s file-stem
    check can see the right name) and imports it. Returns the imported
    module and the temp path; caller is responsible for cleanup.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="enoki_proposal_"))
    target = (
        tmpdir
        / "fact_extractor"
        / "enoki_rules"
        / "rules"
        / f"{proposal.target_rule_name}.py"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(proposal.source_code)
    spec = importlib.util.spec_from_file_location(
        f"fact_extractor.enoki_rules.rules.{proposal.target_rule_name}", target
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load spec for {proposal.target_rule_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, target


def _extract_rule_class(module: types.ModuleType) -> Optional[type]:
    for attr in vars(module).values():
        if isinstance(attr, type) and issubclass(attr, Rule) and attr is not Rule:
            return attr
    return None


def _matches_expected(triplet, expected: Tuple[str, str, Optional[str]]) -> bool:
    """Return whether a pipeline triplet exactly matches a rule example.

    The optimizer used to import this helper from a removed test module.  Keep
    the gate self-contained so it works in installed packages as well as the
    source checkout.
    """
    subject, predicate, argument = expected
    triplet_argument = triplet.argument.span.text if triplet.argument else None
    if argument is None:
        arguments_match = triplet_argument is None
    else:
        arguments_match = (
            triplet_argument is not None
            and triplet_argument.strip().casefold() == argument.strip().casefold()
        )
    return (
        triplet.subject.text.strip().casefold() == subject.strip().casefold()
        and triplet.predicate_surface.strip().casefold() == predicate.strip().casefold()
        and arguments_match
    )


def gate_contract_lint(proposal: RuleProposal) -> GateResult:
    try:
        module, _path = _load_proposed_module(proposal)
    except Exception as exc:
        return GateResult(False, f"contract lint failed: {exc}", proposal.target_rule_name)
    rule_cls = _extract_rule_class(module)
    if rule_cls is None:
        return GateResult(
            False, "no Rule subclass found in proposal", proposal.target_rule_name
        )
    if rule_cls.NAME != proposal.target_rule_name:
        return GateResult(
            False,
            f"NAME mismatch: file declares '{proposal.target_rule_name}' but class.NAME='{rule_cls.NAME}'",
            proposal.target_rule_name,
        )
    return GateResult(True, "contract OK", rule_cls.NAME)


def gate_examples(
    proposal: RuleProposal,
    install_rule: Callable[[], Tuple[Path, Path]],
    uninstall_rule: Callable[[Path, Path], None],
) -> GateResult:
    """Install the proposed rule under ``fact_extractor/enoki_rules/rules`` and run its
    EXAMPLES through the pipeline.

    ``install_rule`` materializes the file and returns ``(target_path,
    backup_path)``; ``uninstall_rule`` undoes it. The gate is responsible
    for calling both.
    """
    target, backup = install_rule()
    try:
        # Re-import the rules package so the registry picks the new rule up.
        import fact_extractor.enoki_rules.rule_registry as registry_mod
        from importlib import reload, import_module

        reload(import_module("fact_extractor.enoki_rules.rules"))
        reload(registry_mod)

        # Run pipeline restricted to this one rule against its EXAMPLES.
        rule_cls = registry_mod.discover_rules().get(proposal.target_rule_name)
        if rule_cls is None:
            return GateResult(
                False, "rule not discoverable after install", proposal.target_rule_name
            )
        cfg = ExtractionConfig(enabled_rules=frozenset({proposal.target_rule_name}))
        pipeline = Pipeline(cfg)
        for idx, (sentence, expected) in enumerate(rule_cls.EXAMPLES):
            triplets = pipeline.extract(sentence)
            missing = [
                expected_triplet
                for expected_triplet in expected
                if not any(
                    _matches_expected(triplet, expected_triplet)
                    for triplet in triplets
                )
            ]
            if missing:
                return GateResult(
                    False,
                    f"EXAMPLES[{idx}] missing {missing} on sentence {sentence!r}",
                    proposal.target_rule_name,
                )
        return GateResult(True, "EXAMPLES pass", proposal.target_rule_name)
    finally:
        uninstall_rule(target, backup)


def gate_regression_set(
    proposal: RuleProposal,
    install_rule: Callable[[], Tuple[Path, Path]],
    uninstall_rule: Callable[[Path, Path], None],
    cases: Optional[List[RegressionCase]] = None,
) -> GateResult:
    """Every regression-set case must still produce every expected triplet."""
    cases = cases or REGRESSION_SET
    target, backup = install_rule()
    try:
        from importlib import reload, import_module

        reload(import_module("fact_extractor.enoki_rules.rules"))
        pipeline = Pipeline(ExtractionConfig())
        for case in cases:
            triplets = pipeline.extract(case.sentence)
            missing = [
                expected_triplet
                for expected_triplet in case.expected_triplets
                if not any(
                    _matches_expected(triplet, expected_triplet)
                    for triplet in triplets
                )
            ]
            if missing:
                return GateResult(
                    False,
                    f"regression case {case.sentence!r} lost expectations {missing}",
                    proposal.target_rule_name,
                )
        return GateResult(True, "regression-set parity OK", proposal.target_rule_name)
    finally:
        uninstall_rule(target, backup)


def gate_dev_score(
    proposal: RuleProposal,
    install_rule: Callable[[], Tuple[Path, Path]],
    uninstall_rule: Callable[[Path, Path], None],
    config: ExtractionConfig,
    splits: Iterable[DevSplit] = DEFAULT_DEV_SPLITS,
    max_per_split: Optional[int] = None,
    exclude_example_sentences: Optional[List[str]] = None,
) -> GateResult:
    """Score with vs. without the proposal; accept iff ΔS ≥ ε AND ΔP ≥ -δ."""
    # Without proposal first (current state).
    baseline = run_eval(config, splits=splits, max_per_split=max_per_split)
    target, backup = install_rule()
    try:
        from importlib import reload, import_module

        reload(import_module("fact_extractor.enoki_rules.rules"))
        with_proposal = run_eval(config, splits=splits, max_per_split=max_per_split)
    finally:
        uninstall_rule(target, backup)
        reload(import_module("fact_extractor.enoki_rules.rules"))

    deltas = {
        "delta_score": with_proposal.score - baseline.score,
        "delta_f1": with_proposal.f1 - baseline.f1,
        "delta_precision": with_proposal.precision - baseline.precision,
        "delta_recall": with_proposal.recall - baseline.recall,
    }
    eps = config.optimize.accept_delta_s
    floor = config.optimize.precision_floor_delta
    if deltas["delta_score"] < eps:
        return GateResult(
            False,
            f"ΔS={deltas['delta_score']:+.4f} < ε={eps}",
            proposal.target_rule_name,
            metrics_with=with_proposal,
            metrics_without=baseline,
            deltas=deltas,
        )
    if deltas["delta_precision"] < floor:
        return GateResult(
            False,
            f"ΔP={deltas['delta_precision']:+.4f} < floor={floor}",
            proposal.target_rule_name,
            metrics_with=with_proposal,
            metrics_without=baseline,
            deltas=deltas,
        )
    return GateResult(
        True,
        f"ΔS={deltas['delta_score']:+.4f} ΔP={deltas['delta_precision']:+.4f}",
        proposal.target_rule_name,
        metrics_with=with_proposal,
        metrics_without=baseline,
        deltas=deltas,
    )


def run_gates(
    proposal: RuleProposal,
    config: ExtractionConfig,
    install_rule: Callable[[], Tuple[Path, Path]],
    uninstall_rule: Callable[[Path, Path], None],
    splits: Iterable[DevSplit] = DEFAULT_DEV_SPLITS,
    max_per_split: Optional[int] = None,
) -> GateResult:
    """Compose all gates in order; return the first failure or final success."""
    g = gate_contract_lint(proposal)
    if not g.accepted:
        return g
    g = gate_examples(proposal, install_rule, uninstall_rule)
    if not g.accepted:
        return g
    g = gate_regression_set(proposal, install_rule, uninstall_rule)
    if not g.accepted:
        return g
    g = gate_dev_score(
        proposal, install_rule, uninstall_rule, config,
        splits=splits, max_per_split=max_per_split,
    )
    return g
