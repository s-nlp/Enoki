"""Top-level optimization driver (PLAN.md §7).

Iteration body:

1. Run pipeline on calibration set, score, identify false negatives.
2. Cluster FNs by dependency-pattern signature.
3. Pick the top-K clusters (size × current-FN rate); respect cooldowns.
4. For each cluster, ask the agent runner for a :class:`RuleProposal`.
5. Run all gates on every proposal (contract → EXAMPLES → regression → ΔS).
6. Accepted proposals: install the rule, refresh registry, log artifacts.
   Rejected proposals: log artifacts and mark the cluster cooled.
7. Repeat until a stop condition is hit.

This module owns the rule-file installation primitive (write, registry
refresh, backup/restore on failure). The agent never sees the live
``rules/`` directory.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from importlib import import_module, reload
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from ..config import ExtractionConfig
from ..evaluation.metrics import is_match, score_against_gold
from ..evaluation.runner import DEFAULT_DEV_SPLITS, DevSplit, iter_gold, run as run_eval
from ..pipeline import Pipeline
from ..preprocess import Parser
from .cluster import FNCluster, cluster_false_negatives
from .gates import GateResult, run_gates
from .proposer import AgentRunner, RuleProposal
from .runs import RULES_DIR, RunArtifacts, new_run


log = logging.getLogger(__name__)


@dataclass
class LoopState:
    """Mutable state across iterations."""

    iteration: int = 0
    last_score: Optional[float] = None
    plateau_streak: int = 0
    cooldown: dict = field(default_factory=dict)  # signature tuple -> iter cooled until
    epsilon: float = 0.0
    delta: float = 0.0


def _refresh_registry() -> None:
    """Force a re-import so newly written rules become discoverable."""
    rules_pkg = import_module("fact_extractor.enoki_rules.rules")
    reload(rules_pkg)
    registry_mod = import_module("fact_extractor.enoki_rules.rule_registry")
    reload(registry_mod)


class OptimizationLoop:
    def __init__(
        self,
        runner: AgentRunner,
        config: Optional[ExtractionConfig] = None,
        splits: Iterable[DevSplit] = DEFAULT_DEV_SPLITS,
        max_per_split: Optional[int] = None,
        artifacts: Optional[RunArtifacts] = None,
    ) -> None:
        self.runner = runner
        self.config = config or ExtractionConfig()
        self.splits = list(splits)
        self.max_per_split = max_per_split
        self.artifacts = artifacts or new_run()
        self.state = LoopState(
            epsilon=self.config.optimize.accept_delta_s,
            delta=self.config.optimize.precision_floor_delta,
        )
        self._wallclock_start = time.time()

    # ----- public driver -----

    def run(self) -> RunArtifacts:
        """Run the loop until a stop condition is hit."""
        while not self._should_stop():
            self._run_iteration()
        self.artifacts.finalize()
        return self.artifacts

    # ----- iteration body -----

    def _run_iteration(self) -> None:
        self.state.iteration += 1
        log.info("--- iteration %d ---", self.state.iteration)
        clusters = self._collect_clusters()
        if not clusters:
            log.info("no FN clusters above floor; stopping")
            self.state.plateau_streak = 10
            return
        accepted_this_iter = 0
        top_k = self.config.optimize.top_k_clusters_per_iter
        for cluster in clusters[:top_k]:
            if self._is_cooled(cluster):
                continue
            proposal = self.runner.propose(cluster)
            if proposal is None:
                self._cool(cluster)
                self._log_no_proposal(cluster)
                continue
            decision = self._evaluate_proposal(proposal, cluster)
            if decision.accepted:
                accepted_this_iter += 1
            else:
                self._cool(cluster)
        # Plateau tightening: if a full iteration didn't move S by more
        # than 2·stop_eps, tighten ε and δ.
        new_score = self._current_score()
        if self.state.last_score is not None:
            ds = new_score - self.state.last_score
            if abs(ds) < 2 * self.config.optimize.stop_eps:
                self.state.plateau_streak += 1
                self.state.epsilon *= 2
                self.state.delta /= 2
                log.info(
                    "plateau detected (ΔS=%+.4f); tightening ε→%.4f δ→%.4f",
                    ds, self.state.epsilon, self.state.delta,
                )
            else:
                self.state.plateau_streak = 0
        self.state.last_score = new_score

    # ----- gates wrapper -----

    def _evaluate_proposal(
        self, proposal: RuleProposal, cluster: FNCluster
    ) -> GateResult:
        prop_dir = self.artifacts.proposal_dir()
        (prop_dir / "spec.json").write_text(
            _dump_cluster_spec(cluster, proposal)
        )
        (prop_dir / "source.py").write_text(proposal.source_code)
        install, uninstall = _file_install_helpers(proposal)
        result = run_gates(
            proposal,
            config=self.config,
            install_rule=install,
            uninstall_rule=uninstall,
            splits=self.splits,
            max_per_split=self.max_per_split,
        )
        decision = {
            "rule_name": result.rule_name,
            "accepted": result.accepted,
            "reason": result.reason,
            "deltas": result.deltas,
            "rationale": proposal.rationale,
        }
        if result.accepted:
            # Promote the proposal: persist the rule file into RULES_DIR.
            target = RULES_DIR / f"{proposal.target_rule_name}.py"
            target.write_text(proposal.source_code)
            _refresh_registry()
            self._append_changelog(proposal, result, cluster)
        self.artifacts.record(prop_dir, decision)
        return result

    # ----- helpers -----

    def _collect_clusters(self) -> List[FNCluster]:
        # Build the per-sentence (preds, golds) list once.
        pipeline = Pipeline(self.config)
        per_sentence = []
        matched_flags = {}
        for conv in iter_gold(self.splits, max_per_split=self.max_per_split):
            if conv.dropped or not conv.triplets:
                continue
            preds = pipeline.extract(conv.sentence_text)
            per_sentence.append((conv.sentence_id, preds, conv.triplets))
            # Pre-compute match flags so the clusterer doesn't re-run the
            # matching loop.
            gold_matched = [False] * len(conv.triplets)
            for p in preds:
                for j, g in enumerate(conv.triplets):
                    if gold_matched[j]:
                        continue
                    if is_match(p, g):
                        gold_matched[j] = True
                        break
            for j, m in enumerate(gold_matched):
                matched_flags[(conv.sentence_id, j)] = m
        parser = Parser(
            model=self.config.spacy_model,
            use_gliner=self.config.use_gliner,
            gliner_model=self.config.gliner_model,
        )
        clusters = cluster_false_negatives(per_sentence, matched_flags, parser=parser)
        floor = self.config.optimize.cluster_size_floor
        return [c for c in clusters if c.size >= floor]

    def _current_score(self) -> float:
        report = run_eval(
            self.config, splits=self.splits, max_per_split=self.max_per_split
        )
        return report.score

    def _is_cooled(self, cluster: FNCluster) -> bool:
        return self.state.cooldown.get(cluster.signature, 0) > self.state.iteration

    def _cool(self, cluster: FNCluster) -> None:
        self.state.cooldown[cluster.signature] = (
            self.state.iteration + self.config.optimize.cooldown_iterations
        )

    def _log_no_proposal(self, cluster: FNCluster) -> None:
        prop_dir = self.artifacts.proposal_dir()
        (prop_dir / "spec.json").write_text(_dump_cluster_spec(cluster, None))
        self.artifacts.record(
            prop_dir,
            {
                "rule_name": None,
                "accepted": False,
                "reason": "no proposal from agent",
                "deltas": {},
            },
        )

    def _append_changelog(
        self, proposal: RuleProposal, result: GateResult, cluster: FNCluster
    ) -> None:
        changelog = RULES_DIR / "CHANGELOG.md"
        entry = (
            f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')} — {proposal.target_rule_name}\n"
            f"- signature: `{cluster.signature}`\n"
            f"- cluster size: {cluster.size}\n"
            f"- ΔS: {result.deltas.get('delta_score', 0):+.4f}, "
            f"ΔP: {result.deltas.get('delta_precision', 0):+.4f}, "
            f"ΔR: {result.deltas.get('delta_recall', 0):+.4f}\n"
            f"- rationale: {proposal.rationale}\n"
        )
        with changelog.open("a") as fh:
            fh.write(entry)

    def _should_stop(self) -> bool:
        if self.state.plateau_streak >= 3:
            log.info("stop: plateau streak hit")
            return True
        elapsed = time.time() - self._wallclock_start
        if elapsed >= self.config.optimize.total_wallclock_sec:
            log.info("stop: wallclock budget exhausted")
            return True
        return False


def _file_install_helpers(proposal: RuleProposal):
    """Build install/uninstall callbacks that the gates use.

    Returns ``(install, uninstall)`` callables. ``install`` writes the
    proposal under ``RULES_DIR`` and returns ``(target_path, backup_path)``.
    ``uninstall`` restores the backup or removes the file.
    """
    def install():
        target = RULES_DIR / f"{proposal.target_rule_name}.py"
        backup = None
        if target.exists():
            backup = target.with_suffix(".py.bak")
            shutil.copy2(target, backup)
        target.write_text(proposal.source_code)
        _refresh_registry()
        return target, backup

    def uninstall(target: Path, backup: Optional[Path]):
        if backup and backup.exists():
            shutil.move(str(backup), str(target))
        else:
            if target.exists():
                target.unlink()
        _refresh_registry()

    return install, uninstall


def _dump_cluster_spec(cluster: FNCluster, proposal: Optional[RuleProposal]) -> str:
    import json

    obj = {
        "signature": list(cluster.signature),
        "size": cluster.size,
        "examples": [
            {
                "sentence": case.sentence_text,
                "subject": case.gold.subject_text,
                "predicate": case.gold.predicate_text,
                "argument": case.gold.argument_text,
            }
            for case in cluster.cases[:15]
        ],
        "proposal_target_name": proposal.target_rule_name if proposal else None,
        "proposal_rationale": proposal.rationale if proposal else None,
    }
    return json.dumps(obj, indent=2)
