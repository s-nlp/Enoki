"""Agent-driven rule authoring loop.

Public surface: :class:`OptimizationLoop`. The driver:

1. Runs the current ruleset on the calibration set;
2. Clusters false negatives by dependency-pattern signature;
3. Spawns one agent proposal per top-K cluster;
4. Validates each proposal through CI + dev-set + regression-set gates;
5. Accepts winners, writes artifacts, repeats until a stop condition is hit.

This module orchestrates; the actual *agent* that writes a rule file lives
outside this package (see :class:`AgentRunner` in :mod:`.proposer`). The
default implementation invokes a shell command per proposal; in production
it is overridden with a subprocess that drives a Claude / Codex /
human-in-the-loop session.
"""

from .cluster import FNCluster, cluster_false_negatives
from .gates import GateResult, run_gates
from .loop import OptimizationLoop
from .proposer import AgentRunner, RuleProposal
from .runs import RunArtifacts, new_run

__all__ = [
    "AgentRunner",
    "FNCluster",
    "GateResult",
    "OptimizationLoop",
    "RuleProposal",
    "RunArtifacts",
    "cluster_false_negatives",
    "new_run",
    "run_gates",
]
