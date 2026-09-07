"""Rule-authoring optimisation loop.

:class:`OptimizationLoop` runs the current ruleset on the dev corpus,
clusters false negatives by dependency signature, requests one rule proposal
per top cluster from an :class:`AgentRunner`, validates each proposal through
the gates, and installs accepted rules until a stop condition is hit. The
proposer itself lives outside this package (see :mod:`.proposer`).
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
