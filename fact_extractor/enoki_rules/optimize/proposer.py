"""Rule proposers for false-negative clusters.

:class:`AgentRunner` abstracts how a rule gets written.
:class:`SubprocessAgentRunner` writes the cluster context to a JSON file and
invokes a shell command that returns a :class:`RuleProposal` as JSON;
:class:`InMemoryAgentRunner` wraps a Python callable.
"""

from __future__ import annotations

import abc
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from .cluster import FNCluster


@dataclass(frozen=True)
class RuleProposal:
    """A proposed rule: the full source of a new or edited rule file."""

    target_rule_name: str
    source_code: str
    is_new_file: bool
    rationale: str = ""


class AgentRunner(abc.ABC):
    """Interface for a component that writes rule proposals."""

    @abc.abstractmethod
    def propose(self, cluster: FNCluster) -> Optional[RuleProposal]:
        """Return a :class:`RuleProposal` or ``None`` if no proposal."""


class InMemoryAgentRunner(AgentRunner):
    """Drives proposals from a Python callable."""

    def __init__(self, fn: Callable[[FNCluster], Optional[RuleProposal]]):
        self._fn = fn

    def propose(self, cluster: FNCluster) -> Optional[RuleProposal]:
        return self._fn(cluster)


@dataclass
class SubprocessAgentRunner(AgentRunner):
    """Shells out to ``command`` for each cluster.

    The command receives the path of a cluster-context JSON file and must
    print a :class:`RuleProposal` JSON to stdout, or exit non-zero for
    "no proposal".
    """

    command: List[str]
    timeout_sec: int = 30 * 60

    def propose(self, cluster: FNCluster) -> Optional[RuleProposal]:
        context = {
            "signature": list(cluster.signature),
            "size": cluster.size,
            "examples": [
                {
                    "sentence": case.sentence_text,
                    "subject": case.gold.subject_text,
                    "predicate": case.gold.predicate_text,
                    "argument": case.gold.argument_text,
                    "role": case.gold.role,
                    "prep": case.gold.prep,
                    "is_passive": case.gold.is_passive,
                    "is_negated": case.gold.is_negated,
                }
                for case in cluster.cases[:15]
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            ctx_path = Path(td) / "context.json"
            ctx_path.write_text(json.dumps(context, indent=2))
            try:
                proc = subprocess.run(
                    self.command + [str(ctx_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return None
        if proc.returncode != 0:
            return None
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return None
        return RuleProposal(
            target_rule_name=payload["target_rule_name"],
            source_code=payload["source_code"],
            is_new_file=payload.get("is_new_file", True),
            rationale=payload.get("rationale", ""),
        )
