"""Spawn agent proposals for FN clusters.

The proposer is intentionally an abstraction over *how* a rule gets
written. The default implementation (:class:`SubprocessAgentRunner`) writes
the cluster context to a JSON file and invokes a configured shell command.
The shell command is expected to return a Python source file representing
the proposed rule (or exit non-zero to indicate "no proposal").

In tests we use :class:`InMemoryAgentRunner` which lets the test specify
the proposed source directly. The optimization loop's gate logic is the
same in either case.
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
    """The artifact an agent returns.

    Either a brand-new rule file (path under
    ``fact_extractor/enoki_rules/rules/``) or an edit to an existing one.
    The optimization loop treats both as a diff applied to that directory.
    """

    target_rule_name: str         # NAME the proposed rule will register as
    source_code: str              # full contents of the .py file
    is_new_file: bool             # True if a new rule, False if edit
    rationale: str = ""           # agent's free-text justification


class AgentRunner(abc.ABC):
    """Interface for a subprocess (or in-process) agent that writes rules."""

    @abc.abstractmethod
    def propose(self, cluster: FNCluster) -> Optional[RuleProposal]:
        """Return a :class:`RuleProposal` or ``None`` if no proposal."""


class InMemoryAgentRunner(AgentRunner):
    """Drives proposals from a Python callable. Used in tests."""

    def __init__(self, fn: Callable[[FNCluster], Optional[RuleProposal]]):
        self._fn = fn

    def propose(self, cluster: FNCluster) -> Optional[RuleProposal]:
        return self._fn(cluster)


@dataclass
class SubprocessAgentRunner(AgentRunner):
    """Driver that shells out to an agent CLI for each cluster.

    The configured command receives the cluster context as JSON on stdin
    and must return a JSON :class:`RuleProposal` on stdout (or exit
    non-zero to mean "no proposal").
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
