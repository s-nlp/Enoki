"""Optimization-run artifact layout.

One directory per loop run, structured per PLAN.md §9::

    optimization_runs/<timestamp>/
      seed_rules/              # snapshot of rules/ at run start
      proposals/<n>/
        spec.json              # cluster signature + cases
        diff.patch             # the file change (unified diff)
        eval_with.json
        eval_without.json
        decision.json
      final_rules/             # rules/ at run end
      REPORT.md
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
RULES_DIR = REPO_ROOT / "fact_extractor" / "v2" / "rules"


@dataclass
class RunArtifacts:
    """Handle to a single optimization run's artifact directory."""

    root: Path
    proposal_count: int = 0
    accepted_count: int = 0
    decisions: List[Dict] = field(default_factory=list)

    def proposal_dir(self) -> Path:
        idx = self.proposal_count
        self.proposal_count += 1
        d = self.root / "proposals" / f"{idx:04d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def record(self, proposal_dir: Path, decision: Dict) -> None:
        (proposal_dir / "decision.json").write_text(json.dumps(decision, indent=2))
        self.decisions.append(decision)
        if decision.get("accepted"):
            self.accepted_count += 1

    def finalize(self) -> None:
        # Snapshot final rules and write summary.
        final = self.root / "final_rules"
        final.mkdir(exist_ok=True)
        _copy_rules_dir(RULES_DIR, final)
        report_lines = [
            f"# Optimization run report",
            "",
            f"- run dir: `{self.root.name}`",
            f"- proposals: {self.proposal_count}",
            f"- accepted: {self.accepted_count}",
            "",
            "## Decisions",
            "",
            "| # | rule | accepted | reason |",
            "|---|------|----------|--------|",
        ]
        for i, dec in enumerate(self.decisions):
            report_lines.append(
                f"| {i:04d} | {dec.get('rule_name', '?')} | "
                f"{dec.get('accepted', False)} | {dec.get('reason', '')} |"
            )
        (self.root / "REPORT.md").write_text("\n".join(report_lines) + "\n")


def _copy_rules_dir(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        if entry.is_file() and entry.suffix in {".py", ".md"}:
            shutil.copy2(entry, dst / entry.name)


def new_run(root: Optional[Path] = None) -> RunArtifacts:
    """Create a new run directory and snapshot the seed rules into it."""
    root = root or (REPO_ROOT / "optimization_runs")
    root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = root / ts
    run_dir.mkdir()
    _copy_rules_dir(RULES_DIR, run_dir / "seed_rules")
    return RunArtifacts(root=run_dir)
