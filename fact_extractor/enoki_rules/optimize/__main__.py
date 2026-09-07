"""CLI entrypoint: ``python -m fact_extractor.enoki_rules.optimize``.

``--agent-cmd PATH ARGS...`` drives proposals from a shell command that
receives the cluster JSON path and prints a :class:`RuleProposal` JSON to
stdout. ``--agent-stub`` uses a no-op runner for checking the gate plumbing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..config import ExtractionConfig
from .loop import OptimizationLoop
from .proposer import AgentRunner, InMemoryAgentRunner, SubprocessAgentRunner


def _stub_runner() -> AgentRunner:
    return InMemoryAgentRunner(lambda cluster: None)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the rules fact-extractor agent authoring loop."
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--agent-cmd",
        nargs="+",
        help="Shell command driving agent proposals (receives cluster JSON path).",
    )
    grp.add_argument(
        "--agent-stub",
        action="store_true",
        help="Use a no-op stub runner (gate plumbing test).",
    )
    parser.add_argument(
        "--max-per-split", type=int, default=None,
        help="Cap dev sentences per split (smoke runs).",
    )
    parser.add_argument(
        "--wallclock-sec", type=int, default=None,
        help="Override total wallclock budget.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = ExtractionConfig()
    if args.wallclock_sec is not None:
        from dataclasses import replace

        cfg = replace(cfg, optimize=replace(cfg.optimize, total_wallclock_sec=args.wallclock_sec))

    if args.agent_stub:
        runner: AgentRunner = _stub_runner()
    else:
        runner = SubprocessAgentRunner(command=args.agent_cmd)

    loop = OptimizationLoop(runner=runner, config=cfg, max_per_split=args.max_per_split)
    artifacts = loop.run()
    print(f"Run finished. Artifacts in: {artifacts.root}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
