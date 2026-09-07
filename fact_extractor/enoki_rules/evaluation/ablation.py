"""One-rule-out ablation.

For each registered rule, run the pipeline with that rule disabled and
report the metric deltas against the full-rule baseline.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..config import ExtractionConfig
from ..rule_registry import discover_rules
from .metrics import EvalReport
from .runner import DEFAULT_DEV_SPLITS, DevSplit, run


log = logging.getLogger(__name__)


def run_ablation(
    splits: Iterable[DevSplit] = DEFAULT_DEV_SPLITS,
    max_per_split: Optional[int] = None,
) -> Dict[str, Dict]:
    rules = discover_rules()
    if not rules:
        return {}

    baseline_cfg = ExtractionConfig()
    baseline = run(baseline_cfg, splits=splits, max_per_split=max_per_split)
    out: Dict[str, Dict] = {
        "_baseline": baseline.as_dict(),
    }
    full_names = frozenset(rules.keys())
    for rule_name in sorted(rules):
        enabled = full_names - {rule_name}
        cfg = baseline_cfg.with_enabled_rules(enabled)
        report = run(cfg, splits=splits, max_per_split=max_per_split)
        out[rule_name] = {
            "report": report.as_dict(),
            "delta_score": report.score - baseline.score,
            "delta_f1": report.f1 - baseline.f1,
            "delta_precision": report.precision - baseline.precision,
            "delta_recall": report.recall - baseline.recall,
        }
        log.info(
            "Ablation %s: ΔS=%+.4f ΔF1=%+.4f ΔP=%+.4f ΔR=%+.4f",
            rule_name,
            out[rule_name]["delta_score"],
            out[rule_name]["delta_f1"],
            out[rule_name]["delta_precision"],
            out[rule_name]["delta_recall"],
        )
    return out


def write_ablation_md(rows: Dict[str, Dict], out_path: Path) -> None:
    if "_baseline" not in rows:
        out_path.write_text("# Ablation report\n\n(no rules registered)\n")
        return
    base = rows["_baseline"]
    lines = [
        "# Ablation report",
        "",
        f"Baseline: F1={base['f1']:.4f} P={base['precision']:.4f} "
        f"R={base['recall']:.4f} coverage={base['predicate_coverage']:.4f} "
        f"score={base['score']:.4f}",
        "",
        "| rule disabled | ΔS | ΔF1 | ΔP | ΔR |",
        "|---------------|----|----|----|----|",
    ]
    for rule, row in rows.items():
        if rule == "_baseline":
            continue
        lines.append(
            f"| {rule} | {row['delta_score']:+.4f} | {row['delta_f1']:+.4f} "
            f"| {row['delta_precision']:+.4f} | {row['delta_recall']:+.4f} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="One-rule-out ablation on dev set.")
    parser.add_argument("--max-per-split", type=int, default=None)
    parser.add_argument(
        "--out", type=Path, default=Path("evaluation_reports") / "ablation"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rows = run_ablation(max_per_split=args.max_per_split)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ablation.json").write_text(json.dumps(rows, indent=2))
    write_ablation_md(rows, args.out / "ablation.md")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
