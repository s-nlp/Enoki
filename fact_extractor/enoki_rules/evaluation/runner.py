"""Dev/test runner.

Loads QA-SRL ``*.jsonl.gz`` files from a configured set of paths, converts
each sentence to gold triplets, runs the v2 pipeline, and scores. Writes a
JSON and a Markdown summary.

Defaults to the union of dev splits enumerated in PLAN.md §1 / §6.1.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

from ..config import ExtractionConfig
from ..models import Triplet
from ..pipeline import Pipeline
from .metrics import EvalReport, score_against_gold
from .oie_to_spo import iter_lsoie, iter_openie4
from .qasrl_to_spo import GoldTriplet, QASRLConversionResult, convert_sentence, iter_jsonl_gz


log = logging.getLogger(__name__)


# A loader streams QASRLConversionResult from one split file. ``None``
# means "QA-SRL jsonl.gz" (the legacy default below).
SplitLoader = Callable[[Path, Optional[int]], Iterable[QASRLConversionResult]]


def _qasrl_loader(
    path: Path, max_per_split: Optional[int]
) -> Iterable[QASRLConversionResult]:
    count = 0
    for record in iter_jsonl_gz(path):
        yield convert_sentence(record)
        count += 1
        if max_per_split is not None and count >= max_per_split:
            break


@dataclass(frozen=True)
class DevSplit:
    """One labelled split, with the loader that knows its on-disk format."""

    name: str
    path: Path
    loader: SplitLoader = field(default=_qasrl_loader)


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data" / "open_ie"

# Dev/calibration corpus: QA-SRL Bank 2.0 was replaced by LSOIE (human
# OpenIE gold, QA-SRL lineage) + an OpenIE4 silver subset. The previous
# QA-SRL baselines / gate thresholds in rules/CHANGELOG.md do NOT carry
# over — this corpus was re-baselined from scratch.
DEFAULT_DEV_SPLITS: Tuple[DevSplit, ...] = (
    DevSplit("lsoie-wiki/validation",
             DATA_ROOT / "lsoie" / "wiki-validation.jsonl.gz", iter_lsoie),
    DevSplit("lsoie-sci/validation",
             DATA_ROOT / "lsoie" / "sci-validation.jsonl.gz", iter_lsoie),
    DevSplit("lsoie-sci/train",
             DATA_ROOT / "lsoie" / "sci-train.jsonl.gz", iter_lsoie),
    DevSplit("openie4/subset-5k",
             DATA_ROOT / "openie4_subset", iter_openie4),
)

# No dedicated OpenIE test split is shipped; reuse the dev corpus until
# a held-out split is curated.
DEFAULT_TEST_SPLITS: Tuple[DevSplit, ...] = DEFAULT_DEV_SPLITS


def iter_gold(
    splits: Iterable[DevSplit], max_per_split: Optional[int] = None
) -> Iterable[QASRLConversionResult]:
    """Stream :class:`QASRLConversionResult` from each split."""
    for split in splits:
        if not split.path.exists():
            log.warning("Split missing: %s (%s)", split.name, split.path)
            continue
        yield from split.loader(split.path, max_per_split)


def run(
    config: Optional[ExtractionConfig] = None,
    splits: Iterable[DevSplit] = DEFAULT_DEV_SPLITS,
    max_per_split: Optional[int] = None,
) -> EvalReport:
    """Run the v2 pipeline against ``splits`` and return the aggregate report."""
    config = config or ExtractionConfig()
    pipeline = Pipeline(config)

    dropped = 0
    per_sentence: List[Tuple[str, List[Triplet], List[GoldTriplet]]] = []
    for conv in iter_gold(splits, max_per_split=max_per_split):
        if conv.dropped or not conv.triplets:
            dropped += 1
            continue
        # The pipeline expects raw text; QA-SRL gives us PTB-style tokens.
        # Joining with single spaces is faithful enough for parsing.
        text = conv.sentence_text
        preds = pipeline.extract(text)
        per_sentence.append((conv.sentence_id, preds, conv.triplets))

    report = score_against_gold(
        per_sentence, score_lambda=config.optimize.score_lambda
    )
    log.info(
        "Evaluated %d sentences, dropped %d (no gold)",
        len(per_sentence),
        dropped,
    )
    return report


def write_report(report: EvalReport, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report.as_dict(), indent=2))
    md_lines = [
        "# Evaluation report",
        "",
        f"- precision: **{report.precision:.4f}**",
        f"- recall:    **{report.recall:.4f}**",
        f"- F1:        **{report.f1:.4f}**",
        f"- predicate coverage: **{report.predicate_coverage:.4f}**",
        f"- score (F1 + λ·coverage): **{report.score:.4f}**",
        "",
        f"TP/FP/FN: {report.tp} / {report.fp} / {report.fn}",
        "",
        "## Per-rule contribution",
        "",
        "| rule | TP | FP | precision |",
        "|------|----|----|-----------|",
    ]
    rules = sorted(set(report.per_rule_tp) | set(report.per_rule_fp))
    for r in rules:
        tp = report.per_rule_tp.get(r, 0)
        fp = report.per_rule_fp.get(r, 0)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        md_lines.append(f"| {r} | {tp} | {fp} | {prec:.4f} |")
    (out_dir / "report.md").write_text("\n".join(md_lines) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run v2 extractor on dev / test data.")
    parser.add_argument(
        "--split", choices=["dev", "test"], default="dev",
        help="Which split to evaluate against.",
    )
    parser.add_argument(
        "--max-per-split", type=int, default=None,
        help="Cap sentences per split (smoke testing).",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("evaluation_reports") / "latest",
        help="Output directory for report.json and report.md.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    splits = DEFAULT_DEV_SPLITS if args.split == "dev" else DEFAULT_TEST_SPLITS
    report = run(splits=splits, max_per_split=args.max_per_split)
    write_report(report, args.out)
    print(json.dumps(report.as_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
