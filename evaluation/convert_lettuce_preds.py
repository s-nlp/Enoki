"""
Convert Lettuce prediction CSVs to the gold/pred format expected by span_coverage_report.py.

Lettuce files have:
  - hallucination_labels: JSON list of {"start", "end", ...} dicts  →  gold spans
  - pred:                 JSON list of {"start", "end", ...} dicts  →  predicted spans

Output: one CSV per input with columns  gold,pred  where each cell is [[start,end],...].

Usage:
    python -m evaluation.convert_lettuce_preds \\
        --input-dir predictions \\
        --output-dir predictions \\
        --pattern "lettuce_large_psilo_*.csv" \\
        --dataset psiloqa
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List


def _to_span_list(raw: str | list) -> List[List[int]]:
    """Parse a JSON-encoded list of span dicts into [[start, end], ...] format."""
    if isinstance(raw, str):
        items = json.loads(raw)
    else:
        items = raw
    spans = []
    for item in items:
        s, e = int(item["start"]), int(item["end"])
        if e > s:
            spans.append([s, e])
    return spans


def convert_file(src: Path, dst: Path) -> None:
    rows_gold: List[List[List[int]]] = []
    rows_pred: List[List[List[int]]] = []

    with src.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_gold.append(_to_span_list(row["hallucination_labels"]))
            rows_pred.append(_to_span_list(row["pred"]))

    with dst.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gold", "pred"])
        for g, p in zip(rows_gold, rows_pred):
            writer.writerow([str(g), str(p)])

    print(f"Converted {src.name} → {dst.name}  ({len(rows_gold)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert Lettuce prediction CSVs to gold/pred format")
    ap.add_argument("--input-dir", type=Path, default=Path("predictions"))
    ap.add_argument("--output-dir", type=Path, default=Path("predictions"))
    ap.add_argument("--pattern", default="lettuce_large_psilo_*.csv", help="Glob pattern for input files")
    ap.add_argument("--dataset", default="psiloqa", help="Dataset name to embed in output filename (default: psiloqa)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    found = sorted(args.input_dir.glob(args.pattern))
    if not found:
        print(f"No files matching '{args.pattern}' in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    for src in found:
        # Replace the first occurrence of "psilo" with the target dataset name in the stem,
        # so e.g. lettuce_large_psilo_mismatch_rephrased.csv → lettuce_large_psiloqa_mismatch_rephrased.csv
        stem = src.stem.replace("psilo_", f"{args.dataset}_", 1)
        if stem == src.stem:
            # pattern didn't match; just prepend dataset
            stem = src.stem + f"_{args.dataset}"
        dst = args.output_dir / (stem + ".csv")
        convert_file(src, dst)


if __name__ == "__main__":
    main()
