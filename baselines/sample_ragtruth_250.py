"""
sample_ragtruth_250.py  –  Sample 250 RAGTruth *sentences* with stratified
sampling across task_type × gold_supported, from rows with quality=='good'
that have non-empty context.

Output: ragtruth_250_sample.jsonl  (one sentence-row per line, same schema
as iter_ragtruth_sentences output – directly consumable by all runners)

Usage:
    python sample_ragtruth_250.py
    python sample_ragtruth_250.py --n 250 --seed 42 --out ragtruth_250_sample.jsonl
    python sample_ragtruth_250.py --split test   # use test split instead
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.dirname(__file__))
from ragtruth_utils import iter_ragtruth_sentences, parse_hallucination_labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250, help="Total sentences to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="ragtruth_250_sample.jsonl")
    ap.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="HuggingFace split to sample from (default: train)",
    )
    ap.add_argument(
        "--only_labeled",
        action="store_true",
        default=True,
        help="Only include responses that have at least one hallucination label "
             "(ensures ~50/50 balance across supported/not_supported is achievable). "
             "Default: True",
    )
    args = ap.parse_args()

    from datasets import load_dataset  # noqa: PLC0415

    print(f"Loading wandb/RAGTruth-processed split='{args.split}' …", flush=True)
    ds = load_dataset("wandb/RAGTruth-processed", split=args.split)
    print(f"  {len(ds)} total rows", flush=True)

    # ------------------------------------------------------------------ #
    # Filter: quality=='good', non-empty context, has hallucination labels
    # ------------------------------------------------------------------ #
    good_rows = []
    for row in ds:
        if row.get("quality") != "good":
            continue
        ctx = (row.get("context") or "").strip()
        if not ctx:
            continue
        output = (row.get("output") or "").strip()
        if not output:
            continue
        if args.only_labeled:
            spans = parse_hallucination_labels(row.get("hallucination_labels"))
            if not spans:
                continue  # skip fully-supported responses
        good_rows.append(dict(row))

    print(f"  After filters: {len(good_rows)} rows", flush=True)

    # ------------------------------------------------------------------ #
    # Expand to sentence rows
    # ------------------------------------------------------------------ #
    # Stratify by (task_type, gold_supported)
    strata: Dict[tuple, List[dict]] = defaultdict(list)
    for sent_row in iter_ragtruth_sentences(good_rows):
        key = (sent_row["task_type"], sent_row["gold_supported"])
        strata[key].append(sent_row)

    print("Strata sizes:", flush=True)
    for k, v in sorted(strata.items()):
        print(f"  task={k[0]:10s}  gold_supported={k[1]}  n={len(v)}", flush=True)

    # ------------------------------------------------------------------ #
    # Stratified sample
    # ------------------------------------------------------------------ #
    rng = random.Random(args.seed)
    n_strata = len(strata)
    base_per_stratum = args.n // n_strata
    remainder = args.n - base_per_stratum * n_strata

    sampled: List[dict] = []
    keys_sorted = sorted(strata.keys())
    for i, key in enumerate(keys_sorted):
        pool = strata[key]
        take = base_per_stratum + (1 if i < remainder else 0)
        take = min(take, len(pool))
        sampled.extend(rng.sample(pool, take))

    rng.shuffle(sampled)

    # ------------------------------------------------------------------ #
    # Write output
    # ------------------------------------------------------------------ #
    out_path = os.path.join(os.path.dirname(__file__), args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in sampled:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    # Summary
    from collections import Counter
    by_task = Counter(r["task_type"] for r in sampled)
    by_gold = Counter(r["gold_supported"] for r in sampled)
    print(f"\nSampled {len(sampled)} sentences → {out_path}")
    print(f"  By task:  {dict(by_task)}")
    print(f"  gold_supported=True:  {by_gold[True]}")
    print(f"  gold_supported=False: {by_gold[False]}")


if __name__ == "__main__":
    main()
