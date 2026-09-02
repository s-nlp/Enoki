"""
sample_anah_250.py  –  Sample 250 ANAH sentences that have a non-empty
ann_reference (i.e. context exists) with balanced gold labels.

Output: anah_250_sample.jsonl  (one sentence-row per line)

Usage:
    python sample_anah_250.py
    python sample_anah_250.py --n 250 --seed 42 --out anah_250_sample.jsonl
"""

import argparse
import json
import random
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from anah_utils import iter_anah_sentences


def has_real_reference(text: str) -> bool:
    s = str(text or "").strip()
    if not s:
        return False
    if s.lower() in {"none", "null", "n/a", "na"}:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250, help="Number of sentences to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="anah_250_sample.jsonl")
    ap.add_argument("--anah_split", type=str, default="train")
    ap.add_argument("--balance", action="store_true", default=True,
                    help="Balance supported / not_supported (default: True)")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("opencompass/anah", split=args.anah_split)

    supported = []
    not_supported = []

    for row in iter_anah_sentences(ds):
        if row["gold_supported"] is None:
            continue  # No Fact
        if not has_real_reference(row["ann_reference"]):
            continue  # no context
        if not row["sentence"].strip():
            continue
        # Skip non-ASCII (Chinese) sentences
        if not row["sentence"].isascii():
            continue
        if row["gold_supported"]:
            supported.append(row)
        else:
            not_supported.append(row)

    print(f"Pool: supported={len(supported)}, not_supported={len(not_supported)}")

    rng = random.Random(args.seed)

    if args.balance:
        half = args.n // 2
        sampled_sup = rng.sample(supported, min(half, len(supported)))
        sampled_not = rng.sample(not_supported, min(args.n - len(sampled_sup), len(not_supported)))
        sampled = sampled_sup + sampled_not
    else:
        pool = supported + not_supported
        sampled = rng.sample(pool, min(args.n, len(pool)))

    rng.shuffle(sampled)

    out_path = os.path.join(os.path.dirname(__file__), args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in sampled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_sup = sum(1 for r in sampled if r["gold_supported"])
    n_not = sum(1 for r in sampled if not r["gold_supported"])
    print(f"Sampled {len(sampled)} sentences → {out_path}")
    print(f"  gold_supported=True:  {n_sup}")
    print(f"  gold_supported=False: {n_not}")


if __name__ == "__main__":
    main()
