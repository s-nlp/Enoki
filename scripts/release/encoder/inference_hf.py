#!/usr/bin/env python3
"""Minimal inference CLI for a local or Hub-hosted Enoki OpenIE model."""

from __future__ import annotations

import argparse
import json

from transformers import AutoModel


EXAMPLES = [
    "Barack Obama was born in Honolulu.",
    "Apple acquired Beats Electronics for $3 billion in 2014.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=".", help="HF repo ID or local model path")
    parser.add_argument("--text", action="append", help="Repeat for multiple sentences")
    parser.add_argument("--min-confidence", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    model.to("cpu").eval()
    results = model.extract_triples(
        args.text or EXAMPLES,
        min_confidence=args.min_confidence,
        top_k=args.top_k,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
