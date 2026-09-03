#!/usr/bin/env python3
"""Run the Enoki IGL/ModernBERT OpenIE checkpoint on CPU.

Each input item should contain one English sentence. With no --text or --input,
the script runs a small built-in smoke-test set.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoTokenizer

from model.data import UNUSED_TOKENS
from model.model import IGLModel
from model.predict import extract


EXAMPLE_SENTENCES = [
    "Barack Obama was born in Honolulu.",
    "Apple acquired Beats Electronics for $3 billion in 2014.",
    "Marie Curie won the Nobel Prize in Physics in 1903.",
    "The Eiffel Tower is located in Paris, France.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU inference for the Enoki IGL ModernBERT encoder"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Lightning .ckpt file",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--text",
        action="append",
        help="One English sentence; repeat the option to pass several sentences",
    )
    source.add_argument(
        "--input",
        type=Path,
        help="UTF-8 text file with one English sentence per non-empty line",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    return parser.parse_args()


def read_sentences(args: argparse.Namespace) -> list[str]:
    if args.text:
        return [sentence.strip() for sentence in args.text if sentence.strip()]
    if args.input:
        with args.input.open(encoding="utf-8") as stream:
            return [line.strip() for line in stream if line.strip()]
    return EXAMPLE_SENTENCES


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1")
    if not 0.0 <= args.min_confidence <= 1.0:
        raise SystemExit("--min-confidence must be between 0 and 1")

    sentences = read_sentences(args)
    if not sentences:
        raise SystemExit("No non-empty sentences to process")

    device = torch.device("cpu")
    torch.set_num_threads(max(1, args.threads))

    print(f"Loading {checkpoint.name} on CPU ...", flush=True)
    model = IGLModel.load_from_checkpoint(
        checkpoint,
        map_location=device,
        init_from_config_only=True,
    )
    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(model.hparams.model_name, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": UNUSED_TOKENS})

    results = extract(
        sentences,
        model,
        tokenizer,
        top_k=args.top_k,
        batch_size=1,
        device=device,
        min_conf=args.min_confidence,
        max_length=args.max_length,
    )

    for sentence, triples in results:
        print(f"\n{sentence}")
        if not triples:
            print("  (no triples)")
            continue
        for confidence, subject, relation, argument in triples:
            print(
                f"  {confidence:.3f}  "
                f"({subject}; {relation}; {argument or '∅'})"
            )


if __name__ == "__main__":
    main()
