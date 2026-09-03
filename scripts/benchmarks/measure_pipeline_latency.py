#!/usr/bin/env python3
"""
Measure end-to-end (extraction + verification) latency for an ENOKI pipeline

Usage
-----
Enoki-Rule (rule-based extraction, local, no GPU needed for extraction):
    python scripts/benchmarks/measure_pipeline_latency.py \
        --extractor-method enoki_rules --nli-method modernbert \
        --n-samples 200 --output predictions/latency_enoki_rule.csv

Enoki-Encoder (trained IGL checkpoint):
    python scripts/benchmarks/measure_pipeline_latency.py \
        --extractor-method enoki_encoder --checkpoint checkpoints/best.ckpt \
        --nli-method modernbert --n-samples 200 \
        --output predictions/latency_enoki_encoder.csv

Enoki-LLM (reads extraction timing already saved by
`enoki_cli.py extract-triplets --save-sentence-metrics`; only the verify
stage is timed here):
    python scripts/benchmarks/measure_pipeline_latency.py \
        --extractor-method cycleoie \
        --pre-extracted-facts-file data/pre_extracted/ragtruth_test.jsonl \
        --nli-method llm --vllm-model Qwen/Qwen3.6-35B-A3B \
        --n-samples 200 --output predictions/latency_enoki_llm.csv

To verify with the Qwen LLM verifier instead of ModernBERT, pass
--nli-method llm --vllm-model <hf repo id> (requires vLLM's OpenAI server or
the vllm package installed locally: nli.LLM_NLI loads it in-process).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factextractor_backend import split_sentences_with_spans


def count_claims(granular_facts) -> int:
    """Count leaf facts, handling both flat Fact lists and IncrementalFactGroup lists."""
    total = 0
    for item in granular_facts:
        total += len(item) if hasattr(item, "__len__") and hasattr(item, "facts") else 1
    return total


def load_preextracted_index(path: str) -> Dict[str, Dict[str, Any]]:
    import json
    index: Dict[str, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = rec.get("id") or rec.get("source_id")
            if rid is not None:
                index[str(rid)] = rec
    return index


DATASET_LABELS = {
    "ragtruth": "RAGTruth QA",
    "psiloqa": "PsiloQA",
    "mushroom": "Mushroom",
    "halluentity": "HalluEntity",
}


def _load_dataset(args: argparse.Namespace) -> List[Dict[str, Any]]:
    from evaluation.dataset_loaders import (
        load_ragtruth_dataset, load_psiloqa_dataset, load_mushroom_dataset,
        load_halluentity_dataset,
    )

    if args.dataset == "ragtruth":
        return load_ragtruth_dataset(split=args.split)
    if args.dataset == "psiloqa":
        return load_psiloqa_dataset(split=args.split)
    if args.dataset == "mushroom":
        return load_mushroom_dataset(data_dir=args.data_dir)
    if args.dataset == "halluentity":
        samples, _eval_type = load_halluentity_dataset(data_dir=args.data_dir)
        return [
            {"id": s["id"], "context": s["context"], "answer": s["text"]}
            for s in samples
        ]
    raise ValueError(f"Unknown --dataset {args.dataset!r}")


def run(args: argparse.Namespace) -> None:
    from nli import check_nli_batch_fast, score_facts_with_nli, score_preextracted_with_nli, get_nli_checker

    dataset_label = DATASET_LABELS.get(args.dataset, args.dataset)
    print(f"Loading {dataset_label} ({args.split} split)...", file=sys.stderr)
    data = _load_dataset(args)
    if args.n_samples:
        data = data[: args.n_samples]
    print(f"Using {len(data)} examples", file=sys.stderr)

    # Pre-warm the NLI verifier once so model loading time isn't counted as verify latency.
    nli_kwargs = {}
    if args.nli_method == "llm" and args.vllm_model:
        nli_kwargs["model"] = args.vllm_model
    print(f"Loading NLI verifier ({args.nli_method})...", file=sys.stderr)
    get_nli_checker(args.nli_method, **nli_kwargs)

    rows_out: List[Dict[str, Any]] = []

    if args.extractor_method == "cycleoie":
        if not args.pre_extracted_facts_file:
            raise ValueError("--pre-extracted-facts-file is required for --extractor-method cycleoie")
        pre_index = load_preextracted_index(args.pre_extracted_facts_file)

        for row in data:
            rid = str(row.get("id"))
            rec = pre_index.get(rid)
            if rec is None:
                print(f"[skip] no pre-extracted facts for id={rid}", file=sys.stderr)
                continue

            triplets = rec.get("triplets", [])
            spans = rec.get("spans", [])
            triplet_span_pairs: List[Tuple[List[str], List[int]]] = list(zip(triplets, spans))
            n_sentences = rec.get("extract_sentence_count") or len(
                split_sentences_with_spans(row["answer"])
            )
            extract_time_s = float(rec.get("extract_time_s", 0.0))
            extract_flops = float(rec.get("extract_flops", 0.0))

            t0 = time.perf_counter()
            if triplet_span_pairs:
                score_preextracted_with_nli(
                    context=row["context"],
                    triplet_span_pairs=triplet_span_pairs,
                    check_nli_batch_fn=partial(
                        check_nli_batch_fast, max_length=args.max_length, method=args.nli_method
                    ),
                    answer=row["answer"],
                )
            verify_time_s = time.perf_counter() - t0

            rows_out.append({
                "id": rid,
                "n_sentences": n_sentences,
                "n_claims": len(triplet_span_pairs),
                "extract_time_s": extract_time_s,
                "extract_flops": extract_flops,
                "verify_time_s": verify_time_s,
                "total_time_s": extract_time_s + verify_time_s,
            })

    else:
        from evaluation.common import load_fact_extractor

        print(f"Loading {args.extractor_method} extractor...", file=sys.stderr)
        extractor = load_fact_extractor(
            extractor_method=args.extractor_method,
            checkpoint=args.checkpoint,
            use_preprocessing=False,
        )

        # Eapproximate FLOPs the same way:
        # i.e. 2 * params * tokens, single forward pass (no autoregressive decode).
        # Enoki-Rule (EnokiRulesFactExtractor) has no `.model`/`.tokenizer` — this
        # stays 0.0 for it, which is correct (no neural net in extraction).
        encoder_params = 0
        if hasattr(extractor, "model") and hasattr(extractor, "tokenizer"):
            encoder_params = sum(p.numel() for p in extractor.model.parameters())
            print(f"Encoder extractor parameters: {encoder_params:,}", file=sys.stderr)

        def _count_encoder_tokens(text: str) -> int:
            if encoder_params == 0 or not text or not text.strip():
                return 0
            from fact_extractor.enoki_encoder_extractor import UNUSED_TOKENS
            import nltk
            total = 0
            for sent in extractor.nlp(text).sents:
                sent_text = sent.text.strip()
                if not sent_text:
                    continue
                words = nltk.word_tokenize(sent_text) + UNUSED_TOKENS
                enc = extractor.tokenizer(
                    [words], is_split_into_words=True,
                    truncation=True, max_length=extractor.max_length,
                )
                total += len(enc["input_ids"][0])
            return total

        for row in data:
            n_sentences = len(split_sentences_with_spans(row["answer"])) or 1

            t0 = time.perf_counter()
            try:
                granular_facts = extractor.extract_granular_facts(row["answer"])
            except Exception as e:
                print(f"[warn] extraction failed for id={row.get('id')}: {e}", file=sys.stderr)
                granular_facts = []
            extract_time_s = time.perf_counter() - t0

            extract_tokens = _count_encoder_tokens(row["answer"]) if encoder_params else 0
            extract_flops = 2.0 * encoder_params * extract_tokens if encoder_params else 0.0

            t0 = time.perf_counter()
            if granular_facts:
                score_facts_with_nli(
                    context=row["context"],
                    granular_facts=granular_facts,
                    check_nli_batch_fn=partial(
                        check_nli_batch_fast, max_length=args.max_length, method=args.nli_method
                    ),
                    chunk_size=32,
                )
            verify_time_s = time.perf_counter() - t0

            rows_out.append({
                "id": row.get("id"),
                "n_sentences": n_sentences,
                "n_claims": count_claims(granular_facts),
                "extract_time_s": extract_time_s,
                "extract_flops": extract_flops,
                "verify_time_s": verify_time_s,
                "total_time_s": extract_time_s + verify_time_s,
            })

    if not rows_out:
        print("No rows measured — nothing to report.", file=sys.stderr)
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Wrote per-example latency to {out_path}", file=sys.stderr)

    total_sentences = sum(r["n_sentences"] for r in rows_out)
    total_claims = sum(r["n_claims"] for r in rows_out)
    total_extract = sum(r["extract_time_s"] for r in rows_out)
    total_verify = sum(r["verify_time_s"] for r in rows_out)
    total_flops = sum(r["extract_flops"] for r in rows_out)
    n = len(rows_out)

    print("=" * 70)
    print(f"{args.extractor_method} / {args.nli_method} verifier — {dataset_label} ({n} examples, "
          f"{total_sentences} sentences)")
    print("=" * 70)
    print(f"Avg claims / sentence:        {total_claims / total_sentences:.2f}")
    print(f"Avg extract_time / sentence:  {total_extract / total_sentences:.4f} s")
    print(f"Avg verify_time  / sentence:  {total_verify / total_sentences:.4f} s")
    print(f"Avg total_time   / sentence:  {(total_extract + total_verify) / total_sentences:.4f} s")
    print(f"Avg extract FLOPs / sentence: {total_flops / total_sentences:.3e}")
    print("-" * 70)
    print(f"Avg extract_time / example:   {total_extract / n:.4f} s")
    print(f"Avg verify_time  / example:   {total_verify / n:.4f} s")
    print(f"Avg total_time   / example:   {(total_extract + total_verify) / n:.4f} s")
    print("=" * 70)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extractor-method", required=True, choices=["enoki_rules", "enoki_encoder", "cycleoie"])
    p.add_argument("--dataset", default="ragtruth",
                    choices=["ragtruth", "psiloqa", "mushroom", "halluentity"],
                    help="Which dataset to measure on")
    p.add_argument("--nli-method", default="modernbert",
                    choices=["modernbert", "alignscore", "qwen_06b", "qwen_4b", "qwen_8b", "llm"])
    p.add_argument("--vllm-model", default=None, help="HF repo id, used when --nli-method llm")
    p.add_argument("--checkpoint", default=None, help="Required for --extractor-method enoki_encoder")
    p.add_argument("--pre-extracted-facts-file", default=None,
                    help="Required for --extractor-method cycleoie (output of extract-triplets)")
    p.add_argument("--split", default="test")
    p.add_argument("--data-dir", default="data",
                    help="Base dir for --dataset mushroom/halluentity (default: data)")
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--max-length", type=int, default=8000)
    p.add_argument("--output", required=True, help="Per-example CSV output path")
    args = p.parse_args()

    if args.extractor_method == "enoki_encoder" and not args.checkpoint:
        raise SystemExit("--checkpoint is required for --extractor-method enoki_encoder")
    if args.extractor_method == "cycleoie" and not args.pre_extracted_facts_file:
        raise SystemExit("--pre-extracted-facts-file is required for --extractor-method cycleoie")

    run(args)


if __name__ == "__main__":
    main()
