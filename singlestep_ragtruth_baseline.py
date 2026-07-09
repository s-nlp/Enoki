#!/usr/bin/env python3
"""
Single-step few-shot LLM baseline for RAGTruth span-level hallucination detection.

Purpose
-------
Reviewer requested an end-to-end latency comparison between ENOKI's three-stage
pipeline (Extraction + Verification + Mapping) and a modern single-step LLM
detector. This script is that single-step detector: for each RAGTruth QA
example, it issues ONE chat-completion call to an OpenAI-compatible endpoint
(e.g. a locally served vLLM Qwen model) with a few-shot prompt asking the
model to directly output the hallucinated substrings of the response. No
fact decomposition, no separate verification pass, no span-mapping step.

NOTE ON THE PROMPT: the original RAGTruth paper (Niu et al., 2024) and the
ENOKI paper's "ZS RAGTruth Prompt" row (Table 2/3, run with GPT-5.2) do not
publish the exact prompt text anywhere in this repo or in the ENOKI paper's
appendix, so the prompt below is our own few-shot reconstruction of that
task (extract hallucinated spans verbatim from a RAG response), not a
byte-for-byte reproduction of a specific prior prompt. Treat this as "a
modern single-step LLM detector, few-shot", consistent with what the
reviewer asked for.

Usage
-----
    # 1. Serve Qwen locally via vLLM's OpenAI-compatible server (separate terminal):
    #    vllm serve Qwen/Qwen3.6-35B-A3B --port 8000
    export OPENAI_API_KEY=dummy
    export OPENAI_BASE_URL=http://localhost:8000/v1

    python singlestep_ragtruth_baseline.py \
        --model Qwen/Qwen3.6-35B-A3B \
        --model-params 3e9 \
        --output predictions/singlestep_ragtruth.jsonl \
        --workers 4

    # Then compute Span Coverage F1 + latency/FLOPs summary:
    python singlestep_ragtruth_baseline.py --score-only \
        --output predictions/singlestep_ragtruth.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the already-battle-tested text-anchoring utilities and OpenAI-client
# plumbing from factextractor_backend.py so FLOPs/timing accounting is
# identical to the Enoki-LLM extraction path (2 * model_params * total_tokens).
from factextractor_backend import (
    find_text_span_in_window,
    trim_span,
    normalize_space,
    make_client,
    get_usage_token_count,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# -----------------------------
# Prompt
# -----------------------------

SYSTEM_PROMPT = """You are a hallucination detector for retrieval-augmented generation (RAG).

You will be given a QUESTION, a CONTEXT (retrieved source passages), and a RESPONSE
written by an AI assistant answering the QUESTION using the CONTEXT.

Task: identify every contiguous span of text in the RESPONSE that is NOT fully
supported by the CONTEXT. This includes:
  - Direct contradictions of facts stated in the CONTEXT.
  - Fabricated details (names, numbers, dates, places, quotes, claims) that do not
    appear in and cannot be inferred from the CONTEXT.
  - Unverifiable claims presented as fact that go beyond what the CONTEXT supports.

Do NOT flag:
  - Reasonable paraphrases or summaries of information that IS in the CONTEXT.
  - Generic connective / filler language that makes no factual claim.
  - Reasonable inferences that follow logically from the CONTEXT.

Output format:
  - Output ONLY the exact hallucinated substrings, copied verbatim (same characters,
    same casing, same punctuation) from the RESPONSE.
  - One span per line. Keep each span as short as possible while still covering the
    unsupported content (do not include whole supported sentences around it).
  - Preserve the order the spans appear in the RESPONSE.
  - If nothing in the RESPONSE is hallucinated, output exactly: NONE
  - Do not output explanations, labels, quotes, numbering, or any text other than
    the spans themselves (or NONE)."""

FEWSHOT: List[Dict[str, str]] = [
    {
        "question": "Where was Lucien Tesnière born?",
        "context": "Lucien Tesnière (13 May 1893 - 6 December 1954) was a French linguist. "
                   "He was born in Mont-Saint-Aignan, France.",
        "response": "Tesnière was born in Montpellier, France, in 1893. He became a well-known "
                    "French linguist.",
        "spans": ["Montpellier"],
    },
    {
        "question": "What programming paradigm does C# support?",
        "context": "C# is a general-purpose programming language. It was designed by "
                   "Anders Hejlsberg and supports multiple paradigms including imperative, "
                   "functional, generic, and object-oriented programming.",
        "response": "C# is a general-purpose programming language that supports object-oriented "
                    "and functional programming.",
        "spans": ["NONE"],
    },
    {
        "question": "How far back can collage techniques be traced?",
        "context": "Paper collage is first known to have existed in the 10th century in Japan, "
                   "when calligraphers began to apply glued paper, using texts on surfaces, "
                   "when writing their poems.",
        "response": "Collage techniques can be traced back much further than the invention of "
                    "paper. The earliest known collages are from the 5th century BC, in the form "
                    "of wall paintings in Pompeii. Later, paper collage emerged in Japan, where "
                    "calligraphers used glued paper when writing poems.",
        "spans": ["5th century BC, in the form of wall paintings in Pompeii"],
    },
]


def build_fewshot_block() -> str:
    blocks = []
    for ex in FEWSHOT:
        spans_text = "\n".join(ex["spans"])
        blocks.append(
            f"QUESTION: {ex['question']}\n"
            f"CONTEXT: {ex['context']}\n"
            f"RESPONSE: {ex['response']}\n"
            f"HALLUCINATED SPANS:\n{spans_text}"
        )
    return "\n\n---\n\n".join(blocks)


FEWSHOT_BLOCK = build_fewshot_block()


def build_user_prompt(question: str, context: str, response: str) -> str:
    return (
        f"Here are worked examples:\n\n{FEWSHOT_BLOCK}\n\n---\n\n"
        f"Now do the same for this case:\n\n"
        f"QUESTION: {question}\n"
        f"CONTEXT: {context}\n"
        f"RESPONSE: {response}\n"
        f"HALLUCINATED SPANS:"
    )


# -----------------------------
# Model call (mirrors factextractor_backend.call_model_once, but with a
# free-form user prompt instead of the fixed CycleOIE USER_PROMPT_TEMPLATE)
# -----------------------------

@dataclass(frozen=True)
class ModelCallResult:
    raw: str
    call_time_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    flops: float


_thread_local_client = {}


import threading


def get_thread_client():
    tid = threading.get_ident()
    client = _thread_local_client.get(tid)
    if client is None:
        client = make_client()
        _thread_local_client[tid] = client
    return client


def call_model_once(
    client: Any,
    *,
    model: str,
    user_prompt: str,
    temperature: float,
    request_timeout: float,
    model_params: float,
) -> ModelCallResult:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        timeout=request_timeout,
    )
    call_time_s = time.perf_counter() - t0

    usage = getattr(resp, "usage", None)
    prompt_tokens = get_usage_token_count(usage, "prompt_tokens")
    completion_tokens = get_usage_token_count(usage, "completion_tokens")
    total_tokens = get_usage_token_count(usage, "total_tokens")
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens

    # FLOPs ~ 2 * P * T, same approximation as Appendix G / factextractor_backend.py.
    # Pass --model-params with the model's ACTIVE parameter count for MoE models
    # (e.g. ~3e9 for Qwen3.6-35B-A3B), not the total parameter count, so this
    # number is comparable to Enoki-LLM's extraction FLOPs measured the same way.
    flops = 2.0 * float(model_params) * float(total_tokens) if total_tokens > 0 else 0.0

    return ModelCallResult(
        raw=resp.choices[0].message.content or "",
        call_time_s=call_time_s,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        flops=flops,
    )


def call_model_with_retries(
    *,
    model: str,
    user_prompt: str,
    temperature: float,
    max_retries: int,
    request_timeout: float,
    model_params: float,
) -> ModelCallResult:
    for attempt in range(max_retries + 1):
        try:
            client = get_thread_client()
            return call_model_once(
                client,
                model=model,
                user_prompt=user_prompt,
                temperature=temperature,
                request_timeout=request_timeout,
                model_params=model_params,
            )
        except Exception as e:
            if attempt >= max_retries:
                raise
            sleep_s = min(60.0, (2 ** attempt) + random.random())
            print(f"[retry] attempt={attempt + 1}/{max_retries} sleep={sleep_s:.1f}s "
                  f"err={type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(sleep_s)


# -----------------------------
# Output parsing -> char spans in `answer`
# -----------------------------

def parse_span_lines(raw: str) -> List[str]:
    lines = [normalize_space(l) for l in raw.splitlines()]
    lines = [l.strip(" -*•\"'“”") for l in lines if l.strip()]
    lines = [l for l in lines if l and l.upper() != "NONE"]
    return lines


def locate_spans(answer: str, span_texts: List[str]) -> List[List[int]]:
    """Anchor each predicted span string back to a [start, end) char offset in `answer`."""
    spans: List[List[int]] = []
    cursor = 0
    for text in span_texts:
        found = find_text_span_in_window(answer, text, window_start=cursor, window_end=len(answer))
        if found is None:
            found = find_text_span_in_window(answer, text, window_start=0, window_end=len(answer))
        if found is None:
            continue
        s, e = found
        s, e = trim_span(answer, s, e)
        if e > s:
            spans.append([s, e])
            cursor = e
    return spans


# -----------------------------
# Row processing
# -----------------------------

def process_row(
    row: Dict[str, Any],
    *,
    model: str,
    temperature: float,
    max_retries: int,
    request_timeout: float,
    model_params: float,
) -> Dict[str, Any]:
    question = row.get("question", "")
    context = row["context"]
    answer = row["answer"]

    user_prompt = build_user_prompt(question, context, answer)

    result = call_model_with_retries(
        model=model,
        user_prompt=user_prompt,
        temperature=temperature,
        max_retries=max_retries,
        request_timeout=request_timeout,
        model_params=model_params,
    )

    span_texts = parse_span_lines(result.raw)
    pred_spans = locate_spans(answer, span_texts)

    return {
        "id": row.get("id"),
        "gold": row["labels"],
        "pred": pred_spans,
        "raw_output": result.raw,
        "call_time_s": result.call_time_s,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "flops": result.flops,
    }


# -----------------------------
# Main
# -----------------------------

def run(args: argparse.Namespace) -> None:
    from evaluation.dataset_loaders import load_ragtruth_dataset

    print(f"Loading RAGTruth QA ({args.split} split)...", file=sys.stderr)
    data = load_ragtruth_dataset(split=args.split)
    if args.limit:
        data = data[: args.limit]
    print(f"Loaded {len(data)} examples", file=sys.stderr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if out_path.exists() and not args.no_resume:
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        if done_ids:
            print(f"Resuming: {len(done_ids)} rows already done", file=sys.stderr)

    todo = [row for row in data if row.get("id") not in done_ids]

    mode = "a" if (out_path.exists() and not args.no_resume) else "w"
    with open(out_path, mode, encoding="utf-8") as out_f:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(
                    process_row,
                    row,
                    model=args.model,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                    request_timeout=args.request_timeout,
                    model_params=args.model_params,
                ): row
                for row in todo
            }
            n_done = 0
            for fut in cf.as_completed(futures):
                row = futures[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    print(f"[ERROR] id={row.get('id')}: {e}", file=sys.stderr)
                    continue
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                n_done += 1
                if n_done % 10 == 0:
                    print(f"  {n_done}/{len(todo)} done", file=sys.stderr)

    print(f"Wrote predictions to {out_path}", file=sys.stderr)


def score(args: argparse.Namespace) -> None:
    from evaluation.metrics import calculate_span_f1

    golds, preds = [], []
    call_times, flops_list, total_tokens_list = [], [], []

    with open(args.output, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            golds.append(rec["gold"])
            preds.append(rec["pred"])
            call_times.append(rec["call_time_s"])
            flops_list.append(rec["flops"])
            total_tokens_list.append(rec["total_tokens"])

    metrics = calculate_span_f1(golds, preds)
    n = len(golds)
    print("=" * 60)
    print(f"Single-step baseline — RAGTruth QA ({n} examples)")
    print("=" * 60)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"Span Coverage F1: {metrics['f1']:.4f}")
    print("-" * 60)
    print(f"Avg latency / example:   {sum(call_times) / n:.3f} s")
    print(f"Avg total tokens / ex.:  {sum(total_tokens_list) / n:.1f}")
    print(f"Avg FLOPs / example:     {sum(flops_list) / n:.3e}")
    print(f"Total latency (sum):     {sum(call_times):.1f} s")
    print("=" * 60)
    print("NOTE: these are PER-EXAMPLE (whole RAGTruth response) numbers, since the "
          "single-step baseline makes one call per response. To compare against "
          "Enoki's PER-SENTENCE extract+verify timings from evaluate-span "
          "--save-facts output, sum Enoki's per-sentence times within each example "
          "first (see aggregate_latency.py).")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, help="Output/predictions JSONL path")
    p.add_argument("--split", default="test", help="RAGTruth HF split (test or train)")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B", help="Model name passed to the API")
    p.add_argument("--model-params", type=float, default=3e9,
                   help="Active parameter count for FLOPs approx (2*P*T). "
                        "Use ACTIVE params for MoE models, e.g. ~3e9 for Qwen3.6-35B-A3B.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=4, help="Parallel in-flight requests")
    p.add_argument("--max-retries", type=int, default=6)
    p.add_argument("--request-timeout", type=float, default=120.0)
    p.add_argument("--limit", type=int, default=None, help="Limit to first N examples (debugging)")
    p.add_argument("--no-resume", action="store_true", help="Do not skip already-written ids")
    p.add_argument("--score-only", action="store_true", help="Skip inference, only score an existing output file")
    args = p.parse_args()

    if args.score_only:
        score(args)
    else:
        run(args)
        score(args)


if __name__ == "__main__":
    main()
