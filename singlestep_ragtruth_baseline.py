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

NOTE ON THE PROMPT: this uses the exact "ZS RAGTruth Prompt" text supplied
for the rebuttal (matches the published RAGTruth zero-shot span-extraction
prompt) verbatim, as a single zero-shot user turn — no system prompt, no
few-shot examples. Output is a JSON dict with key "hallucination list"
whose value is a list of verbatim hallucinated substrings copied from the
answer (empty list if none).

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

from tqdm import tqdm

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
#
# Exact "ZS RAGTruth Prompt" text (zero-shot, single user turn, no system
# message, no few-shot examples). {question}/{passages}/{answer} are filled
# in per example. Do not reformat/rephrase this — it's the literal text
# agreed for the rebuttal.

PROMPT_TEMPLATE = """Below is a question:
{question}
Below are related passages:
{passages}
Below is an answer:
{answer}
Your task is to determine whether the answer contains either or both of the following two types of hallucinations:
1. conflict: instances where the answer presents direct contraction or opposition to the passages;
2. baseless info: instances where the answer includes information which is not substantiated by or inferred from the passages.
Then, compile the labeled hallucinated spans into a JSON dict, with a key "hallucination list" and its value is a list of
hallucinated spans. If there exist potential hallucinations, the output should be in the following JSON format: {{"hallucination
list": [hallucination span1, hallucination span2, ...]}}. Otherwise, leave the value as a empty list as following: {{"hallucination
list": []}}.
Output:"""


def build_user_prompt(question: str, passages: str, answer: str) -> str:
    return PROMPT_TEMPLATE.format(question=question, passages=passages, answer=answer)


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
        # Single zero-shot user turn, no system message — matches the exact
        # RAGTruth prompt text verbatim (it's a complete self-contained
        # instruction, not meant to be split across a system/user pair).
        messages=[
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
#
# Model is asked for {"hallucination list": [span1, span2, ...]}. Real model
# output is rarely perfectly clean JSON (markdown code fences, leading/trailing
# prose, trailing commas, smart quotes, etc.), so this parses defensively:
#   1. try the whole response as JSON,
#   2. else try the largest {...} substring as JSON,
#   3. else regex out the array following "hallucination list" and json-load
#      just that array (also tolerating single quotes / trailing commas),
#   4. else give up and return no spans (better than crashing the whole run).

import re as _re


def _try_json_loads(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    # Repair: models sometimes emit a raw (unescaped) newline/tab inside a
    # JSON string value when a hallucinated span spans multiple lines/
    # sentences — that's invalid JSON and would otherwise zero out every
    # span for the example. Escape control chars found strictly inside
    # quoted string literals (respecting existing backslash-escapes) and
    # retry once.
    def _escape_inner(m: "_re.Match") -> str:
        s = m.group(0)
        return s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

    repaired = _re.sub(r'"(?:\\.|[^"\\])*"', _escape_inner, text)
    if repaired != text:
        try:
            return json.loads(repaired)
        except Exception:
            return None
    return None


def parse_hallucination_json(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return []

    # Strip ```json ... ``` / ``` ... ``` fences if present.
    fence_match = _re.search(r"```(?:json)?\s*(.*?)```", raw, flags=_re.DOTALL | _re.IGNORECASE)
    candidates = [raw]
    if fence_match:
        candidates.insert(0, fence_match.group(1).strip())

    parsed_dict: Optional[Dict[str, Any]] = None
    for cand in candidates:
        obj = _try_json_loads(cand)
        if isinstance(obj, dict):
            parsed_dict = obj
            break
        # Largest {...} substring (handles leading/trailing prose around the dict).
        start = cand.find("{")
        end = cand.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = _try_json_loads(cand[start:end + 1])
            if isinstance(obj, dict):
                parsed_dict = obj
                break

    def _clean_spans(items: List[Any]) -> List[str]:
        out = []
        for v in items:
            s = normalize_space(str(v))
            if s and s.upper() != "NONE":
                out.append(s)
        return out

    if parsed_dict is not None:
        for key, value in parsed_dict.items():
            if isinstance(key, str) and key.strip().lower() == "hallucination list":
                if isinstance(value, list):
                    return _clean_spans(value)
                return []
        # Dict parsed but key not found under the exact name -> no spans.
        return []

    # Last-resort regex fallback: pull out the array after "hallucination list"
    # and json-load just that (covers minor malformed-dict cases the block
    # above couldn't recover, e.g. unbalanced braces from truncated output).
    m = _re.search(r'"hallucination[\s\S]{0,3}list"\s*:\s*(\[[\s\S]*?\])', raw, flags=_re.IGNORECASE)
    if m:
        arr = _try_json_loads(m.group(1))
        if isinstance(arr, list):
            return _clean_spans(arr)

    return []


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
    # NOTE: 'question' relies on the load_ragtruth_dataset fix (evaluation/
    # dataset_loaders.py) that surfaces the HF dataset's 'query' column as
    # 'question'. Before that fix this was always "" (row never had the key).
    question = row.get("question", "")
    passages = row["context"]
    answer = row["answer"]

    user_prompt = build_user_prompt(question, passages, answer)

    result = call_model_with_retries(
        model=model,
        user_prompt=user_prompt,
        temperature=temperature,
        max_retries=max_retries,
        request_timeout=request_timeout,
        model_params=model_params,
    )

    span_texts = parse_hallucination_json(result.raw)
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
            n_errors = 0
            total_call_time = 0.0
            total_tokens = 0
            pbar = tqdm(
                cf.as_completed(futures),
                total=len(todo),
                desc="Single-step baseline",
                unit="ex",
                ncols=100,
                file=sys.stderr,
            )
            for fut in pbar:
                row = futures[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    n_errors += 1
                    pbar.write(f"[ERROR] id={row.get('id')}: {e}")
                    pbar.set_postfix({"errors": n_errors})
                    continue
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                total_call_time += rec["call_time_s"]
                total_tokens += rec["total_tokens"]
                n_ok = pbar.n + 1 - n_errors
                pbar.set_postfix({
                    "avg_s": f"{total_call_time / max(n_ok, 1):.2f}",
                    "avg_tok": f"{total_tokens / max(n_ok, 1):.0f}",
                    "errors": n_errors,
                })

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
