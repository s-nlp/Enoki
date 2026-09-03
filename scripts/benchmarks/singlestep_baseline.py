#!/usr/bin/env python3
"""
Single-step few-shot LLM baseline for hallucination detection
(RAGTruth / PsiloQA / Mushroom / HalluEntity).

Usage
-----
    export OPENAI_API_KEY=dummy
    export OPENAI_BASE_URL=http://localhost:8000/v1

    # Span-level datasets (one at a time, or --all-datasets for all three)
    python scripts/benchmarks/singlestep_baseline.py --dataset ragtruth \
        --model Qwen/Qwen3.6-35B-A3B --model-params 3e9 \
        --output predictions/singlestep_ragtruth.jsonl --workers 8

    python scripts/benchmarks/singlestep_baseline.py --all-datasets \
        --model Qwen/Qwen3.6-35B-A3B --model-params 3e9 \
        --output predictions/singlestep.jsonl --workers 8

    # Entity-level (HalluEntity) — extract-and-align onto the dataset's own
    # entity list, scored with AUROC/AUPRC instead of span F1.
    python scripts/benchmarks/singlestep_baseline.py --dataset halluentity \
        --model Qwen/Qwen3.6-35B-A3B --model-params 3e9 \
        --output predictions/singlestep_halluentity.jsonl --workers 8

    # Re-score an existing output file without re-calling the API:
    python scripts/benchmarks/singlestep_baseline.py --score-only --dataset halluentity \
        --output predictions/singlestep_halluentity.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import re as _re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_extractor.llm_backend import (
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
# Prompt (shared by span- and entity-level datasets)
# -----------------------------
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
# Model call
# -----------------------------

@dataclass(frozen=True)
class ModelCallResult:
    raw: str
    call_time_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    flops: float


_thread_local_client: Dict[int, Any] = {}


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
    enable_thinking: bool = False,
    max_tokens: Optional[int] = None,
) -> ModelCallResult:
    t0 = time.perf_counter()
    request_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": temperature,
        "timeout": request_timeout,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    }
    if max_tokens is not None:
        request_kwargs["max_tokens"] = max_tokens
    resp = client.chat.completions.create(**request_kwargs)
    call_time_s = time.perf_counter() - t0

    usage = getattr(resp, "usage", None)
    prompt_tokens = get_usage_token_count(usage, "prompt_tokens")
    completion_tokens = get_usage_token_count(usage, "completion_tokens")
    total_tokens = get_usage_token_count(usage, "total_tokens")
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens

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
    enable_thinking: bool = False,
    max_tokens: Optional[int] = None,
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
                enable_thinking=enable_thinking,
                max_tokens=max_tokens,
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

def _try_json_loads(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        pass

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
        return []

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
# Row/sample processing — span datasets vs. HalluEntity (entity-level)
# -----------------------------

def process_span_row(
    row: Dict[str, Any],
    *,
    model: str,
    temperature: float,
    max_retries: int,
    request_timeout: float,
    model_params: float,
    enable_thinking: bool = False,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    question = row.get("question", "")
    passages = row["context"]
    answer = row["answer"]

    user_prompt = build_user_prompt(question, passages, answer)
    result = call_model_with_retries(
        model=model, user_prompt=user_prompt, temperature=temperature,
        max_retries=max_retries, request_timeout=request_timeout,
        model_params=model_params, enable_thinking=enable_thinking, max_tokens=max_tokens,
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


def process_entity_sample(
    sample: Dict[str, Any],
    *,
    model: str,
    temperature: float,
    max_retries: int,
    request_timeout: float,
    model_params: float,
    enable_thinking: bool = False,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    # Extract-and-align design: reuse the same span-extraction prompt/parsing
    # above, then project predicted spans onto HalluEntity's own entity list
    # via the same alignment function Enoki itself uses. Per-entity score is
    # binary (1.0/0.0 — presence in the predicted list), not a continuous
    # confidence, since a single zero-shot call has no such signal.
    from evaluation.fact_alignment import align_fact_scores_to_entities_orig

    question = sample.get("prompt", "") or ""
    passages = sample["context"]
    answer = sample["text"]
    entities = sample["entities"]
    entity_labels = sample["entity_labels"]

    user_prompt = build_user_prompt(question, passages, answer)
    result = call_model_with_retries(
        model=model, user_prompt=user_prompt, temperature=temperature,
        max_retries=max_retries, request_timeout=request_timeout,
        model_params=model_params, enable_thinking=enable_thinking, max_tokens=max_tokens,
    )
    span_texts = parse_hallucination_json(result.raw)
    pred_spans = locate_spans(answer, span_texts)

    fact_scores_norm = [
        {
            "orig_span_start": s, "orig_span_end": e,
            "hall_prob": 1.0, "fact": answer[s:e], "span_kind": "argument",
        }
        for s, e in pred_spans
    ]
    if fact_scores_norm:
        y_score = align_fact_scores_to_entities_orig(
            orig_text=answer, ds_entities=entities, fact_scores_norm=fact_scores_norm,
        ).tolist()
    else:
        y_score = [0.0] * len(entities)

    # Same convention as evaluation/entity.py: True (factual) -> 0, False (hallucination) -> 1.
    y_true = [0 if label else 1 for label in entity_labels]

    return {
        "id": sample.get("id"),
        "gold_labels": y_true,
        "entity_scores": y_score,
        "n_pred_spans": len(pred_spans),
        "raw_output": result.raw,
        "call_time_s": result.call_time_s,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "flops": result.flops,
    }


# -----------------------------
# Dataset dispatch
# -----------------------------

SPAN_DATASETS = ("ragtruth", "psiloqa", "mushroom")
ALL_DATASETS = SPAN_DATASETS + ("halluentity",)


def load_dataset_rows(dataset: str, *, split: str, data_dir: str) -> List[Dict[str, Any]]:
    if dataset == "ragtruth":
        from evaluation.dataset_loaders import load_ragtruth_dataset
        print(f"Loading RAGTruth QA ({split} split)...", file=sys.stderr)
        return load_ragtruth_dataset(split=split)
    if dataset == "psiloqa":
        from evaluation.dataset_loaders import load_psiloqa_dataset
        print(f"Loading PsiloQA ({split} split)...", file=sys.stderr)
        return load_psiloqa_dataset(split=split)
    if dataset == "mushroom":
        from evaluation.dataset_loaders import load_mushroom_dataset
        print(f"Loading Mushroom (data_dir={data_dir})...", file=sys.stderr)
        return load_mushroom_dataset(data_dir=data_dir)
    if dataset == "halluentity":
        from evaluation.dataset_loaders import load_halluentity_dataset
        print(f"Loading HalluEntity (data_dir={data_dir})...", file=sys.stderr)
        samples, _eval_type = load_halluentity_dataset(data_dir=data_dir)
        return samples
    raise ValueError(f"Unknown dataset: {dataset!r}; choose from {ALL_DATASETS}")


def dataset_output_path(base_output: Path, dataset: str, *, suffix_per_dataset: bool) -> Path:
    if not suffix_per_dataset:
        return base_output
    return base_output.with_name(f"{base_output.stem}_{dataset}{base_output.suffix}")


# -----------------------------
# Run / score
# -----------------------------

def run(args: argparse.Namespace, *, dataset: str, out_path: Path) -> None:
    is_entity = dataset == "halluentity"
    data = load_dataset_rows(dataset, split=args.split, data_dir=args.data_dir)
    if args.limit:
        data = data[: args.limit]
    print(f"Loaded {len(data)} examples", file=sys.stderr)

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
    process_fn = process_entity_sample if is_entity else process_span_row

    mode = "a" if (out_path.exists() and not args.no_resume) else "w"
    with open(out_path, mode, encoding="utf-8") as out_f:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(
                    process_fn,
                    row,
                    model=args.model,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                    request_timeout=args.request_timeout,
                    model_params=args.model_params,
                    enable_thinking=args.enable_thinking,
                    max_tokens=args.max_tokens,
                ): row
                for row in todo
            }
            n_errors = 0
            total_call_time = 0.0
            total_tokens = 0
            pbar = tqdm(
                cf.as_completed(futures),
                total=len(todo),
                desc=f"Single-step baseline ({dataset})",
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


def score(out_path: Path, *, dataset: str) -> None:
    is_entity = dataset == "halluentity"
    call_times, flops_list, total_tokens_list = [], [], []

    if is_entity:
        from evaluation.metrics import calculate_entity_metrics, print_entity_metrics_summary
        auroc_list, auprc_list = [], []
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                auroc, auprc = calculate_entity_metrics(rec["gold_labels"], rec["entity_scores"])
                auroc_list.append(auroc)
                auprc_list.append(auprc)
                call_times.append(rec["call_time_s"])
                flops_list.append(rec["flops"])
                total_tokens_list.append(rec["total_tokens"])
        n = len(auroc_list)
        print("=" * 60)
        print(f"Single-step baseline — {dataset} ({n} examples)")
        print("=" * 60)
        print_entity_metrics_summary(auroc_list, auprc_list)
        print("(Note: per-entity scores are binary, not continuous — see process_entity_sample docstring.)")
    else:
        from evaluation.metrics import calculate_span_f1
        golds, preds = [], []
        with open(out_path, "r", encoding="utf-8") as f:
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
        print(f"Single-step baseline — {dataset} ({n} examples)")
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, help="Output/predictions JSONL path (base name; --all-datasets suffixes it per dataset)")
    p.add_argument("--dataset", default="ragtruth", choices=list(ALL_DATASETS))
    p.add_argument(
        "--all-datasets", action="store_true",
        help="Run all three SPAN-level datasets (ragtruth, psiloqa, mushroom). "
             "HalluEntity is entity-level and must be run separately (--dataset halluentity).",
    )
    p.add_argument("--split", default="test", help="HF split for ragtruth/psiloqa; ignored for mushroom/halluentity")
    p.add_argument("--data-dir", default="data", help="Base dir for mushroom/halluentity data files")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B", help="Model name passed to the API")
    p.add_argument("--model-params", type=float, default=3e9,
                   help="Active parameter count for FLOPs approx (2*P*T). Use ACTIVE params for MoE models.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=4, help="Parallel in-flight requests")
    p.add_argument("--max-retries", type=int, default=6)
    p.add_argument("--request-timeout", type=float, default=120.0)
    p.add_argument("--limit", type=int, default=None, help="Limit to first N examples (debugging)")
    p.add_argument("--no-resume", action="store_true", help="Do not skip already-written ids")
    p.add_argument("--score-only", action="store_true", help="Skip inference, only score an existing output file")
    p.add_argument("--enable-thinking", action="store_true",
                   help="Let hybrid-thinking models emit chain-of-thought before the JSON answer. OFF by default.")
    p.add_argument("--max-tokens", type=int, default=None, help="Cap generated tokens per call")
    args = p.parse_args()

    datasets_to_run = list(SPAN_DATASETS) if args.all_datasets else [args.dataset]
    base_output = Path(args.output)

    for ds in datasets_to_run:
        out_path = dataset_output_path(base_output, ds, suffix_per_dataset=args.all_datasets)
        if len(datasets_to_run) > 1:
            print(f"\n{'#' * 60}\n# Dataset: {ds}  ->  {out_path}\n{'#' * 60}", file=sys.stderr)
        try:
            if not args.score_only:
                run(args, dataset=ds, out_path=out_path)
            score(out_path, dataset=ds)
        except FileNotFoundError as e:
            print(f"[SKIP] {ds}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
