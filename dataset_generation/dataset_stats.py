#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataset_stats.py

Collects dataset stats (length distributions, NER counts) for generated QA data.
Uses the same OpenAI-compatible model (e.g., gpt-oss) to extract named entities
from question/context/answer triples.

Input shapes supported:
- Context-level (qa_gen_fast.py): multiple questions + context_text.
- Per-question with answers (hallucination_answerer.py): question + answer + context_text in source.

Outputs:
- JSON summary with length percentiles and NER counts.
- Optional PNG histograms if matplotlib is available and --plot-prefix is set.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openai import AsyncOpenAI

try:
    import orjson  # type: ignore

    def json_dumps(obj: Any) -> str:
        return orjson.dumps(obj).decode("utf-8")
except Exception:  # pragma: no cover

    def json_dumps(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False)


# ---------- I/O ----------
def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json_dumps(obj))


# ---------- prompts ----------
NER_SYSTEM = """
You are a precise named-entity recognizer.
Extract distinct entities with coarse types: PERSON, ORG, GPE, LOC, EVENT, WORK, DATE, NUMBER, OTHER.
Return compact JSON only.
""".strip()

NER_USER_TEMPLATE = """
Extract entities from the following fields.
Return JSON: {{"question_entities":[...], "context_entities":[...], "answer_entities":[...]}}
Each entity: {{"text": "...", "type": "PERSON|ORG|GPE|LOC|EVENT|WORK|DATE|NUMBER|OTHER"}}

Question:
{question}

Context:
{context}

Answer:
{answer}
""".strip()


def build_ner_messages(
    question: str, context: str, answer: str
) -> List[Dict[str, str]]:
    user = NER_USER_TEMPLATE.format(question=question, context=context, answer=answer)
    return [
        {"role": "system", "content": NER_SYSTEM},
        {"role": "user", "content": user},
    ]


# ---------- config ----------
@dataclass(frozen=True)
class Config:
    input_path: Path
    out_json: Path
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    concurrency: int
    max_retries: int
    max_records: int  # 0 = all
    max_text_chars: int
    plot_prefix: Optional[str]
    seed: int


# ---------- openai helper ----------
async def chat_json(
    client: AsyncOpenAI, cfg: Config, messages: List[Dict[str, str]]
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            resp = await client.chat.completions.create(
                model=cfg.model,
                messages=messages,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                n=1,
            )
            msg = resp.choices[0].message.content if resp.choices else ""
            if not msg:
                raise ValueError("empty response")
            return json.loads(msg)
        except Exception as e:
            last_err = e
            sleep_s = min(60.0, 2.0**attempt) + random.random() * 0.3
            await asyncio.sleep(sleep_s)
    raise last_err or RuntimeError("chat_json failed")


# ---------- stats helpers ----------
def summary(nums: List[int]) -> Dict[str, Any]:
    if not nums:
        return {"count": 0}
    nums_sorted = sorted(nums)

    def pct(p: float) -> int:
        idx = min(len(nums_sorted) - 1, int(p * (len(nums_sorted) - 1)))
        return nums_sorted[idx]

    return {
        "count": len(nums),
        "mean": statistics.mean(nums),
        "median": statistics.median(nums),
        "p90": pct(0.9),
        "p99": pct(0.99),
        "min": nums_sorted[0],
        "max": nums_sorted[-1],
    }


def normalize_entities(obj: Any) -> List[Dict[str, str]]:
    out = []
    if not isinstance(obj, list):
        return out
    for it in obj:
        if not isinstance(it, dict):
            continue
        text = str(it.get("text") or "").strip()
        etype = str(it.get("type") or "OTHER").upper()
        if text:
            out.append({"text": text, "type": etype})
    return out


def detect_has_answers(path: Path) -> bool:
    for rec in iter_jsonl(path):
        if "answer" in rec or (rec.get("source") or {}).get("answer"):
            return True
        return False
    return False


# ---------- plotting ----------
def maybe_plot_hist(data: List[int], title: str, path: Path, bins: int = 50) -> None:
    if not data:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print(f"[warn] matplotlib not available; skip plot {path}")
        return
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins, color="#1f77b4", alpha=0.8)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ---------- processing ----------
async def run(cfg: Config) -> None:
    client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=120.0)
    sem = asyncio.Semaphore(max(1, cfg.concurrency))

    has_answers = detect_has_answers(cfg.input_path)

    lengths = defaultdict(list)  # field -> List[int]
    ent_counts = defaultdict(list)  # field -> List[int]
    ent_types = defaultdict(Counter)  # field -> Counter

    produced = 0
    started = time.time()

    async def handle(triple: Dict[str, str]):
        question = triple.get("question", "")
        context = triple.get("context", "")
        answer = triple.get("answer", "")

        lengths["question"].append(len(question))
        lengths["context"].append(len(context))
        lengths["answer"].append(len(answer))

        context_trunc = (
            context[: cfg.max_text_chars] if cfg.max_text_chars > 0 else context
        )

        msgs = build_ner_messages(question, context_trunc, answer)
        async with sem:
            try:
                resp = await chat_json(client, cfg, msgs)
            except Exception as e:
                print(f"[warn] NER failed: {type(e).__name__}: {e}")
                return

        q_ents = normalize_entities(resp.get("question_entities"))
        c_ents = normalize_entities(resp.get("context_entities"))
        a_ents = normalize_entities(resp.get("answer_entities"))

        ent_counts["question"].append(len(q_ents))
        ent_counts["context"].append(len(c_ents))
        ent_counts["answer"].append(len(a_ents))

        for ent in q_ents:
            ent_types["question"][ent["type"]] += 1
        for ent in c_ents:
            ent_types["context"][ent["type"]] += 1
        for ent in a_ents:
            ent_types["answer"][ent["type"]] += 1

    tasks: List[asyncio.Task] = []

    for rec in iter_jsonl(cfg.input_path):
        if 0 < cfg.max_records <= produced:
            break

        context_text = str(
            rec.get("context_text")
            or rec.get("context")
            or (rec.get("source", {}) or {}).get("context_text")
            or ""
        )

        if has_answers:
            question = str(rec.get("question") or "")
            answer = str(rec.get("answer") or "")
            tasks.append(
                asyncio.create_task(
                    handle(
                        {
                            "question": question,
                            "context": context_text,
                            "answer": answer,
                        }
                    )
                )
            )
            produced += 1
        else:
            questions = rec.get("questions_filtered") or rec.get("questions") or []
            if not isinstance(questions, list):
                continue
            for q in questions:
                if 0 < cfg.max_records <= produced:
                    break
                tasks.append(
                    asyncio.create_task(
                        handle(
                            {"question": str(q), "context": context_text, "answer": ""}
                        )
                    )
                )
                produced += 1

        if len(tasks) >= cfg.concurrency * 20:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            tasks = [t for t in tasks if not t.done()]

        if produced and produced % 200 == 0:
            rate = produced / max(1e-6, time.time() - started)
            print(f"[progress] processed={produced} rate={rate:.2f}/s")

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    summary_obj: Dict[str, Any] = {
        "total_examples": produced,
        "lengths": {k: summary(v) for k, v in lengths.items()},
        "entity_counts": {k: summary(v) for k, v in ent_counts.items()},
        "entity_types": {k: dict(v) for k, v in ent_types.items()},
    }

    write_json(cfg.out_json, summary_obj)
    print(f"[done] wrote {cfg.out_json}")

    if cfg.plot_prefix:
        prefix = Path(cfg.plot_prefix)
        maybe_plot_hist(
            lengths["question"],
            "Question length (chars)",
            prefix.with_suffix(".q_len.png"),
        )
        maybe_plot_hist(
            lengths["context"],
            "Context length (chars)",
            prefix.with_suffix(".ctx_len.png"),
        )
        maybe_plot_hist(
            lengths["answer"],
            "Answer length (chars)",
            prefix.with_suffix(".ans_len.png"),
        )
        maybe_plot_hist(
            ent_counts["question"],
            "Question entity count",
            prefix.with_suffix(".q_ent.png"),
        )
        maybe_plot_hist(
            ent_counts["context"],
            "Context entity count",
            prefix.with_suffix(".ctx_ent.png"),
        )
        maybe_plot_hist(
            ent_counts["answer"],
            "Answer entity count",
            prefix.with_suffix(".ans_ent.png"),
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", required=True, help="Input JSONL (filtered questions or answers)"
    )
    ap.add_argument("--out-json", required=True, help="Where to write stats JSON")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--max-records", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--max-text-chars", type=int, default=6000, help="Truncate context before NER"
    )
    ap.add_argument(
        "--plot-prefix",
        default=None,
        help="If set, save PNG histograms with this prefix",
    )
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    cfg = Config(
        input_path=Path(args.input),
        out_json=Path(args.out_json),
        base_url=str(args.base_url),
        api_key=str(args.api_key),
        model=str(args.model),
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        concurrency=int(args.concurrency),
        max_retries=int(args.max_retries),
        max_records=int(args.max_records),
        max_text_chars=int(args.max_text_chars),
        plot_prefix=args.plot_prefix,
        seed=int(args.seed),
    )

    random.seed(cfg.seed)
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
