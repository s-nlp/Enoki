#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-based filter for generated answers (hallucination and grounded).

Inputs: JSONL with at least question and answer; optional context_text.
Outputs: kept answers with filter_meta, and an errors file for dropped items.

Reasons (single token):
- ok
- refusal          (answer is a refusal/deferral/boilerplate)
- not_relevant     (does not address the question)
- other
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from openai import AsyncOpenAI

try:
    import orjson  # type: ignore

    def json_dumps(obj: Any) -> str:
        return orjson.dumps(obj).decode("utf-8")
except Exception:  # pragma: no cover

    def json_dumps(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False)


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


SYSTEM = """
You are a strict judge of answer quality for a QA dataset.
Decide if the ANSWER is acceptable for the QUESTION (optionally use CONTEXT if given).

Reject if:
- It is a refusal/deferral/boilerplate (“I cannot answer”, “as an AI”, “sorry”, etc.).
- It does not address the question (off-topic, missing the asked info).

Allowed reasons (exact tokens): ok, refusal, not_relevant, other.

Respond with JSON only:
{"keep": true|false, "reason": "<token>", "scores": {"relevance": 0-1}}
""".strip()

USER_TMPL = """
Question:
{question}

{context_block}
Answer:
{answer}
""".strip()


@dataclass(frozen=True)
class Config:
    input_path: Path
    out_path: Path
    errors_path: Path

    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    concurrency: int
    max_retries: int
    max_records: int


async def chat_json(client: AsyncOpenAI, cfg: Config, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    last_err = None
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
            await asyncio.sleep(min(60.0, 2.0**attempt) + random.random() * 0.25)
    raise last_err or RuntimeError("chat_json failed")


def build_messages(question: str, answer: str, context: str) -> List[Dict[str, str]]:
    context_block = f"Context:\n{context}\n" if context else "Context: (none provided)\n"
    user = USER_TMPL.format(question=question, answer=answer, context_block=context_block)
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]


async def rate_one(client: AsyncOpenAI, cfg: Config, rec: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    question = str(
        rec.get("question")
        or (rec.get("source") or {}).get("question")
        or rec.get("prompt")
        or ""
    ).strip()
    answer = str(
        rec.get("answer")
        or rec.get("response")
        or (rec.get("source") or {}).get("answer")
        or ""
    ).strip()
    context = str(
        rec.get("context_text")
        or rec.get("ref_text")
        or (rec.get("source") or {}).get("context_text")
        or ""
    ).strip()

    if not question or not answer:
        return False, {"reason": "other", "explain": "missing question or answer"}

    msgs = build_messages(question, answer, context)
    result = await chat_json(client, cfg, msgs)
    keep = bool(result.get("keep"))
    reason = str(result.get("reason") or ("ok" if keep else "other")).lower().replace(" ", "_")
    scores = result.get("scores") or {}
    meta = {"llm": result, "reason": reason, "scores": scores}
    return keep, meta


async def process(cfg: Config) -> None:
    client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=120.0)
    sem = asyncio.Semaphore(max(1, cfg.concurrency))

    produced = 0
    kept = 0
    dropped = 0
    started = time.time()

    async def handle(rec: Dict[str, Any]):
        nonlocal kept, dropped
        async with sem:
            try:
                keep, meta = await rate_one(client, cfg, rec)
            except Exception as e:
                keep = False
                meta = {"reason": "llm_error", "error": f"{type(e).__name__}: {e}"}

            if keep:
                kept += 1
                rec_out = dict(rec)
                rec_out["filter_meta"] = meta
                with cfg.out_path.open("a", encoding="utf-8") as f_out:
                    f_out.write(json_dumps(rec_out) + "\n")
            else:
                dropped += 1
                err_rec = {
                    "_error": True,
                    "stage": "filter",
                    "question": rec.get("question") or (rec.get("source") or {}).get("question"),
                    "answer": rec.get("answer") or rec.get("response") or (rec.get("source") or {}).get("answer"),
                    "meta": meta,
                }
                with cfg.errors_path.open("a", encoding="utf-8") as f_err:
                    f_err.write(json_dumps(err_rec) + "\n")

    tasks: List[asyncio.Task] = []
    for rec in iter_jsonl(cfg.input_path):
        if 0 < cfg.max_records <= produced:
            break
        tasks.append(asyncio.create_task(handle(rec)))
        produced += 1
        if produced and produced % 200 == 0:
            rate = produced / max(1e-6, time.time() - started)
            print(f"[progress] seen={produced} kept={kept} dropped={dropped} r={rate:.2f}/s")
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = max(1e-6, time.time() - started)
    print(f"[done] seen={produced} kept={kept} dropped={dropped} elapsed_s={elapsed:.1f} rate={produced/elapsed:.2f}/s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--errors-out", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--max-records", type=int, default=0)
    args = ap.parse_args()

    cfg = Config(
        input_path=Path(args.input),
        out_path=Path(args.out),
        errors_path=Path(args.errors_out),
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        concurrency=int(args.concurrency),
        max_retries=int(args.max_retries),
        max_records=int(args.max_records),
    )
    asyncio.run(process(cfg))


if __name__ == "__main__":
    main()
