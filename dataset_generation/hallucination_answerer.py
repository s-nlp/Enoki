#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openai import AsyncOpenAI


try:
    import orjson  # type: ignore

    def json_dumps(obj: Any) -> str:
        return orjson.dumps(obj).decode("utf-8")
except Exception:

    def json_dumps(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False)


def now_utc_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


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


async def append_jsonl_async(path: Path, obj: Dict[str, Any]) -> None:
    # Single-writer coroutine should call this to avoid file races
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json_dumps(obj) + "\n")


def safe_list(x) -> List[Any]:
    return x if isinstance(x, list) else []


@dataclass(frozen=True)
class Config:
    input_path: Path
    out_path: Path
    errors_path: Path

    base_url: str
    api_key: str
    model: str

    concurrency: int
    max_records: int  # 0 = unlimited
    n_samples: int

    temperature: float
    top_p: Optional[float]
    max_tokens: int

    prompt_mode: str  # no_context|with_context
    force_confident: bool

    max_retries: int
    timeout_s: float

    seed: int


def build_messages(
    cfg: Config, *, question: str, context_text: str
) -> List[Dict[str, str]]:
    if cfg.force_confident:
        system = (
            "You are a helpful assistant. Answer the user's question, concerete details in our answer. "
            "Do not mention that you lack context. Do not ask clarifying questions."
            "Do not use markdown or lists. Answer in a single passage."
        )
    else:
        system = "You are a helpful assistant. Answer the user's question in a detailed, long-form way."

    if cfg.prompt_mode == "with_context":
        user = (
            "Use the following context to answer the question. "
            "If the context does not contain enough information, say you do not know.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question:\n{question}"
        )
    else:
        # no_context: hallucination-friendly
        user = question

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


async def chat_complete_with_retry(
    client: AsyncOpenAI,
    cfg: Config,
    *,
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None

    for attempt in range(1, cfg.max_retries + 1):
        try:
            kwargs: Dict[str, Any] = {
                "model": cfg.model,
                "messages": messages,
                "temperature": cfg.temperature,
                "max_tokens": cfg.max_tokens,
                "n": cfg.n_samples,
            }
            if cfg.top_p is not None:
                kwargs["top_p"] = cfg.top_p

            resp = await client.chat.completions.create(**kwargs)
            return resp.model_dump()

        except Exception as e:
            last_err = e
            sleep_s = min(60.0, 2.0**attempt) + random.random() * 0.25
            await asyncio.sleep(sleep_s)

    assert last_err is not None
    raise last_err


def extract_choices(resp_dump: Dict[str, Any]) -> List[Dict[str, Any]]:
    choices = resp_dump.get("choices")
    return choices if isinstance(choices, list) else []


def extract_text_from_choice(choice: Dict[str, Any]) -> str:
    msg = choice.get("message") or {}
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        return msg["content"]
    return ""


async def writer_loop(out_path: Path, err_path: Path, q: asyncio.Queue) -> None:
    while True:
        item = await q.get()
        if item is None:
            q.task_done()
            break

        kind = item.get("_kind")
        rec = item.get("record")

        if kind == "out":
            await append_jsonl_async(out_path, rec)
        else:
            await append_jsonl_async(err_path, rec)

        q.task_done()


async def run(cfg: Config) -> None:
    client = AsyncOpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        timeout=cfg.timeout_s,
    )

    out_q: asyncio.Queue = asyncio.Queue(maxsize=5000)
    writer = asyncio.create_task(writer_loop(cfg.out_path, cfg.errors_path, out_q))

    sem = asyncio.Semaphore(max(1, cfg.concurrency))

    produced = 0
    started = time.time()

    async def handle_one(
        source_rec: Dict[str, Any], question: str, question_index: int
    ) -> None:
        nonlocal produced

        async with sem:
            context_text = str(source_rec.get("context_text") or "")
            messages = build_messages(cfg, question=question, context_text=context_text)

            try:
                resp_dump = await chat_complete_with_retry(
                    client, cfg, messages=messages
                )

                choices = extract_choices(resp_dump)
                if not choices:
                    err = {
                        "_error": True,
                        "stage": "empty_choices",
                        "generated_at": now_utc_iso(),
                        "question": question,
                        "question_index": question_index,
                        "source": source_rec,
                        "response": resp_dump,
                    }
                    await out_q.put({"_kind": "err", "record": err})
                    return

                for si, ch in enumerate(choices):
                    answer = extract_text_from_choice(ch)
                    out_rec = {
                        "generated_at": now_utc_iso(),
                        "answer_model": cfg.model,
                        "answer_params": {
                            "temperature": cfg.temperature,
                            "top_p": cfg.top_p,
                            "max_tokens": cfg.max_tokens,
                            "n_samples": cfg.n_samples,
                            "prompt_mode": cfg.prompt_mode,
                            "force_confident": cfg.force_confident,
                        },
                        "question": question,
                        "question_index": question_index,
                        "sample_index": si,
                        "answer": answer,
                        "finish_reason": ch.get("finish_reason"),
                        "usage": resp_dump.get("usage"),
                        "source": source_rec,
                    }
                    await out_q.put({"_kind": "out", "record": out_rec})

            except Exception as e:
                err = {
                    "_error": True,
                    "stage": "openai_call",
                    "generated_at": now_utc_iso(),
                    "question": question,
                    "question_index": question_index,
                    "error": f"{type(e).__name__}: {e}",
                    "source": source_rec,
                }
                await out_q.put({"_kind": "err", "record": err})

        produced += 1

    pending: List[asyncio.Task] = []

    for src in iter_jsonl(cfg.input_path):
        if cfg.max_records > 0 and produced >= cfg.max_records:
            break

        questions = safe_list(src.get("questions"))
        if not questions:
            questions = safe_list(src.get("questions_final")) or safe_list(
                src.get("questions_parsed")
            )

        for qi, qtext in enumerate(questions):
            if cfg.max_records > 0 and produced >= cfg.max_records:
                break
            if not isinstance(qtext, str) or not qtext.strip():
                continue

            t = asyncio.create_task(handle_one(src, qtext.strip(), qi))
            pending.append(t)

            # Simple backpressure to avoid unbounded task growth
            if len(pending) >= cfg.concurrency * 50:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                pending = list(pending)

        # Progress log (stdout only, lightweight)
        if produced > 0 and produced % 200 == 0:
            elapsed = max(1e-6, time.time() - started)
            rate = produced / elapsed
            print(
                f"[progress] produced={produced} rate={rate:.2f} records/s pending={len(pending)}"
            )

    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    await out_q.put(None)
    await out_q.join()
    await writer


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=True,
        help="Merged JSONL with questions + context_text, one record per context.",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output JSONL with answers (one record per question per sample).",
    )
    ap.add_argument("--errors-out", required=True, help="Errors JSONL.")

    ap.add_argument(
        "--base-url",
        required=True,
        help="vLLM OpenAI server base URL, e.g. http://localhost:8000/v1",
    )
    ap.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key for OpenAI-compatible server (often unused by vLLM).",
    )
    ap.add_argument(
        "--model",
        required=True,
        help="Model name as served by vLLM, e.g. openai/gpt-oss-120b",
    )

    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Max number of questions to answer (0 = unlimited).",
    )
    ap.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Number of completions per request (n).",
    )

    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=2048)

    ap.add_argument(
        "--prompt-mode", choices=["no_context", "with_context"], default="no_context"
    )
    ap.add_argument("--force-confident", action="store_true")

    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--timeout-s", type=float, default=120.0)

    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    cfg = Config(
        input_path=Path(args.input),
        out_path=Path(args.out),
        errors_path=Path(args.errors_out),
        base_url=str(args.base_url),
        api_key=str(args.api_key),
        model=str(args.model),
        concurrency=int(args.concurrency),
        max_records=int(args.max_records),
        n_samples=int(args.n_samples),
        temperature=float(args.temperature),
        top_p=(float(args.top_p) if args.top_p is not None else None),
        max_tokens=int(args.max_tokens),
        prompt_mode=str(args.prompt_mode),
        force_confident=bool(args.force_confident),
        max_retries=int(args.max_retries),
        timeout_s=float(args.timeout_s),
        seed=int(args.seed),
    )

    random.seed(cfg.seed)
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.errors_path.parent.mkdir(parents=True, exist_ok=True)

    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
