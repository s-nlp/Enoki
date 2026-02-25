#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM-based filter for generated QA data.

Supports two input shapes:
- Context-level records from qa_gen_fast.py: a list of questions plus context_text.
- Per-question records from hallucination_answerer.py: one question (and answer) with source.context_text.

For each question (or question+answer), the script asks a model (default: gpt-oss)
to judge grounding, quality, and answer support. Accepted items are written to --out;
rejections with reasons go to --errors-out.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import AsyncOpenAI

try:
    import orjson  # type: ignore

    def json_dumps(obj: Any) -> str:
        return orjson.dumps(obj).decode("utf-8")
except Exception:  # pragma: no cover

    def json_dumps(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False)


# -------- I/O helpers --------
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


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json_dumps(obj) + "\n")


# -------- light regex prefilter (fast) --------
_BANNED_RE = [
    re.compile(r"\bin the passage\b", re.IGNORECASE),
    re.compile(r"\bin this passage\b", re.IGNORECASE),
    re.compile(r"\bin the text\b", re.IGNORECASE),
    re.compile(r"\baccording to\b", re.IGNORECASE),
    re.compile(r"\bas mentioned\b", re.IGNORECASE),
    re.compile(r"\bas described\b", re.IGNORECASE),
    re.compile(r"\bas stated\b", re.IGNORECASE),
    re.compile(r"\blisted\b", re.IGNORECASE),
    re.compile(r"\bthe passage\b", re.IGNORECASE),
    re.compile(r"\bthe text\b", re.IGNORECASE),
    re.compile(r"\bexcerpt\b", re.IGNORECASE),
]


def quick_reject(question: str) -> bool:
    for rx in _BANNED_RE:
        if rx.search(question):
            return True
    return False


# -------- prompts --------
QUESTION_SYSTEM = """
You are a strict data quality rater for a long-form QA dataset built from Wikipedia.

Goal: Decide if a QUESTION is acceptable and grounded in the provided CONTEXT.

Accept only if ALL are true:
- The question can be fully answered using the context.
- It does not reveal the answer directly. Copying multiple specific numbers/dates from the context is a leak. Using a single year or coarse time range as a constraint is OK. Asking for roles/credits or an area/rank without stating the numbers is OK.
- It is neutral, factual, not opinionated, no list-requests, no meta references to the passage.
- It is well-formed and standalone (names its subject explicitly).

Allowed reasons (use EXACT tokens):
- ok  (only when keep=true)
- not_grounded         (context doesn’t support question)
- answer_leak          (copies multiple specific numbers/dates/answers)
- meta_reference       ("according to the passage", etc.)
- subjective_or_list   (opinions, list requests)
- bad_form             (unclear subject, incomplete, malformed)
- other

Respond with compact JSON only:
{"keep": true|false, "reasons": ["<one_allowed_reason>"], "scores": {"relevance": 0-1, "quality": 0-1}}
""".strip()

QUESTION_FEWSHOT = [
    {
        "role": "user",
        "content": "Context: Charles-Omer Valois ... ordained a priest on June 3, 1950 ...\nQuestion: Provide a detailed overview of Charles-Omer Valois, including his background and role in the Catholic Church, and in your answer include the key dated milestones of his ordinations, appointments, and the end of his service as bishop of the Diocese of Saint-Jérôme.",
    },
    {
        "role": "assistant",
        "content": '{"keep": true, "reasons": ["ok"], "scores": {"relevance": 0.96, "quality": 0.9}}',
    },
    {
        "role": "user",
        "content": "Context: Charles-Omer Valois ... ordained a priest on June 3, 1950 ...\nQuestion: According to the passage, when exactly was he ordained and when did he resign?",
    },
    {
        "role": "assistant",
        "content": '{"keep": false, "reasons": ["meta_reference"], "scores": {"relevance": 0.4, "quality": 0.2}}',
    },
    {
        "role": "user",
        "content": "Context: Charles-Omer Valois ... ordained a priest on June 3, 1950 ... appointed bishop June 10, 1977 ... resigned January 22, 1997.\nQuestion: How did Charles-Omer Valois's role and duties change between 1977 and 1997?",
    },
    {
        "role": "assistant",
        "content": '{"keep": true, "reasons": ["ok"], "scores": {"relevance": 0.92, "quality": 0.88}}',
    },
    {
        "role": "user",
        "content": "Context: A tailcoat is a knee-length coat with a cut-away front and long back tails. Variants include the shadbelly used in some equestrian disciplines.\nQuestion: What is a shadbelly and how does it relate to the traditional tailcoat in style and use within equestrian dress?",
    },
    {
        "role": "assistant",
        "content": '{"keep": true, "reasons": ["ok"], "scores": {"relevance": 0.9, "quality": 0.85}}',
    },
    {
        "role": "user",
        "content": 'Context: Haunted is a studio album by Six Feet Under. Produced by Brian Slagel and Scott Burns; engineered and mixed by Scott Burns; mastered by Eddy Schreyer.\nQuestion: Who produced, engineered, mixed, and mastered the album "Haunted," and what did each role entail?',
    },
    {
        "role": "assistant",
        "content": '{"keep": true, "reasons": ["ok"], "scores": {"relevance": 0.92, "quality": 0.9}}',
    },
]

QUESTION_USER_TEMPLATE = """
Context:
{context}

Question:
{question}
""".strip()

ANSWER_SYSTEM = """
You judge whether an ANSWER is supported by the provided CONTEXT for a QUESTION.

Reject if the answer contradicts, invents, or omits key facts from context, or if the question cannot be answered from context.

Allowed reasons (use EXACT tokens):
- ok  (only when keep=true)
- not_grounded         (answer not supported by context)
- incomplete           (misses key facts present)
- contradiction        (conflicts with context)
- other

Respond with JSON only:
{"keep": true|false, "reasons": ["<one_allowed_reason>"], "scores": {"grounding": 0-1, "completeness": 0-1}}
""".strip()

ANSWER_FEWSHOT = [
    {
        "role": "user",
        "content": "Question: Who was Charles-Omer Valois?\nContext: Charles-Omer Valois (April 24, 1924 – August 4, 2013) was a Canadian prelate... ordained a priest on June 3, 1950... appointed bishop of Saint-Jérôme on June 10, 1977 ... resigned January 22, 1997.\nAnswer: Charles-Omer Valois was a Canadian Catholic prelate. Born in 1924 in Montreal, he was ordained a priest in 1950 and later became bishop of Saint-Jérôme in 1977, serving until his resignation in 1997.",
    },
    {
        "role": "assistant",
        "content": '{"keep": true, "reasons": ["ok"], "scores": {"grounding": 0.95, "completeness": 0.9}}',
    },
    {
        "role": "user",
        "content": "Question: Who was Charles-Omer Valois?\nContext: Charles-Omer Valois ...\nAnswer: He was a famous painter from Paris in the 1800s.",
    },
    {
        "role": "assistant",
        "content": '{"keep": false, "reasons": ["contradiction"], "scores": {"grounding": 0.0, "completeness": 0.0}}',
    },
]

ANSWER_USER_TEMPLATE = """
Question:
{question}

Context:
{context}

Answer:
{answer}
""".strip()


# -------- config --------
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
    max_records: int  # 0 = unlimited

    eval_target: str  # auto|question|answer
    allow_empty_context: bool
    rule_prefilter: bool


# -------- OpenAI helpers --------
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
            sleep_s = min(60.0, 2.0**attempt) + random.random() * 0.25
            await asyncio.sleep(sleep_s)
    raise last_err or RuntimeError("chat_json failed")


def build_question_messages(question: str, context: str) -> List[Dict[str, str]]:
    user = QUESTION_USER_TEMPLATE.format(context=context, question=question)
    return (
        [{"role": "system", "content": QUESTION_SYSTEM}]
        + QUESTION_FEWSHOT
        + [{"role": "user", "content": user}]
    )


def build_answer_messages(
    question: str, context: str, answer: str
) -> List[Dict[str, str]]:
    user = ANSWER_USER_TEMPLATE.format(
        question=question, context=context, answer=answer
    )
    return (
        [{"role": "system", "content": ANSWER_SYSTEM}]
        + ANSWER_FEWSHOT
        + [{"role": "user", "content": user}]
    )


# -------- processing --------
def pick_context(rec: Dict[str, Any]) -> str:
    return str(
        rec.get("context_text")
        or rec.get("context")
        or (rec.get("source", {}) or {}).get("context_text")
        or ""
    )


def pick_questions(rec: Dict[str, Any]) -> List[str]:
    # Preferred: list fields
    qs = rec.get("questions")
    if isinstance(qs, list):
        return [str(x) for x in qs if isinstance(x, str)]
    for k in ("questions_final", "questions_parsed"):
        q2 = rec.get(k)
        if isinstance(q2, list):
            return [str(x) for x in q2 if isinstance(x, str)]
    # Fallback: single-question records
    single = rec.get("question") or rec.get("prompt")
    if isinstance(single, str) and single.strip():
        return [single]
    return []


def detect_mode(cfg: Config) -> str:
    for rec in iter_jsonl(cfg.input_path):
        if "answer" in rec or ("source" in rec and (rec["source"] or {}).get("answer")):
            return "answer"
        return "question"
    return "question"


async def filter_question(
    client: AsyncOpenAI, cfg: Config, question: str, context: str
) -> Tuple[bool, Dict[str, Any]]:
    if not context and not cfg.allow_empty_context:
        return False, {"reason": "no_context", "explain": "missing context text"}
    if cfg.rule_prefilter and quick_reject(question):
        return False, {
            "reason": "banned_regex",
            "explain": "question matched banned phrase",
        }
    msg = build_question_messages(question, context)
    result = await chat_json(client, cfg, msg)
    keep = bool(result.get("keep"))
    reasons = result.get("reasons") or []
    first = reasons[0] if reasons else None
    reason = first or ("ok" if keep else "llm_reject")
    reason = str(reason).lower().replace(" ", "_")[:40]
    meta = {"llm": result, "reason": reason, "explain": first}
    return keep, meta


async def filter_answer(
    client: AsyncOpenAI, cfg: Config, question: str, context: str, answer: str
) -> Tuple[bool, Dict[str, Any]]:
    if not context and not cfg.allow_empty_context:
        return False, {"reason": "no_context", "explain": "missing context text"}
    msg = build_answer_messages(question, context, answer)
    result = await chat_json(client, cfg, msg)
    keep = bool(result.get("keep"))
    reasons = result.get("reasons") or []
    first = reasons[0] if reasons else None
    reason = first or ("ok" if keep else "llm_reject")
    reason = str(reason).lower().replace(" ", "_")[:40]
    meta = {"llm": result, "reason": reason, "explain": first}
    return keep, meta


async def process(cfg: Config) -> None:
    mode = cfg.eval_target
    if mode == "auto":
        mode = detect_mode(cfg)

    client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=120.0)
    sem = asyncio.Semaphore(max(1, cfg.concurrency))

    produced = 0
    kept = 0
    dropped = 0
    started = time.time()

    async def rate_and_record(task):
        nonlocal kept, dropped
        async with sem:
            try:
                if mode == "answer":
                    keep, meta = await filter_answer(
                        client, cfg, task["question"], task["context"], task["answer"]
                    )
                else:
                    keep, meta = await filter_question(
                        client, cfg, task["question"], task["context"]
                    )
            except Exception as e:
                meta = {"reason": "llm_error", "error": f"{type(e).__name__}: {e}"}
                keep = False

            if keep:
                kept += 1
                append_jsonl(cfg.out_path, task["out_rec"] | {"filter_meta": meta})
            else:
                dropped += 1
                err = {
                    "_error": True,
                    "stage": "filter",
                    "question": task.get("question"),
                    "answer": task.get("answer"),
                    "title": task.get("title"),
                    "meta": meta,
                }
                append_jsonl(cfg.errors_path, err)

    pending: List[asyncio.Task] = []

    for rec in iter_jsonl(cfg.input_path):
        if 0 < cfg.max_records <= produced:
            break
        context_text = pick_context(rec)
        title = str(rec.get("title") or (rec.get("source") or {}).get("title") or "")

        if mode == "answer":
            question = str(rec.get("question") or "")
            answer = str(rec.get("answer") or "")
            task = {
                "question": question,
                "answer": answer,
                "context": context_text,
                "title": title,
                "out_rec": rec,
            }
            pending.append(asyncio.create_task(rate_and_record(task)))
            produced += 1
        else:
            questions = pick_questions(rec)
            if not questions:
                continue
            kept_questions: List[str] = []
            for q in questions:
                if 0 < cfg.max_records <= produced:
                    break
                task = {
                    "question": q,
                    "context": context_text,
                    "title": title,
                    "out_rec": rec,
                }
                pending.append(asyncio.create_task(rate_and_record(task)))
                produced += 1
                kept_questions.append(q)

        if len(pending) >= cfg.concurrency * 20:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            pending = list(pending)

        if produced and produced % 50 == 0:
            rate = produced / max(1e-6, time.time() - started)
            print(
                f"[progress] seen={produced} kept={kept} dropped={dropped} r={rate:.2f}/s"
            )

    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    elapsed = max(1e-6, time.time() - started)
    print(
        f"[done] mode={mode} seen={produced} kept={kept} dropped={dropped} "
        f"elapsed_s={elapsed:.1f} rate={produced / elapsed:.2f}/s"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL (questions or answers)")
    ap.add_argument("--out", required=True, help="Filtered output JSONL")
    ap.add_argument("--errors-out", required=True, help="Rejected items JSONL")

    ap.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible base URL",
    )
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=16000)

    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--max-records", type=int, default=0, help="0 = all questions")

    ap.add_argument(
        "--eval-target",
        choices=["auto", "question", "answer"],
        default="auto",
        help="auto=detect; question=just questions+context; answer=question+context+answer",
    )
    ap.add_argument("--allow-empty-context", action="store_true")
    ap.add_argument(
        "--no-rule-prefilter", action="store_true", help="Disable quick regex prefilter"
    )

    args = ap.parse_args()

    cfg = Config(
        input_path=Path(args.input),
        out_path=Path(args.out),
        errors_path=Path(args.errors_out),
        base_url=str(args.base_url),
        api_key=str(args.api_key),
        model=str(args.model),
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        concurrency=int(args.concurrency),
        max_retries=int(args.max_retries),
        max_records=int(args.max_records),
        eval_target=str(args.eval_target),
        allow_empty_context=bool(args.allow_empty_context),
        rule_prefilter=not bool(args.no_rule_prefilter),
    )

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.errors_path.parent.mkdir(parents=True, exist_ok=True)

    asyncio.run(process(cfg))


if __name__ == "__main__":
    main()
