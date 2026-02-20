#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED


LONGFACT_PROMPT_SPEC = {
    "system_prompt": r"""
Reasoning: high

You generate long-form factual QUESTIONS for a context-only QA dataset.

Input: a Wikipedia passage (one or more paragraphs).
Output: JSON only: {"questions": ["...", "...", ...]}.

Goal:
- Each question must look like a natural standalone question asked on the open web.
- Without the passage, a model should be likely to guess or hallucinate.
- With the passage, a model should be able to answer fully and factually.

CRITICAL: The question must NOT contain the answer.
- Ask for a comprehensive answer that uses the passage details, but do not enumerate those details in the question.

Hard rules:
1) Output format:
   - Return exactly N questions.
   - Return only valid JSON with a single key "questions".
   - No extra keys, no markdown, no commentary.

2) No passage references:
   - Do not mention the passage, text, excerpt, or "above".
   - Do not use phrases like: "in the passage", "according to the text", "as mentioned", "listed".
   - Do not ask meta-questions about what is described/mentioned/listed.

3) Factual and neutral only:
   - No opinions, preferences, value judgments, persuasion, or debate prompts.
   - Avoid subjective framing words such as: best, worst, good, bad, important, significant, remarkable, interesting, controversial, should.

4) Not incomplete:
   - The question must name its subject explicitly.
   - No dangling pronouns without antecedents ("this", "it", "they", "he", "she", "these").

5) Long-form answer requirement without leaking details:
   - Request a long-form answer (multiple sentences) and a comprehensive overview.
   - The question may mention TYPES of details to include (dates, numbers, locations, roles, organizations),
     but MUST NOT include specific values from the passage.
   - Do NOT include more than ONE numeric value in the question (preferably include ZERO numbers).
   - Do NOT include comma-separated lists of facts pulled from the passage.

6) No list-requests:
   - Do not ask to list or enumerate items, members, subspecies, features, etc.

Language:
- Write the questions in the same language as the passage, unless a target language is explicitly provided.

Return only JSON: {"questions": [...]}.
""".strip(),
    "user_template": r"""
N: {n_questions}
Title: {title}
Language: {lang}
Passage:
{passage}
{target_language_line}
""".strip(),
    "few_shot": [
        {
            "role": "user",
            "content": r"""
N: 3
Title: Charles-Omer Valois
Language: en
Passage:
Charles-Omer Valois (April 24, 1924 – August 4, 2013) was a Canadian prelate of the Catholic Church.
Charles-Omer Valois was born in Montreal and was ordained a priest on June 3, 1950.
Valois was appointed bishop of the Diocese of Saint-Jérôme on June 10, 1977, and ordained bishop on June 29, 1977. Valois would resign from the diocese on January 22, 1997.
""".strip(),
        },
        {
            "role": "assistant",
            "content": r"""
{"questions":["Provide a detailed overview of Charles-Omer Valois, including his background and role in the Catholic Church, and in your answer include the key dated milestones of his ordinations, appointments, and the end of his service as bishop of the Diocese of Saint-Jérôme.","Who was Charles-Omer Valois? Give a comprehensive biographical account that explains his origins and ecclesiastical career, and include the major dates and positions mentioned for his priestly ordination and episcopal tenure.","Explain the life and church career of Charles-Omer Valois in a long-form way, covering where he was from, what offices he held, and the timeline of his progression from priest to bishop through the specific events described in the passage."]}
""".strip(),
        },
    ],
}


try:
    import orjson  # type: ignore

    def json_dumps(obj: Any) -> str:
        return orjson.dumps(obj).decode("utf-8")
except Exception:

    def json_dumps(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False)


def now_utc_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def setup_logging(log_path: Path, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("qa_generator_parallel")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


@dataclass(frozen=True)
class Config:
    input_path: Path
    output_path: Path
    errors_path: Path
    raw_output_path: Path
    log_path: Path

    model: str
    temperature: float
    max_output_tokens: int

    context_paragraphs: int
    stride: int
    questions_per_context: int
    generations: int
    max_context_chars: int

    language_mode: str  # auto|force
    output_language: Optional[str]

    resume: bool
    max_retries: int
    max_examples: int

    workers: int
    max_in_flight: int

    raw_include_messages: bool
    raw_max_message_chars: int

    no_postprocess: bool
    min_question_chars: int
    min_question_words: int


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json_dumps(obj) + "\n")


def load_done_context_ids(out_path: Path, logger: logging.Logger) -> set[str]:
    if not out_path.exists():
        return set()
    done: set[str] = set()
    n = 0
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            n += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cid = obj.get("context_id")
                if cid:
                    done.add(str(cid))
            except Exception:
                continue
    logger.info(
        f"Resume: loaded {len(done)} context_ids from {out_path} (lines scanned: {n})"
    )
    return done


def make_context_id(
    lang: str, pageid: int, start_idx: int, end_idx: int, gen_idx: int
) -> str:
    s = f"{lang}|{pageid}|{start_idx}|{end_idx}|{gen_idx}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def truncate_text(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip()


def contexts_from_page(
    paras: List[Dict[str, Any]],
    *,
    context_size: int,
    stride: int,
    max_context_chars: int,
) -> List[Dict[str, Any]]:
    paras_sorted = sorted(paras, key=lambda x: int(x.get("para_index", 0)))
    out = []

    if context_size <= 0:
        context_size = 1
    if stride <= 0:
        stride = context_size

    for start in range(0, len(paras_sorted) - context_size + 1, stride):
        chunk = paras_sorted[start : start + context_size]
        para_ids = [str(x.get("para_id")) for x in chunk]
        para_indices = [int(x.get("para_index", 0)) for x in chunk]
        chunk_paras = [
            {
                "para_id": x.get("para_id"),
                "para_index": int(x.get("para_index", 0)),
                "para_text": str(x.get("para_text", "")).strip(),
            }
            for x in chunk
        ]
        text = "\n\n".join(p["para_text"] for p in chunk_paras).strip()
        text = truncate_text(text, max_context_chars)

        out.append(
            {
                "para_ids": para_ids,
                "para_indices": para_indices,
                "context_text": text,
                "paragraphs": chunk_paras,
                "start_index": para_indices[0] if para_indices else start,
                "end_index": para_indices[-1] if para_indices else start,
            }
        )

    return out


def clip_messages(
    messages: List[Dict[str, str]], max_chars: int
) -> List[Dict[str, str]]:
    out = []
    for m in messages:
        role = str(m.get("role") or "")
        content = str(m.get("content") or "")
        if max_chars > 0 and len(content) > max_chars:
            content = content[:max_chars] + "...[truncated]"
        out.append({"role": role, "content": content})
    return out


def build_messages(
    cfg: Config, *, title: str, lang: str, passage: str
) -> List[Dict[str, str]]:
    target_language_line = ""
    if cfg.language_mode == "force" and cfg.output_language:
        target_language_line = f"Target language: {cfg.output_language}"

    user_prompt = (
        LONGFACT_PROMPT_SPEC["user_template"]
        .format(
            n_questions=cfg.questions_per_context,
            title=title,
            lang=lang,
            passage=passage,
            target_language_line=target_language_line,
        )
        .strip()
        + "\n"
    )

    msgs: List[Dict[str, str]] = [
        {"role": "system", "content": LONGFACT_PROMPT_SPEC["system_prompt"]}
    ]
    for m in LONGFACT_PROMPT_SPEC.get("few_shot", []):
        role = str(m.get("role") or "")
        content = str(m.get("content") or "")
        if role in {"user", "assistant"} and content.strip():
            msgs.append({"role": role, "content": content.strip()})
    msgs.append({"role": "user", "content": user_prompt})
    return msgs


def strip_code_fences(s: str) -> str:
    s2 = s.strip()
    if s2.startswith("```"):
        s2 = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s2)
        s2 = re.sub(r"\s*```$", "", s2)
    return s2.strip()


class ParseError(Exception):
    pass


def parse_questions_simple(raw_text: str) -> Tuple[List[str], Dict[str, Any]]:
    meta: Dict[str, Any] = {"parse_method": None}
    if not raw_text or not raw_text.strip():
        raise ParseError("empty_output_text")

    s = strip_code_fences(raw_text)

    # Full JSON parse
    try:
        obj = json.loads(s, strict=False)
        meta["parse_method"] = "json_full"
    except Exception:
        obj = None

    # Substring parse
    if obj is None:
        i = s.find("{")
        j = s.rfind("}")
        if i >= 0 and j > i:
            sub = s[i : j + 1]
            try:
                obj = json.loads(sub, strict=False)
                meta["parse_method"] = "json_substring"
            except Exception as e:
                raise ParseError(f"json_parse_failed: {type(e).__name__}: {e}")
        else:
            raise ParseError("no_json_object_found")

    if isinstance(obj, dict) and isinstance(obj.get("questions"), list):
        questions = obj["questions"]
    elif isinstance(obj, list):
        questions = obj
    else:
        raise ParseError(f"unexpected_json_shape: {type(obj).__name__}")

    if not all(isinstance(q, str) for q in questions):
        raise ParseError("questions_not_all_strings")

    questions = [q.strip() for q in questions if q and q.strip()]
    return questions, meta


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

_SUBJECTIVE_RE = [
    re.compile(r"\b(best|worst|good|bad|better|worse)\b", re.IGNORECASE),
    re.compile(
        r"\b(important|significant|remarkable|interesting|controversial)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(should|ought to)\b", re.IGNORECASE),
]

_LISTY_RE = [
    re.compile(r"\blist\b", re.IGNORECASE),
    # re.compile(r"\bwhich are\b", re.IGNORECASE),
    # re.compile(r"\bwhat are the\b", re.IGNORECASE),
    re.compile(r"\benumerate\b", re.IGNORECASE),
]


def approx_word_count(s: str) -> int:
    return len([w for w in re.split(r"\s+", s.strip()) if w])


def is_bad_question(cfg: Config, q: str) -> bool:
    s = q.strip()
    if not s:
        return True

    for rx in _BANNED_RE:
        if rx.search(s):
            return True
    for rx in _SUBJECTIVE_RE:
        if rx.search(s):
            return True
    for rx in _LISTY_RE:
        if rx.search(s):
            return True

    if len(s) < cfg.min_question_chars:
        return True

    if " " in s and approx_word_count(s) < cfg.min_question_words:
        return True

    # Anti-leak heuristic: allow up to 3 numeric groups (e.g., two years); reject heavier numeric leakage
    nums = re.findall(r"\d+", s)
    if len(nums) >= 4:
        return True

    return False


def postprocess_questions(cfg: Config, qs: List[str]) -> List[str]:
    cleaned = []
    seen = set()
    for q in qs:
        q2 = " ".join(str(q).strip().split())
        k = q2.lower()
        if not q2 or k in seen:
            continue
        seen.add(k)
        cleaned.append(q2)

    good = [q for q in cleaned if not is_bad_question(cfg, q)]
    return good[: cfg.questions_per_context]


_thread_local = threading.local()


def get_client() -> OpenAI:
    c = getattr(_thread_local, "client", None)
    if c is None:
        c = OpenAI()
        _thread_local.client = c
    return c


def extract_output_text(resp: Any) -> str:
    t = getattr(resp, "output_text", None)
    if isinstance(t, str):
        return t
    try:
        d = resp.model_dump()
        if isinstance(d, dict) and isinstance(d.get("output_text"), str):
            return d["output_text"]
    except Exception:
        pass
    return ""


def call_openai_raw(
    cfg: Config, *, messages: List[Dict[str, str]]
) -> Tuple[Dict[str, Any], str]:
    last_err: Optional[Exception] = None
    client = get_client()

    for attempt in range(1, cfg.max_retries + 1):
        try:
            resp = client.responses.create(
                model=cfg.model,
                input=messages,
                temperature=cfg.temperature,
                max_output_tokens=cfg.max_output_tokens,
            )
            try:
                dump = resp.model_dump()
            except Exception:
                dump = {"repr": repr(resp)}
            text = extract_output_text(resp)
            return dump, text
        except Exception as e:
            last_err = e
            sleep_s = min(60, 2**attempt)
            time.sleep(sleep_s)

    assert last_err is not None
    raise last_err


def generate_one(task: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    """
    Worker function executed in a thread.
    Returns a dict with:
      - status: ok|parse_error|call_error|quality
      - raw_rec (always for ok/parse_error/quality)
      - out_rec (only for ok)
      - err_rec (for errors)
    """
    lang = task["lang"]
    pageid = task["pageid"]
    title = task["title"]
    cid = task["context_id"]
    gen_idx = task["generation_index"]
    ctx = task["ctx"]

    messages = build_messages(cfg, title=title, lang=lang, passage=ctx["context_text"])

    try:
        resp_dump, raw_text = call_openai_raw(cfg, messages=messages)

        raw_rec = {
            "created_at": now_utc_iso(),
            "context_id": cid,
            "lang": lang,
            "pageid": pageid,
            "title": title,
            "generation_index": gen_idx,
            "context_paragraphs": cfg.context_paragraphs,
            "stride": cfg.stride,
            "questions_per_context": cfg.questions_per_context,
            "model": cfg.model,
            "temperature": cfg.temperature,
            "raw_text": raw_text,
            "response": resp_dump,
            "context_text": ctx.get("context_text"),
        }
        if cfg.raw_include_messages:
            raw_rec["messages"] = clip_messages(messages, cfg.raw_max_message_chars)

        try:
            questions_parsed, parse_meta = parse_questions_simple(raw_text)
        except Exception as pe:
            err_rec = {
                "_error": True,
                "stage": "parse",
                "created_at": now_utc_iso(),
                "context_id": cid,
                "lang": lang,
                "pageid": pageid,
                "title": title,
                "generation_index": gen_idx,
                "error": f"{type(pe).__name__}: {pe}",
                "raw_text_head": raw_text[:2000] if isinstance(raw_text, str) else "",
            }
            return {"status": "parse_error", "raw_rec": raw_rec, "err_rec": err_rec}

        if cfg.no_postprocess:
            questions_final = questions_parsed[: cfg.questions_per_context]
        else:
            questions_final = postprocess_questions(cfg, questions_parsed)
        # TODO: think about it later, but now even 1 question is good.
        if len(questions_final) < 1:
            # if len(questions_final) < cfg.questions_per_context:
            err_rec = {
                "_error": True,
                "stage": "quality",
                "created_at": now_utc_iso(),
                "context_id": cid,
                "lang": lang,
                "pageid": pageid,
                "title": title,
                "generation_index": gen_idx,
                "error": f"too_few_questions_final: got={len(questions_final)} expected={cfg.questions_per_context}",
                "parse_meta": parse_meta,
                "questions_parsed": questions_parsed,
                "questions_final": questions_final,
            }
            out_rec = {
                "context_id": cid,
                "created_at": now_utc_iso(),
                "model": cfg.model,
                "temperature": cfg.temperature,
                "language_mode": cfg.language_mode,
                "output_language": cfg.output_language,
                "context_paragraphs": cfg.context_paragraphs,
                "stride": cfg.stride,
                "questions_per_context": cfg.questions_per_context,
                "generation_index": gen_idx,
                "lang": lang,
                "pageid": pageid,
                "title": title,
                "para_ids": ctx["para_ids"],
                "para_indices": ctx["para_indices"],
                "context_text": ctx.get("context_text"),
                "context_paragraphs_list": ctx.get("paragraphs"),
                "parse_meta": parse_meta,
                "questions_parsed": questions_parsed,
                "questions": questions_final,
            }
            return {
                "status": "quality",
                "raw_rec": raw_rec,
                "out_rec": out_rec,
                "err_rec": err_rec,
            }

        out_rec = {
            "context_id": cid,
            "created_at": now_utc_iso(),
            "model": cfg.model,
            "temperature": cfg.temperature,
            "language_mode": cfg.language_mode,
            "output_language": cfg.output_language,
            "context_paragraphs": cfg.context_paragraphs,
            "stride": cfg.stride,
            "questions_per_context": cfg.questions_per_context,
            "generation_index": gen_idx,
            "lang": lang,
            "pageid": pageid,
            "title": title,
            "para_ids": ctx["para_ids"],
            "para_indices": ctx["para_indices"],
            "context_text": ctx.get("context_text"),
            "context_paragraphs_list": ctx.get("paragraphs"),
            "parse_meta": parse_meta,
            "questions_parsed": questions_parsed,
            "questions": questions_final,
        }
        return {"status": "ok", "raw_rec": raw_rec, "out_rec": out_rec}

    except Exception as e:
        err_rec = {
            "_error": True,
            "stage": "openai_call",
            "created_at": now_utc_iso(),
            "context_id": cid,
            "lang": lang,
            "pageid": pageid,
            "title": title,
            "generation_index": gen_idx,
            "error": f"{type(e).__name__}: {e}",
        }
        return {"status": "call_error", "err_rec": err_rec}


def tasks_from_paragraphs(cfg: Config, done_ids: set[str]) -> Iterable[Dict[str, Any]]:
    current_key: Optional[Tuple[str, int]] = None
    page_paras: List[Dict[str, Any]] = []
    page_meta: Dict[str, Any] = {}

    def flush_page_tasks() -> Iterable[Dict[str, Any]]:
        if not page_paras:
            return []
        lang = str(page_meta.get("lang") or "")
        pageid = int(page_meta.get("pageid") or 0)
        title = str(page_meta.get("title") or "")

        contexts = contexts_from_page(
            page_paras,
            context_size=cfg.context_paragraphs,
            stride=cfg.stride,
            max_context_chars=cfg.max_context_chars,
        )

        out_tasks = []
        for ctx in contexts:
            if not ctx["context_text"]:
                continue
            for gen_idx in range(cfg.generations):
                cid = make_context_id(
                    lang,
                    pageid,
                    int(ctx["start_index"]),
                    int(ctx["end_index"]),
                    gen_idx,
                )
                if cid in done_ids:
                    continue
                out_tasks.append(
                    {
                        "lang": lang,
                        "pageid": pageid,
                        "title": title,
                        "context_id": cid,
                        "generation_index": gen_idx,
                        "ctx": ctx,
                    }
                )
        return out_tasks

    for rec in iter_jsonl(cfg.input_path):
        lang = str(rec.get("lang") or "")
        pid_raw = rec.get("pageid")
        if pid_raw is None:
            continue
        try:
            pageid = int(pid_raw)
        except Exception:
            continue

        key = (lang, pageid)
        if current_key is None:
            current_key = key
            page_meta = {"lang": lang, "pageid": pageid, "title": rec.get("title")}

        if key != current_key:
            for t in flush_page_tasks():
                yield t
            current_key = key
            page_paras = []
            page_meta = {"lang": lang, "pageid": pageid, "title": rec.get("title")}

        page_paras.append(
            {
                "para_id": rec.get("para_id"),
                "para_index": rec.get("para_index", 0),
                "para_text": rec.get("para_text", ""),
            }
        )

    for t in flush_page_tasks():
        yield t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="paragraphs.jsonl")
    ap.add_argument("--out", required=True, help="questions.jsonl")
    ap.add_argument("--errors-out", required=True, help="questions.errors.jsonl")
    ap.add_argument(
        "--raw-out", default=None, help="raw responses JSONL (default: <out>.raw.jsonl)"
    )

    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-output-tokens", type=int, default=4096)

    ap.add_argument("--context-paragraphs", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--questions-per-context", type=int, default=3)
    ap.add_argument("--generations", type=int, default=1)
    ap.add_argument("--max-context-chars", type=int, default=7000)

    ap.add_argument("--language-mode", choices=["auto", "force"], default="auto")
    ap.add_argument("--output-language", default=None)

    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--max-examples", type=int, default=0)

    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-in-flight", type=int, default=48)

    ap.add_argument("--raw-include-messages", action="store_true")
    ap.add_argument("--raw-max-message-chars", type=int, default=1500)

    ap.add_argument("--no-postprocess", action="store_true")
    ap.add_argument("--min-question-chars", type=int, default=80)
    ap.add_argument("--min-question-words", type=int, default=10)

    ap.add_argument("--log", default="logs/qa_generator_parallel.log")
    ap.add_argument("--log-level", default="INFO")

    args = ap.parse_args()

    out_path = Path(args.out)
    raw_path = (
        Path(args.raw_out) if args.raw_out else Path(str(out_path) + ".raw.jsonl")
    )

    cfg = Config(
        input_path=Path(args.input),
        output_path=out_path,
        errors_path=Path(args.errors_out),
        raw_output_path=raw_path,
        log_path=Path(args.log),
        model=args.model,
        temperature=float(args.temperature),
        max_output_tokens=int(args.max_output_tokens),
        context_paragraphs=int(args.context_paragraphs),
        stride=int(args.stride),
        questions_per_context=int(args.questions_per_context),
        generations=int(args.generations),
        max_context_chars=int(args.max_context_chars),
        language_mode=str(args.language_mode),
        output_language=args.output_language,
        resume=not bool(args.no_resume),
        max_retries=int(args.max_retries),
        max_examples=int(args.max_examples),
        workers=int(args.workers),
        max_in_flight=int(args.max_in_flight),
        raw_include_messages=bool(args.raw_include_messages),
        raw_max_message_chars=int(args.raw_max_message_chars),
        no_postprocess=bool(args.no_postprocess),
        min_question_chars=int(args.min_question_chars),
        min_question_words=int(args.min_question_words),
    )

    logger = setup_logging(cfg.log_path, level=args.log_level)
    logger.info(f"Starting: {cfg}")

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.errors_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.raw_output_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = load_done_context_ids(cfg.output_path, logger) if cfg.resume else set()

    n_written = 0
    n_errors = 0
    stop_requested = False

    def handle_result(res: Dict[str, Any]) -> None:
        nonlocal n_written, n_errors
        status = res.get("status")

        if res.get("raw_rec") is not None:
            append_jsonl(cfg.raw_output_path, res["raw_rec"])

        if status == "ok":
            append_jsonl(cfg.output_path, res["out_rec"])
            done_ids.add(res["out_rec"]["context_id"])
            n_written += 1
            return

        if status in {"quality", "parse_error", "call_error"}:
            if res.get("out_rec") is not None:
                append_jsonl(cfg.output_path, res["out_rec"])
                done_ids.add(res["out_rec"]["context_id"])
                n_written += 1
            if res.get("err_rec") is not None:
                append_jsonl(cfg.errors_path, res["err_rec"])
            n_errors += 1
            return

    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futures = set()

        for task in tasks_from_paragraphs(cfg, done_ids):
            if stop_requested:
                break

            if cfg.max_examples > 0 and n_written >= cfg.max_examples:
                stop_requested = True
                break

            fut = ex.submit(generate_one, task, cfg)
            futures.add(fut)

            if len(futures) >= cfg.max_in_flight:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for f in done:
                    res = f.result()
                    handle_result(res)

                    if cfg.max_examples > 0 and n_written >= cfg.max_examples:
                        stop_requested = True
                        break

            if n_written > 0 and n_written % 50 == 0:
                logger.info(
                    f"written={n_written} errors={n_errors} in_flight={len(futures)}"
                )

        # Drain remaining
        while futures and not (cfg.max_examples > 0 and n_written >= cfg.max_examples):
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for f in done:
                res = f.result()
                handle_result(res)
                if cfg.max_examples > 0 and n_written >= cfg.max_examples:
                    break

    logger.info(f"Done. written={n_written} errors={n_errors}")


if __name__ == "__main__":
    main()
