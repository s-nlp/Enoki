#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import aiohttp

try:
    import orjson  # type: ignore

    def json_dumps(obj: Any) -> str:
        return orjson.dumps(obj).decode("utf-8")
except Exception:

    def json_dumps(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False)


def mw_api_endpoint(lang: str) -> str:
    return f"https://{lang}.wikipedia.org/w/api.php"


def now_utc_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def setup_logging(log_path: Path, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("paragraph_miner")
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
    lang: str
    user_agent: str

    in_meta: Path
    out_paragraphs: Path
    out_errors: Path
    log_path: Path

    extract_mode: str  # full|lead
    max_chars: int
    min_paragraph_chars: int
    max_paragraph_chars: int
    max_paragraphs_per_page: int

    batch_size: int
    concurrency: int

    timeout_s: int
    max_retries: int

    drop_disambiguation: bool
    resume: bool


class AsyncHTTP:
    def __init__(
        self, session: aiohttp.ClientSession, cfg: Config, logger: logging.Logger
    ):
        self.session = session
        self.cfg = cfg
        self.logger = logger

    async def get_json(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"User-Agent": self.cfg.user_agent, "Accept": "application/json"}
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                async with self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.cfg.timeout_s),
                ) as r:
                    if r.status == 429:
                        ra = r.headers.get("Retry-After")
                        sleep_s = (
                            int(ra) if (ra and ra.isdigit()) else min(60, 2**attempt)
                        )
                        self.logger.warning(
                            f"429 rate limited, sleeping {sleep_s}s (attempt {attempt})"
                        )
                        await asyncio.sleep(sleep_s)
                        continue
                    if 500 <= r.status < 600:
                        sleep_s = min(60, 2**attempt)
                        self.logger.warning(
                            f"{r.status} server error, sleeping {sleep_s}s (attempt {attempt})"
                        )
                        await asyncio.sleep(sleep_s)
                        continue
                    r.raise_for_status()
                    return await r.json()

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                sleep_s = min(60, 2**attempt)
                self.logger.warning(
                    f"Request failed: {type(e).__name__}: {e} | sleeping {sleep_s}s (attempt {attempt})"
                )
                await asyncio.sleep(sleep_s)

        return {}


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


def load_processed_pageids(paragraphs_path: Path, logger: logging.Logger) -> set[int]:
    """
    Resume helper: reads paragraphs.jsonl and marks pageids that already produced at least one paragraph.
    """
    if not paragraphs_path.exists():
        return set()

    seen: set[int] = set()
    n = 0
    with paragraphs_path.open("r", encoding="utf-8") as f:
        for line in f:
            n += 1
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                pid = obj.get("pageid")
                if pid is not None:
                    seen.add(int(pid))
            except Exception:
                continue

    logger.info(
        f"Resume: loaded {len(seen)} processed pageids from {paragraphs_path} (lines scanned: {n})"
    )
    return seen


def split_paragraphs(
    text: str,
    *,
    min_chars: int,
    max_paragraph_chars: int,
    max_paragraphs: int,
) -> List[str]:
    """
    Splits plaintext into paragraphs using blank lines.
    Removes section header lines like '== Heading ==' if present.
    """
    if not text:
        return []

    t = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = t.split("\n")

    cleaned_lines: List[str] = []
    header_re = re.compile(r"^\s*==[^=].*==\s*$")
    for ln in lines:
        if header_re.match(ln):
            cleaned_lines.append("")  # force paragraph break
            continue
        cleaned_lines.append(ln)

    cleaned = "\n".join(cleaned_lines).strip()
    if not cleaned:
        return []

    raw_paras = [p.strip() for p in re.split(r"\n{2,}", cleaned) if p.strip()]
    paras: List[str] = []

    banned_section_titles = {
        "references",
        "see also",
        "external links",
        "further reading",
        "bibliography",
        "notes",
    }

    def is_bullet_heavy(paragraph: str, threshold: float = 0.35) -> bool:
        lines = [ln.strip() for ln in paragraph.split("\n") if ln.strip()]
        if not lines:
            return False
        bullet_lines = [
            ln
            for ln in lines
            if ln.startswith(("*", "-", "•", "#"))
            or re.match(r"^\d+[\.\)]\s", ln)
        ]
        return (len(bullet_lines) / len(lines)) >= threshold

    def is_banned_section(paragraph: str) -> bool:
        first_line = paragraph.split("\n", 1)[0].strip().strip(":").lower()
        return first_line in banned_section_titles

    def split_long_paragraph(paragraph: str) -> List[str]:
        if len(paragraph) <= max_paragraph_chars:
            return [paragraph]
        sentences = re.split(r"(?<=[.?!])\s+", paragraph.strip())
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        for sent in sentences:
            if not sent:
                continue
            add_len = len(sent) + (1 if current else 0)
            if current_len + add_len <= max_paragraph_chars:
                current.append(sent)
                current_len += add_len
            else:
                chunk = " ".join(current).strip()
                if chunk and len(chunk) >= min_chars:
                    chunks.append(chunk)
                current = [sent]
                current_len = len(sent)
        if current:
            chunk = " ".join(current).strip()
            if chunk and len(chunk) >= min_chars:
                chunks.append(chunk)
        # fallback: if nothing passed min_chars, keep the original paragraph
        return chunks or [paragraph]

    for p in raw_paras:
        if len(p) < min_chars:
            continue
        if is_banned_section(p):
            continue
        if is_bullet_heavy(p):
            continue
        chunks = split_long_paragraph(p)
        for ch in chunks:
            if len(ch) >= min_chars:
                paras.append(ch)

    if max_paragraphs > 0:
        paras = paras[:max_paragraphs]
    return paras


async def fetch_extracts_batch(
    http: AsyncHTTP, cfg: Config, pageids: List[int]
) -> Dict[int, Dict[str, Any]]:
    """
    Returns mapping: pageid -> {"title":..., "extract":...}
    """
    url = mw_api_endpoint(cfg.lang)
    params: Dict[str, Any] = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "prop": "extracts",
        "pageids": "|".join(str(pid) for pid in pageids),
        "explaintext": 1,
        "exsectionformat": "plain",
        "exchars": max(1, int(cfg.max_chars)),
    }
    if cfg.extract_mode == "lead":
        params["exintro"] = 1

    data = await http.get_json(url, params)
    pages = (data.get("query", {}) or {}).get("pages", []) or []

    out: Dict[int, Dict[str, Any]] = {}
    for p in pages:
        pid = p.get("pageid")
        if pid is None:
            continue
        try:
            pid_i = int(pid)
        except Exception:
            continue
        out[pid_i] = {
            "title": p.get("title"),
            "extract": p.get("extract") or "",
        }
    return out


async def writer_task(
    out_path: Path, err_path: Path, q: asyncio.Queue, logger: logging.Logger
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    err_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        out_path.open("a", encoding="utf-8") as f_out,
        err_path.open("a", encoding="utf-8") as f_err,
    ):
        while True:
            item = await q.get()
            if item is None:
                q.task_done()
                break

            if isinstance(item, dict) and item.get("_error"):
                f_err.write(json_dumps(item) + "\n")
            else:
                f_out.write(json_dumps(item) + "\n")
            q.task_done()


async def worker(
    name: str,
    http: AsyncHTTP,
    cfg: Config,
    in_q: asyncio.Queue,
    out_q: asyncio.Queue,
    logger: logging.Logger,
) -> None:
    while True:
        batch = await in_q.get()
        if batch is None:
            in_q.task_done()
            break

        # batch: list of meta records
        try:
            pageids = [int(x["pageid"]) for x in batch if x.get("pageid") is not None]
            meta_by_pid = {
                int(x["pageid"]): x for x in batch if x.get("pageid") is not None
            }
        except Exception as e:
            logger.warning(f"{name}: invalid batch: {e}")
            in_q.task_done()
            continue

        extracts_map = await fetch_extracts_batch(http, cfg, pageids)

        text_mined_at = now_utc_iso()
        for pid in pageids:
            meta = meta_by_pid.get(pid) or {}
            title = meta.get("title") or extracts_map.get(pid, {}).get("title")
            extract = extracts_map.get(pid, {}).get("extract", "")

            if not extract:
                await out_q.put(
                    {
                        "_error": True,
                        "stage": "extracts",
                        "lang": cfg.lang,
                        "pageid": pid,
                        "title": title,
                        "error": "empty_extract",
                    }
                )
                continue

            paras = split_paragraphs(
                extract,
                min_chars=cfg.min_paragraph_chars,
                max_paragraph_chars=cfg.max_paragraph_chars,
                max_paragraphs=cfg.max_paragraphs_per_page,
            )
            if not paras:
                await out_q.put(
                    {
                        "_error": True,
                        "stage": "split",
                        "lang": cfg.lang,
                        "pageid": pid,
                        "title": title,
                        "error": "no_paragraphs_after_filter",
                    }
                )
                continue

            for i, ptxt in enumerate(paras):
                rec = dict(meta)
                rec["lang"] = cfg.lang
                rec["title"] = title
                rec["pageid"] = pid

                rec["para_id"] = f"{cfg.lang}:{pid}:{i}"
                rec["para_index"] = i
                rec["para_text"] = ptxt
                rec["para_char_len"] = len(ptxt)

                rec["text_mined_at"] = text_mined_at
                rec["extract_mode"] = cfg.extract_mode
                rec["extract_max_chars"] = cfg.max_chars

                await out_q.put(rec)

        in_q.task_done()


async def run(cfg: Config, logger: logging.Logger) -> None:
    processed = (
        load_processed_pageids(cfg.out_paragraphs, logger) if cfg.resume else set()
    )

    connector = aiohttp.TCPConnector(limit=cfg.concurrency + 20, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector) as session:
        http = AsyncHTTP(session, cfg, logger)

        out_q: asyncio.Queue = asyncio.Queue(maxsize=50_000)
        writer = asyncio.create_task(
            writer_task(cfg.out_paragraphs, cfg.out_errors, out_q, logger)
        )

        in_q: asyncio.Queue = asyncio.Queue(maxsize=2 * cfg.concurrency)

        workers = [
            asyncio.create_task(worker(f"worker-{i}", http, cfg, in_q, out_q, logger))
            for i in range(cfg.concurrency)
        ]

        batch: List[Dict[str, Any]] = []
        enqueued_pages = 0
        skipped_pages = 0

        for meta in iter_jsonl(cfg.in_meta):
            pid = meta.get("pageid")
            if pid is None:
                continue
            try:
                pid_i = int(pid)
            except Exception:
                continue

            if cfg.drop_disambiguation and bool(meta.get("is_disambiguation")):
                skipped_pages += 1
                continue

            if pid_i in processed:
                skipped_pages += 1
                continue

            meta = dict(meta)
            meta["lang"] = cfg.lang
            batch.append(meta)

            if len(batch) >= cfg.batch_size:
                await in_q.put(batch)
                enqueued_pages += len(batch)
                batch = []

                if enqueued_pages % 2000 == 0:
                    logger.info(
                        f"enqueued_pages={enqueued_pages} skipped_pages={skipped_pages}"
                    )

        if batch:
            await in_q.put(batch)
            enqueued_pages += len(batch)

        for _ in workers:
            await in_q.put(None)

        await in_q.join()
        for w in workers:
            await w

        await out_q.put(None)
        await out_q.join()
        await writer

        logger.info(
            f"Done. enqueued_pages={enqueued_pages} skipped_pages={skipped_pages}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True)
    p.add_argument("--in-meta", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--errors-out", required=True)

    p.add_argument("--extract-mode", choices=["full", "lead"], default="full")
    p.add_argument("--max-chars", type=int, default=200_000)
    p.add_argument("--min-paragraph-chars", type=int, default=60)
    p.add_argument("--max-paragraph-chars", type=int, default=2_500)
    p.add_argument("--max-paragraphs-per-page", type=int, default=40)

    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--concurrency", type=int, default=4)

    p.add_argument("--timeout-s", type=int, default=30)
    p.add_argument("--max-retries", type=int, default=6)

    p.add_argument("--drop-disambiguation", action="store_true")
    p.add_argument("--no-resume", action="store_true")

    p.add_argument("--log", default="logs/paragraph_miner.log")
    p.add_argument("--log-level", default="INFO")

    p.add_argument(
        "--user-agent",
        default="WikiParagraphMiner/0.1 (contact: you@example.com)",
        help="Set a descriptive UA with contact info.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        lang=args.lang,
        user_agent=args.user_agent,
        in_meta=Path(args.in_meta),
        out_paragraphs=Path(args.out),
        out_errors=Path(args.errors_out),
        log_path=Path(args.log),
        extract_mode=args.extract_mode,
        max_chars=args.max_chars,
        min_paragraph_chars=args.min_paragraph_chars,
        max_paragraph_chars=args.max_paragraph_chars,
        max_paragraphs_per_page=args.max_paragraphs_per_page,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        timeout_s=args.timeout_s,
        max_retries=args.max_retries,
        drop_disambiguation=bool(args.drop_disambiguation),
        resume=not bool(args.no_resume),
    )

    logger = setup_logging(cfg.log_path, level=args.log_level)
    logger.info(f"Starting: {cfg}")
    asyncio.run(run(cfg, logger))


if __name__ == "__main__":
    main()
