#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
wiki_miner.py

Fast async Wikipedia metadata miner for dataset creation.

What it collects per page (JSONL):
- core: title, pageid, urls, length, lastrevid, touched
- wikidata QID via pageprops.wikibase_item
- categories (sampled, configurable)
- optional outlinks sample (for link-mining / graph expansion)
- pageviews stats (total/mean/quantiles for last N days)

Notes:
- MediaWiki API titles/pageids are limited to 50 per query for normal clients.
- Pageviews is per-article endpoint, so it is parallelized with async concurrency.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import aiohttp


# ---------- fast JSON dumps (optional) ----------
try:
    import orjson  # type: ignore

    def json_dumps(obj: Any) -> str:
        return orjson.dumps(obj).decode("utf-8")
except Exception:

    def json_dumps(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False)


# ---------- endpoints ----------
def mw_api_endpoint(lang: str) -> str:
    return f"https://{lang}.wikipedia.org/w/api.php"


PAGEVIEWS_BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"


# ---------- logging ----------
def setup_logging(log_path: Path, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("wiki_miner")
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


# ---------- config ----------
@dataclass(frozen=True)
class MinerConfig:
    lang: str
    user_agent: str

    # mining target
    n_pages: int
    seed_mode: str  # random | file | category

    # seed options
    input_titles_path: Optional[Path] = None
    category_title: Optional[str] = (
        None  # e.g. "Категория:Физика" (localized namespace)
    )

    # batching and sampling
    mw_batch_size: int = 50
    max_categories: int = 50
    outlinks_sample: int = 0  # 0 disables
    expand_from_outlinks: bool = False  # if True, enqueue outlinks to mine more pages

    # pageviews options
    pv_days: int = 90
    pv_access: str = "all-access"
    pv_agent: str = "user"  # user | all-agents | spider | bot

    # concurrency
    mw_concurrency: int = 4
    pv_concurrency: int = 40

    # io
    out_path: Path = Path("wiki_meta.jsonl")
    errors_path: Path = Path("wiki_meta.errors.jsonl")
    resume: bool = True

    # safety throttles
    request_timeout_s: int = 30
    max_retries: int = 5


# ---------- async http client with retries ----------
class AsyncHTTP:
    def __init__(
        self, session: aiohttp.ClientSession, cfg: MinerConfig, logger: logging.Logger
    ):
        self.session = session
        self.cfg = cfg
        self.logger = logger

    async def get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        base_headers = {"User-Agent": self.cfg.user_agent, "Accept": "application/json"}
        if headers:
            base_headers.update(headers)

        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                async with self.session.get(
                    url,
                    params=params,
                    headers=base_headers,
                    timeout=aiohttp.ClientTimeout(total=self.cfg.request_timeout_s),
                ) as resp:
                    if resp.status == 429:
                        retry_after = resp.headers.get("Retry-After")
                        sleep_s = (
                            int(retry_after)
                            if (retry_after and retry_after.isdigit())
                            else min(60, 2**attempt)
                        )
                        self.logger.warning(
                            f"429 Too Many Requests: sleeping {sleep_s}s (attempt {attempt})"
                        )
                        await asyncio.sleep(sleep_s)
                        continue

                    if 500 <= resp.status < 600:
                        sleep_s = min(60, 2**attempt)
                        self.logger.warning(
                            f"{resp.status} server error: sleeping {sleep_s}s (attempt {attempt})"
                        )
                        await asyncio.sleep(sleep_s)
                        continue

                    resp.raise_for_status()
                    return await resp.json()

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                sleep_s = min(60, 2**attempt)
                self.logger.warning(
                    f"Request failed: {type(e).__name__}: {e} | sleeping {sleep_s}s (attempt {attempt})"
                )
                await asyncio.sleep(sleep_s)

        raise RuntimeError(f"Failed after {self.cfg.max_retries} retries: {url}")


# ---------- MediaWiki helpers ----------
async def mw_query(
    http: AsyncHTTP, lang: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    base = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "redirects": 1,
    }
    merged = dict(base)
    merged.update(params)
    return await http.get_json(mw_api_endpoint(lang), params=merged)


async def mw_query_with_continue(
    http: AsyncHTTP, lang: str, params: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Fetches all batches for a query that uses MediaWiki 'continue'.
    Returns list of response payloads.
    """
    out: List[Dict[str, Any]] = []
    cur = dict(params)
    while True:
        data = await mw_query(http, lang, cur)
        out.append(data)
        cont = data.get("continue")
        if not cont:
            break
        cur.update(cont)
    return out


def chunked(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i : i + n] for i in range(0, len(xs), n)]


# ---------- seeders ----------
async def seed_random_titles(http: AsyncHTTP, cfg: MinerConfig, need: int) -> List[str]:
    """
    Uses list=random to fetch titles from main namespace.
    """
    titles: List[str] = []
    while len(titles) < need:
        batch_n = min(500, need - len(titles))
        data = await mw_query(
            http,
            cfg.lang,
            {
                "list": "random",
                "rnnamespace": 0,
                "rnlimit": batch_n,
            },
        )
        items = (data.get("query", {}) or {}).get("random", []) or []
        titles.extend([it["title"] for it in items if "title" in it])
    return titles


async def seed_titles_from_file(path: Path) -> List[str]:
    titles: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                titles.append(t)
    return titles


async def seed_titles_from_category(
    http: AsyncHTTP, cfg: MinerConfig, category_title: str, limit: int
) -> List[str]:
    """
    Uses list=categorymembers to fetch pages from a category (namespace 0).
    """
    titles: List[str] = []
    params = {
        "list": "categorymembers",
        "cmtitle": category_title,
        "cmnamespace": 0,
        "cmlimit": "max",
    }
    batches = await mw_query_with_continue(http, cfg.lang, params)
    for b in batches:
        cms = (b.get("query", {}) or {}).get("categorymembers", []) or []
        for it in cms:
            if "title" in it:
                titles.append(it["title"])
                if len(titles) >= limit:
                    return titles
    return titles


# ---------- collectors ----------
def parse_mw_pages(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    pages = (payload.get("query", {}) or {}).get("pages", []) or []
    out: List[Dict[str, Any]] = []
    for p in pages:
        if p.get("missing"):
            continue
        pageprops = p.get("pageprops") or {}
        cats = p.get("categories") or []
        out.append(
            {
                "title": p.get("title"),
                "pageid": p.get("pageid"),
                "fullurl": p.get("fullurl"),
                "canonicalurl": p.get("canonicalurl"),
                "length": p.get("length"),
                "lastrevid": p.get("lastrevid"),
                "touched": p.get("touched"),
                "qid": pageprops.get("wikibase_item"),
                "is_disambiguation": bool(pageprops.get("disambiguation")),
                "categories": [
                    c.get("title")
                    for c in cats
                    if isinstance(c, dict) and c.get("title")
                ],
            }
        )
    return out


async def fetch_meta_batch(
    http: AsyncHTTP, cfg: MinerConfig, titles: List[str]
) -> List[Dict[str, Any]]:
    """
    One batched MW request: info + pageprops + categories.
    Categories are sampled via cllimit/max_categories (no continuation, by design).
    """
    titles_str = "|".join(titles)
    payload = await mw_query(
        http,
        cfg.lang,
        {
            "titles": titles_str,
            "prop": "info|pageprops|categories",
            "inprop": "url",
            "ppprop": "wikibase_item|disambiguation",
            "clshow": "!hidden",
            "cllimit": min(
                500, max(1, cfg.max_categories)
            ),  # MW caps cllimit; we still keep cfg.max_categories downstream
        },
    )
    records = parse_mw_pages(payload)
    # enforce max_categories strictly (server may return up to cllimit)
    for r in records:
        r["categories"] = (r.get("categories") or [])[: cfg.max_categories]
    return records


async def fetch_outlinks_sample_batch(
    http: AsyncHTTP, cfg: MinerConfig, titles: List[str], sample_k: int
) -> Dict[str, List[str]]:
    """
    One batched MW request: prop=links for multiple titles.
    Returns mapping: title -> [outlink_title...], truncated to sample_k.
    """
    if sample_k <= 0:
        return {}

    titles_str = "|".join(titles)
    payload = await mw_query(
        http,
        cfg.lang,
        {
            "titles": titles_str,
            "prop": "links",
            "plnamespace": 0,
            "pllimit": min(500, sample_k),
        },
    )

    pages = (payload.get("query", {}) or {}).get("pages", []) or []
    out: Dict[str, List[str]] = {}
    for p in pages:
        t = p.get("title")
        links = p.get("links") or []
        out[t] = [
            x.get("title") for x in links if isinstance(x, dict) and x.get("title")
        ][:sample_k]
    return out


# ---------- pageviews ----------
def _pv_date_range_utc(days: int) -> Tuple[str, str]:
    """
    Pageviews can lag for the latest day; we end at yesterday UTC.
    """
    end = dt.datetime.utcnow().date() - dt.timedelta(days=1)
    start = end - dt.timedelta(days=days - 1)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


async def fetch_pageviews(
    http: AsyncHTTP, cfg: MinerConfig, title: str
) -> Dict[str, Any]:
    project = f"{cfg.lang}.wikipedia.org"
    article = quote(title.replace(" ", "_"), safe="")
    start_s, end_s = _pv_date_range_utc(cfg.pv_days)

    url = f"{PAGEVIEWS_BASE}/{project}/{cfg.pv_access}/{cfg.pv_agent}/{article}/daily/{start_s}/{end_s}"
    data = await http.get_json(url)

    items = data.get("items") or []
    views = [it.get("views", 0) for it in items if isinstance(it, dict)]
    views = [int(v) for v in views if v is not None]

    if not views:
        return {
            "pv_days": cfg.pv_days,
            "pv_total": 0,
            "pv_mean": 0.0,
            "pv_p50": 0,
            "pv_p95": 0,
            "pv_last7": 0,
        }

    views_sorted = sorted(views)
    p50 = views_sorted[len(views_sorted) // 2]
    p95 = views_sorted[
        min(len(views_sorted) - 1, int(math.floor(0.95 * (len(views_sorted) - 1))))
    ]

    return {
        "pv_days": cfg.pv_days,
        "pv_total": int(sum(views)),
        "pv_mean": float(sum(views)) / float(len(views)),
        "pv_p50": int(p50),
        "pv_p95": int(p95),
        "pv_last7": int(sum(views[-7:])) if len(views) >= 7 else int(sum(views)),
    }


async def gather_limited(coros: List[asyncio.Future], limit: int) -> List[Any]:
    sem = asyncio.Semaphore(max(1, limit))

    async def run_one(c):
        async with sem:
            return await c

    return await asyncio.gather(*[run_one(c) for c in coros], return_exceptions=True)


# ---------- writer ----------
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


def load_seen_titles(out_path: Path, logger: logging.Logger) -> set:
    seen = set()
    if not out_path.exists():
        return seen

    n = 0
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                t = obj.get("title")
                if t:
                    seen.add(t)
                n += 1
            except Exception:
                continue

    logger.info(
        f"Resume: loaded {len(seen)} seen titles from {out_path} (lines scanned: {n})"
    )
    return seen


# ---------- main mining loop ----------
async def run_mine(cfg: MinerConfig, logger: logging.Logger) -> None:
    connector = aiohttp.TCPConnector(
        limit=cfg.pv_concurrency + cfg.mw_concurrency + 20, ttl_dns_cache=300
    )
    async with aiohttp.ClientSession(connector=connector) as session:
        http = AsyncHTTP(session, cfg, logger)

        seen_titles = load_seen_titles(cfg.out_path, logger) if cfg.resume else set()
        frontier: List[str] = []

        # seed initial frontier
        if cfg.seed_mode == "file":
            assert cfg.input_titles_path is not None
            seed = await seed_titles_from_file(cfg.input_titles_path)
            frontier.extend(seed)
        elif cfg.seed_mode == "category":
            assert cfg.category_title is not None
            seed = await seed_titles_from_category(
                http, cfg, cfg.category_title, limit=cfg.n_pages * 3
            )
            frontier.extend(seed)
        else:
            # random
            frontier.extend(
                await seed_random_titles(http, cfg, need=min(cfg.n_pages * 2, 5000))
            )

        # shuffle to avoid any ordering artifacts
        random.shuffle(frontier)

        out_q: asyncio.Queue = asyncio.Queue(maxsize=20000)
        writer = asyncio.create_task(
            writer_task(cfg.out_path, cfg.errors_path, out_q, logger)
        )

        started = time.time()
        processed = 0
        mw_calls = 0
        pv_calls = 0
        errors = 0

        # keep mining until cfg.n_pages records written
        while processed < cfg.n_pages:
            # refill frontier if low and random mode
            if cfg.seed_mode == "random" and len(frontier) < cfg.mw_batch_size * 5:
                new_titles = await seed_random_titles(http, cfg, need=2000)
                for t in new_titles:
                    if t not in seen_titles:
                        frontier.append(t)
                random.shuffle(frontier)

            # pop batch
            batch: List[str] = []
            while (
                frontier
                and len(batch) < cfg.mw_batch_size
                and processed + len(batch) < cfg.n_pages
            ):
                t = frontier.pop()
                if t in seen_titles:
                    continue
                seen_titles.add(t)
                batch.append(t)

            if not batch:
                logger.warning("Frontier exhausted. Stopping early.")
                break

            # fetch meta (batched)
            try:
                records = await fetch_meta_batch(http, cfg, batch)
                mw_calls += 1
            except Exception as e:
                errors += len(batch)
                for t in batch:
                    await out_q.put(
                        {
                            "_error": True,
                            "stage": "mw_meta",
                            "title": t,
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )
                continue

            # optional outlinks sample (batched)
            outlinks_map: Dict[str, List[str]] = {}
            if cfg.outlinks_sample > 0:
                try:
                    outlinks_map = await fetch_outlinks_sample_batch(
                        http, cfg, [r["title"] for r in records], cfg.outlinks_sample
                    )
                    mw_calls += 1
                except Exception as e:
                    logger.warning(f"Outlinks sample failed: {type(e).__name__}: {e}")

            # attach outlinks, optionally expand frontier
            if cfg.outlinks_sample > 0:
                for r in records:
                    ol = outlinks_map.get(r["title"], [])
                    r["outlinks_sample"] = ol
                    if cfg.expand_from_outlinks:
                        for t2 in ol:
                            if t2 not in seen_titles:
                                frontier.append(t2)

            # pageviews in parallel (per-article)
            pv_coros = [fetch_pageviews(http, cfg, r["title"]) for r in records]
            pv_results = await gather_limited(pv_coros, limit=cfg.pv_concurrency)
            pv_calls += len(records)

            # merge + emit
            now_iso = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
            for r, pv in zip(records, pv_results):
                out_rec = dict(r)
                out_rec["mined_at"] = now_iso

                if isinstance(pv, Exception):
                    errors += 1
                    await out_q.put(
                        {
                            "_error": True,
                            "stage": "pageviews",
                            "title": r.get("title"),
                            "error": f"{type(pv).__name__}: {pv}",
                        }
                    )
                    # still write base record without pageviews
                    await out_q.put(out_rec)
                else:
                    out_rec.update(pv)
                    await out_q.put(out_rec)

                processed += 1
                if processed >= cfg.n_pages:
                    break

            # periodic logging
            if processed % 500 == 0:
                elapsed = max(1e-6, time.time() - started)
                rate = processed / elapsed
                logger.info(
                    f"progress={processed}/{cfg.n_pages} | rate={rate:.1f} pages/s | "
                    f"mw_calls={mw_calls} | pv_calls={pv_calls} | errors={errors} | frontier={len(frontier)}"
                )

        # finish writer
        await out_q.put(None)
        await out_q.join()
        await writer

        elapsed = max(1e-6, time.time() - started)
        logger.info(
            f"Done: processed={processed} | elapsed_s={elapsed:.1f} | rate={processed / elapsed:.2f} pages/s | "
            f"mw_calls={mw_calls} | pv_calls={pv_calls} | errors={errors}"
        )


def build_cfg_from_args(args: argparse.Namespace) -> MinerConfig:
    return MinerConfig(
        lang=args.lang,
        user_agent=args.user_agent,
        n_pages=args.n_pages,
        seed_mode=args.seed_mode,
        input_titles_path=Path(args.input_titles) if args.input_titles else None,
        category_title=args.category,
        mw_batch_size=args.mw_batch_size,
        max_categories=args.max_categories,
        outlinks_sample=args.outlinks_sample,
        expand_from_outlinks=args.expand_from_outlinks,
        pv_days=args.pv_days,
        pv_access=args.pv_access,
        pv_agent=args.pv_agent,
        mw_concurrency=args.mw_concurrency,
        pv_concurrency=args.pv_concurrency,
        out_path=Path(args.out),
        errors_path=Path(args.errors_out),
        resume=not args.no_resume,
        request_timeout_s=args.timeout_s,
        max_retries=args.max_retries,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="ru")
    p.add_argument("--n-pages", type=int, default=2000)

    p.add_argument(
        "--seed-mode", choices=["random", "file", "category"], default="random"
    )
    p.add_argument(
        "--input-titles", default=None, help="Path to file with titles (one per line)"
    )
    p.add_argument(
        "--category", default=None, help='Category title, e.g. "Категория:Физика"'
    )

    p.add_argument("--mw-batch-size", type=int, default=50)
    p.add_argument("--max-categories", type=int, default=50)

    p.add_argument("--outlinks-sample", type=int, default=0)
    p.add_argument("--expand-from-outlinks", action="store_true")

    p.add_argument("--pv-days", type=int, default=90)
    p.add_argument("--pv-access", default="all-access")
    p.add_argument("--pv-agent", default="user", help="user|all-agents|spider|bot")

    p.add_argument("--mw-concurrency", type=int, default=4)
    p.add_argument("--pv-concurrency", type=int, default=40)

    p.add_argument("--out", default="wiki_meta.jsonl")
    p.add_argument("--errors-out", default="wiki_meta.errors.jsonl")

    p.add_argument("--no-resume", action="store_true")

    p.add_argument("--timeout-s", type=int, default=30)
    p.add_argument("--max-retries", type=int, default=5)

    p.add_argument("--log", default="wiki_miner.log")
    p.add_argument("--log-level", default="INFO")

    p.add_argument(
        "--user-agent",
        default="WikiMetaMiner/0.2 (contact: you@example.com)",
        help="Set a descriptive UA with contact info for responsible use.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)
    logger = setup_logging(log_path, level=args.log_level)

    cfg = build_cfg_from_args(args)

    # basic validation
    if cfg.seed_mode == "file" and not cfg.input_titles_path:
        raise SystemExit("--seed-mode file requires --input-titles")
    if cfg.seed_mode == "category" and not cfg.category_title:
        raise SystemExit("--seed-mode category requires --category")

    logger.info(f"Starting with cfg={cfg}")
    asyncio.run(run_mine(cfg, logger))


if __name__ == "__main__":
    main()
