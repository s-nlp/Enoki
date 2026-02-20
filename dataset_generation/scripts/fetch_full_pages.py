#!/usr/bin/env python3
"""Fetch full Wikipedia page plain text for all pages used in EnokiQA.

Uses the MediaWiki API with explaintext to get clean plain text,
then strips non-content sections (References, See also, etc.)
and section header markers.

Usage:
    python scripts/fetch_full_pages.py \
        --splits data/splits/enoki_train.jsonl data/splits/enoki_test.jsonl \
        --meta data/wiki/en/meta_en.sampled20k.jsonl \
        --output data/wiki/en/full_pages.jsonl \
        --concurrency 4
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

import aiohttp

LOG = logging.getLogger("fetch_full_pages")

MW_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "EnokiResearch/1.0 (https://github.com/enoki; research@example.com)"

# Sections whose content is not useful for factual verification
BANNED_SECTIONS = {
    "references",
    "see also",
    "external links",
    "further reading",
    "bibliography",
    "notes",
    "footnotes",
    "citations",
    "sources",
}


def clean_math_blocks(text: str) -> str:
    """Strip Wikipedia math rendering artifacts from explaintext output.

    The MediaWiki API renders math as indented single-char-per-line blocks
    followed by {\\displaystyle ...}. Example:
        \\n  \\n    \\n      \\n        f\\n        :\\n        M\\n      \\n    \\n    {\\displaystyle f\\colon M\\to N}\\n  \\n
    We replace the entire block (indented chars + displaystyle) with just
    the compact LaTeX content.
    """
    # Pattern: optional leading whitespace-only lines, then indented single-char
    # lines, then {\displaystyle ...}, then trailing whitespace
    # The indented block: lines of (spaces + short content), ending with {\displaystyle ...}
    pattern = re.compile(
        r'(?:\n\s*)*'                    # leading blank/whitespace lines
        r'(?:\n\s{2,}\S[^\n]{0,10}){2,}'  # 2+ indented short lines (single chars/symbols)
        r'\s*'                            # whitespace
        r'\{\\displaystyle\s+([^}]+)\}'   # {\displaystyle <content>}
        r'(?:\s*\n\s*)*',                # trailing whitespace
    )
    return pattern.sub(r' \1 ', text)


def clean_extract(text: str) -> str:
    """Clean Wikipedia extract: strip banned sections and header markers."""
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip math rendering artifacts before section processing
    text = clean_math_blocks(text)

    lines = text.split("\n")
    result_lines: list[str] = []
    skip_section = False
    current_level = 0  # depth of the banned section header

    for line in lines:
        # Detect section headers: == Foo ==, === Bar ===, etc.
        header_match = re.match(r"^\s*(={2,})\s*(.+?)\s*\1\s*$", line)
        if header_match:
            level = len(header_match.group(1))
            section_name = header_match.group(2).strip().lower()

            if section_name in BANNED_SECTIONS:
                skip_section = True
                current_level = level
                continue

            # If we hit a header at the same or higher level, stop skipping
            if skip_section and level <= current_level:
                skip_section = False

            if not skip_section:
                result_lines.append("")
            continue

        if skip_section:
            continue

        result_lines.append(line)

    cleaned = "\n".join(result_lines)
    # Collapse 3+ newlines into 2
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


async def fetch_one(
    session: aiohttp.ClientSession,
    pageid: int,
    sem: asyncio.Semaphore,
    max_retries: int = 5,
) -> dict | None:
    """Fetch full plain text extract for a single page."""
    params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "prop": "extracts",
        "pageids": str(pageid),
        "explaintext": 1,
        "exsectionformat": "wiki",
    }

    async with sem:
        for attempt in range(1, max_retries + 1):
            try:
                async with session.get(
                    MW_API,
                    params=params,
                    headers={"User-Agent": USER_AGENT},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 429:
                        wait = int(resp.headers.get("Retry-After", min(60, 2**attempt)))
                        LOG.warning(f"429 rate limited, waiting {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    if 500 <= resp.status < 600:
                        await asyncio.sleep(min(60, 2**attempt))
                        continue
                    resp.raise_for_status()
                    data = await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                LOG.warning(f"Error for pageid={pageid}: {e}, attempt {attempt}")
                await asyncio.sleep(min(60, 2**attempt))
                continue
            else:
                break
        else:
            LOG.error(f"Failed after {max_retries} retries for pageid={pageid}")
            return None

    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return None

    page = pages[0]
    raw = page.get("extract", "")
    cleaned = clean_extract(raw)
    return {
        "pageid": pageid,
        "title": page.get("title", ""),
        "raw_extract_length": len(raw),
        "text": cleaned,
        "text_length": len(cleaned),
    }


async def run(
    pageids_with_titles: dict[int, str],
    output_path: Path,
    concurrency: int,
):
    # Resume support
    done_pids: set[int] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    done_pids.add(json.loads(line)["pageid"])
                except Exception:
                    pass
        LOG.info(f"Resume: {len(done_pids)} pages already fetched")

    todo = [pid for pid in pageids_with_titles if pid not in done_pids]
    LOG.info(f"Total: {len(pageids_with_titles)}, todo: {len(todo)}")

    if not todo:
        LOG.info("Nothing to fetch")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    fetched = 0
    errors = 0

    async with aiohttp.ClientSession() as session:
        f_out = open(output_path, "a", encoding="utf-8")

        async def process(pid: int):
            nonlocal fetched, errors
            result = await fetch_one(session, pid, sem)
            if result:
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                fetched += 1
                if fetched % 100 == 0:
                    f_out.flush()
                    LOG.info(f"Progress: {fetched}/{len(todo)}")
            else:
                errors += 1
                LOG.warning(f"No extract: pageid={pid} title={pageids_with_titles.get(pid, '?')}")

        await asyncio.gather(*[process(pid) for pid in todo])
        f_out.flush()
        f_out.close()

    LOG.info(f"Done: {fetched} fetched, {errors} errors")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--output", default="data/wiki/en/full_pages.jsonl")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stderr,
    )

    # Collect unique titles from splits
    titles_needed: set[str] = set()
    for path in args.splits:
        with open(path) as f:
            for line in f:
                titles_needed.add(json.loads(line)["title"])
    LOG.info(f"Unique titles: {len(titles_needed)}")

    # Map title -> pageid
    title_to_pageid: dict[str, int] = {}
    with open(args.meta) as f:
        for line in f:
            rec = json.loads(line)
            if rec["title"] in titles_needed:
                title_to_pageid[rec["title"]] = rec["pageid"]

    LOG.info(f"Matched {len(title_to_pageid)}/{len(titles_needed)} titles")
    unmatched = titles_needed - set(title_to_pageid)
    if unmatched:
        LOG.warning(f"{len(unmatched)} unmatched: {list(unmatched)[:5]}")

    pageids_with_titles = {pid: t for t, pid in title_to_pageid.items()}
    asyncio.run(run(pageids_with_titles, Path(args.output), args.concurrency))

    # Stats
    out = Path(args.output)
    if out.exists():
        lengths = []
        with open(out) as f:
            for line in f:
                lengths.append(json.loads(line)["text_length"])
        if lengths:
            import statistics
            LOG.info(
                f"Stats: {len(lengths)} pages, "
                f"mean={statistics.mean(lengths):.0f}, "
                f"median={statistics.median(lengths):.0f}, "
                f"min={min(lengths)}, max={max(lengths)}"
            )


if __name__ == "__main__":
    main()
