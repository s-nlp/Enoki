#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    import orjson  # type: ignore

    def json_dumps(obj: Any) -> str:
        return orjson.dumps(obj).decode("utf-8")
except Exception:

    def json_dumps(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json_dumps(obj) + "\n")


def safe_list(x) -> List[Any]:
    return x if isinstance(x, list) else []


def truncate_text(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip()


@dataclass(frozen=True)
class ParaRecord:
    para_id: str
    para_index: int
    pageid: int
    title: str
    text: str


def collect_questions_files(patterns: List[str]) -> List[Path]:
    files: List[Path] = []
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            fp = Path(p)
            if fp.is_file():
                files.append(fp)
    files = sorted(set(files))
    return files


def collect_needed_para_ids(q_files: List[Path]) -> Set[str]:
    needed: Set[str] = set()
    for qf in q_files:
        for rec in iter_jsonl(qf):
            for pid in safe_list(rec.get("para_ids")):
                if pid is None:
                    continue
                needed.add(str(pid))
    return needed


def load_paragraphs_subset(
    paragraphs_path: Path,
    needed_para_ids: Set[str],
    *,
    max_para_chars: int,
) -> Dict[str, ParaRecord]:
    """
    Streams paragraphs.jsonl and keeps only requested para_ids.
    """
    out: Dict[str, ParaRecord] = {}
    if not paragraphs_path.exists() or not needed_para_ids:
        return out

    for rec in iter_jsonl(paragraphs_path):
        pid = rec.get("para_id")
        if pid is None:
            continue
        pid_s = str(pid)
        if pid_s not in needed_para_ids:
            continue

        try:
            para_index = int(rec.get("para_index", 0))
        except Exception:
            para_index = 0

        try:
            pageid = int(rec.get("pageid", 0))
        except Exception:
            pageid = 0

        title = str(rec.get("title") or "")
        text = str(rec.get("para_text") or "")
        text = truncate_text(text, max_para_chars)

        out[pid_s] = ParaRecord(
            para_id=pid_s,
            para_index=para_index,
            pageid=pageid,
            title=title,
            text=text,
        )

        if len(out) >= len(needed_para_ids):
            break

    return out


def build_context_from_para_ids(
    para_ids: List[str],
    para_map: Dict[str, ParaRecord],
    *,
    join_with: str = "\n\n",
) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    """
    Returns:
      paragraphs: list[{para_id, para_index, text}]
      context_text: joined
      missing_para_ids: list[str]
    """
    missing: List[str] = []
    paras: List[ParaRecord] = []

    for pid in para_ids:
        pid_s = str(pid)
        pr = para_map.get(pid_s)
        if pr is None:
            missing.append(pid_s)
        else:
            paras.append(pr)

    paras_sorted = sorted(paras, key=lambda x: x.para_index)

    paragraphs = [
        {"para_id": p.para_id, "para_index": p.para_index, "para_text": p.text}
        for p in paras_sorted
    ]
    context_text = join_with.join([p.text for p in paras_sorted]).strip()

    return paragraphs, context_text, missing


def merge_one_questions_file(
    q_path: Path,
    out_path: Path,
    para_map: Dict[str, ParaRecord],
    *,
    max_context_chars: int,
) -> Dict[str, Any]:
    n_in = 0
    n_out = 0
    n_missing = 0

    for rec in iter_jsonl(q_path):
        n_in += 1
        para_ids = [str(x) for x in safe_list(rec.get("para_ids")) if x is not None]

        paragraphs, context_text, missing = build_context_from_para_ids(
            para_ids,
            para_map,
        )
        if missing:
            n_missing += 1

        context_text = truncate_text(context_text, max_context_chars)

        out_rec = dict(rec)
        out_rec["context_paragraphs"] = paragraphs
        out_rec["context_text"] = context_text
        if missing:
            out_rec["missing_para_ids"] = missing

        append_jsonl(out_path, out_rec)
        n_out += 1

    return {
        "questions_file": str(q_path),
        "out_file": str(out_path),
        "n_in": n_in,
        "n_out": n_out,
        "n_with_missing_paras": n_missing,
    }


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--questions",
        nargs="+",
        required=True,
        help="One or more glob patterns for questions JSONL, e.g. data/wiki/en/qa_sweep/questions__*.jsonl",
    )
    ap.add_argument(
        "--paragraphs",
        required=True,
        help="Path to paragraphs.jsonl for the same language.",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Output directory. Merged files will be written here.",
    )

    ap.add_argument(
        "--combine",
        action="store_true",
        help="If set, write a single merged_all.jsonl for all questions files. Otherwise, one merged file per input file.",
    )

    ap.add_argument(
        "--max-para-chars",
        type=int,
        default=4000,
        help="Truncate each paragraph text to this many chars in memory/output (0 = no limit).",
    )
    ap.add_argument(
        "--max-context-chars",
        type=int,
        default=12000,
        help="Truncate joined context_text to this many chars (0 = no limit).",
    )

    ap.add_argument(
        "--manifest",
        default=None,
        help="Optional path to write a JSON manifest with counts.",
    )

    args = ap.parse_args()

    q_files = collect_questions_files(args.questions)
    if not q_files:
        raise SystemExit("No questions files matched the provided patterns.")

    paragraphs_path = Path(args.paragraphs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    needed_para_ids = collect_needed_para_ids(q_files)
    para_map = load_paragraphs_subset(
        paragraphs_path,
        needed_para_ids,
        max_para_chars=int(args.max_para_chars),
    )

    manifest: Dict[str, Any] = {
        "questions_files": [str(p) for p in q_files],
        "paragraphs_path": str(paragraphs_path),
        "n_questions_files": len(q_files),
        "n_needed_para_ids": len(needed_para_ids),
        "n_loaded_paras": len(para_map),
        "results": [],
    }

    if args.combine:
        out_path = out_dir / "merged_all.jsonl"
        if out_path.exists():
            out_path.unlink()

        # Merge each questions file into the same output
        for qf in q_files:
            stats = merge_one_questions_file(
                qf,
                out_path,
                para_map,
                max_context_chars=int(args.max_context_chars),
            )
            manifest["results"].append(stats)
    else:
        for qf in q_files:
            out_path = out_dir / (qf.stem + ".merged.jsonl")
            if out_path.exists():
                out_path.unlink()

            stats = merge_one_questions_file(
                qf,
                out_path,
                para_map,
                max_context_chars=int(args.max_context_chars),
            )
            manifest["results"].append(stats)

    if args.manifest:
        mpath = Path(args.manifest)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        with mpath.open("w", encoding="utf-8") as f:
            f.write(json_dumps(manifest))

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
