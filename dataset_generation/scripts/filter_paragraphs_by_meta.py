#!/usr/bin/env python3
"""
Filter an existing paragraphs.jsonl by keeping only pageids present in a (downsampled) meta file.
Useful when you already mined full paragraphs but later decided to sample the meta.
"""
import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="Sampled meta JSONL (defines keep set).")
    ap.add_argument("--paragraphs", required=True, help="Full paragraphs JSONL.")
    ap.add_argument("--out", required=True, help="Filtered paragraphs output JSONL.")
    ap.add_argument("--log", action="store_true", help="Print progress every 100k lines.")
    args = ap.parse_args()

    meta_path = Path(args.meta)
    para_path = Path(args.paragraphs)
    out_path = Path(args.out)

    keep_pageids = set()
    for rec in read_jsonl(meta_path):
        pid = rec.get("pageid")
        if pid is None:
            continue
        try:
            keep_pageids.add(int(pid))
        except Exception:
            continue

    kept = 0
    seen = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f_out:
        for rec in read_jsonl(para_path):
            seen += 1
            pid = rec.get("pageid")
            try:
                pid = int(pid)
            except Exception:
                continue
            if pid in keep_pageids:
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
            if args.log and seen % 100000 == 0:
                print(f"[progress] seen={seen} kept={kept}")

    print(f"finished: seen={seen} kept={kept} keep_pageids={len(keep_pageids)} out={out_path}")


if __name__ == "__main__":
    main()
