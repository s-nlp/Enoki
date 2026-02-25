#!/usr/bin/env python3
import argparse
import json
import math
random_seed_note = "Set --seed for reproducible sampling."
import random
from collections import defaultdict, Counter
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


PV_THRESHOLDS = [0, 50, 200, 800, 4000, 20000, 10**9]


def pv_bucket(v: int) -> str:
    v = int(v) if v is not None else 0
    for i in range(len(PV_THRESHOLDS) - 1):
        lo, hi = PV_THRESHOLDS[i], PV_THRESHOLDS[i + 1]
        if lo <= v < hi:
            return f"{lo}-{hi}"
    return "unknown"


TOPIC_KEYWORDS = [
    ("biography", ["births", "deaths", "people", "biography", "living people"]),
    ("geography", ["cities", "towns", "countries", "states", "geography", "places"]),
    ("history", ["history", "wars", "revolutions", "battles", "century"]),
    ("science", ["science", "scientists", "physics", "chemistry", "biology"]),
    ("technology", ["technology", "software", "hardware", "computing"]),
    ("math", ["mathematics", "math"]),
    ("culture", ["culture", "art", "architecture", "painting", "sculpture"]),
    ("literature", ["novels", "writers", "poets", "literature"]),
    ("music", ["music", "songs", "albums", "musicians", "bands"]),
    ("film", ["film", "films", "movies", "actors", "actresses", "television", "tv"]),
    ("sports", ["sports", "football", "soccer", "basketball", "baseball", "hockey"]),
    ("business", ["companies", "economy", "business", "markets"]),
    ("law", ["law", "legal"]),
    ("health", ["medicine", "medical", "health"]),
    ("education", ["universities", "schools", "education"]),
    ("religion", ["religion", "church", "saints", "islam", "bible"]),
    ("environment", ["environment", "climate", "ecology"]),
    ("transport", ["transport", "railway", "airports", "roads"]),
]


def assign_topic(categories):
    cats = [c.lower() for c in categories or []]
    for topic, keys in TOPIC_KEYWORDS:
        for k in keys:
            if any(k in c for c in cats):
                return topic
    return "other"


def allocate_quota(counts, total):
    # counts: dict key -> available
    total_avail = sum(counts.values())
    total = min(total, total_avail)
    alloc = {k: 0 for k in counts}
    remaining_total = total
    remaining_keys = list(counts.keys())
    while remaining_total > 0 and remaining_keys:
        share = remaining_total / len(remaining_keys)
        new_keys = []
        for k in remaining_keys:
            room = counts[k] - alloc[k]
            take = min(room, int(math.floor(share)))
            if take < 0:
                take = 0
            alloc[k] += take
            remaining_total -= take
            if alloc[k] < counts[k]:
                new_keys.append(k)
        # distribute leftovers one by one to groups with room
        if remaining_total > 0 and new_keys:
            for k in new_keys:
                if remaining_total == 0:
                    break
                if alloc[k] < counts[k]:
                    alloc[k] += 1
                    remaining_total -= 1
            remaining_keys = [k for k in new_keys if alloc[k] < counts[k]]
        else:
            break
    return alloc


def quantiles(vals, ps=(0, 25, 50, 75, 90, 95, 99)):
    if not vals:
        return {}
    vals = sorted(vals)
    out = {}
    for p in ps:
        idx = int((len(vals) - 1) * p / 100)
        out[p] = vals[idx]
    return out


def main():
    ap = argparse.ArgumentParser(description="Downsample wiki meta by pageviews and topic.")
    ap.add_argument("--input", required=True, help="Input meta JSONL (from wiki_miner).")
    ap.add_argument("--out", required=True, help="Output downsampled meta JSONL.")
    ap.add_argument("--titles-out", default=None, help="Optional file to save titles list.")
    ap.add_argument("--total", type=int, default=20000, help="Target total records.")
    ap.add_argument("--seed", type=int, default=42, help=random_seed_note)
    args = ap.parse_args()

    random.seed(args.seed)
    input_path = Path(args.input)
    out_path = Path(args.out)
    titles_path = Path(args.titles_out) if args.titles_out else None

    records = []
    for rec in read_jsonl(input_path):
        pv = rec.get("pv_total", 0)
        bucket = pv_bucket(pv)
        topic = assign_topic(rec.get("categories"))
        rec["_bucket"] = bucket
        rec["_topic"] = topic
        records.append(rec)

    bucket_counts = Counter(r["_bucket"] for r in records)
    bucket_alloc = allocate_quota(bucket_counts, args.total)
    topic_counts = Counter(r["_topic"] for r in records)

    selected = []
    for bucket, b_quota in bucket_alloc.items():
        bucket_recs = [r for r in records if r["_bucket"] == bucket]
        topic_counts = Counter(r["_topic"] for r in bucket_recs)
        topic_alloc = allocate_quota(topic_counts, b_quota)
        for topic, t_quota in topic_alloc.items():
            pool = [r for r in bucket_recs if r["_topic"] == topic]
            if t_quota >= len(pool):
                chosen = pool
            else:
                chosen = random.sample(pool, t_quota)
            selected.extend(chosen)

    # Trim if slightly over
    if len(selected) > args.total:
        selected = random.sample(selected, args.total)

    sel_bucket_counts = Counter(r["_bucket"] for r in selected)
    sel_topic_counts = Counter(r["_topic"] for r in selected)
    pv_all = [int(r.get("pv_total", 0) or 0) for r in records]
    pv_sel = [int(r.get("pv_total", 0) or 0) for r in selected]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec in selected:
            rec.pop("_bucket", None)
            rec.pop("_topic", None)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if titles_path:
        titles_path.parent.mkdir(parents=True, exist_ok=True)
        with titles_path.open("w") as f:
            for rec in selected:
                t = rec.get("title")
                if t:
                    f.write(str(t) + "\n")

    print(f"input_records={len(records)} selected={len(selected)} target={args.total}")
    print("bucket_counts:", dict(bucket_counts))
    print("bucket_alloc:", dict(bucket_alloc))
    print("bucket_selected:", dict(sel_bucket_counts))
    print("topic_counts:", dict(topic_counts))
    print("topic_selected:", dict(sel_topic_counts))
    print("pv_quantiles_all:", quantiles(pv_all))
    print("pv_quantiles_selected:", quantiles(pv_sel))


if __name__ == "__main__":
    main()
