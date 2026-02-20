#!/usr/bin/env python3
"""
Export EnokiQA dataset with all metadata into a single JSONL per split.

Output schema per record:
  - id: unique identifier ({split}_{index})
  - split: "train" or "test"
  - title: Wikipedia article title
  - question: generated question
  - answer: model-generated answer (no_context mode)
  - answer_model: generator model name
  - answer_length: len(answer) in chars
  - paragraph_context: paragraph used for question generation
  - full_page_context: full Wikipedia article text (if available)
  - context_id: hash identifying the paragraph context

  Wikipedia metadata:
  - wiki_url: full URL to Wikipedia article
  - wiki_pageid: Wikipedia page ID
  - wiki_categories: list of Wikipedia categories
  - wiki_qid: Wikidata QID
  - wiki_article_length: article length in chars (from API)

  Pageview stats:
  - pv_mean: mean daily pageviews (90-day window)
  - pv_total: total pageviews in window
  - pv_p50: median daily pageviews
  - pv_p95: 95th percentile daily pageviews
  - popularity_tier: "low" (<100), "medium" (100-1000), "high" (>1000)

  Annotations (test only):
  - n_facts: number of extracted atomic facts
  - n_hallucinated: number of hallucinated facts (hall_prob >= 0.5)
  - hall_rate: fraction of hallucinated facts
  - mean_hall_prob: mean hallucination probability across facts
  - max_hall_prob: max hallucination probability
  - facts: list of extracted facts, each with:
      - fact: extracted fact text
      - span_text: localized span in the answer
      - span_start: character offset start
      - span_end: character offset end
      - span_kind: "argument" or "predicate"
      - entailment: E probability
      - neutral: N probability
      - contradiction: C probability
      - hall_prob: hallucination probability (N + C)
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_jsonl_index(path, key):
    """Load JSONL file and index by a key field."""
    index = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            index[rec[key]] = rec
    return index


def load_wiki_meta(path):
    """Load Wikipedia metadata, indexed by title."""
    return load_jsonl_index(path, "title")


def load_full_pages(path):
    """Load full page texts, indexed by title."""
    return load_jsonl_index(path, "title")


def load_annotations(path):
    """Load annotations, indexed by (context_id, answer_model)."""
    index = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["context_id"], rec["answer_model"])
            index[key] = rec
    return index


def popularity_tier(pv_mean):
    if pv_mean is None:
        return "unknown"
    if pv_mean < 100:
        return "low"
    elif pv_mean <= 1000:
        return "medium"
    else:
        return "high"


def clean_fact(fs):
    """Extract clean fact dict from raw fact_scores entry."""
    return {
        "fact": fs["fact"],
        "span_text": fs["span_text"],
        "span_start": fs["span_start"],
        "span_end": fs["span_end"],
        "span_kind": fs["span_kind"],
        "entailment": round(fs["entailment"], 6),
        "neutral": round(fs["neutral"], 6),
        "contradiction": round(fs["contradiction"], 6),
        "hall_prob": round(fs["hall_prob"], 6),
    }


def export_split(split_name, split_path, meta, full_pages, annotations, output_path):
    """Export a single split to JSONL."""
    records = []
    with open(split_path) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"  {split_name}: {len(records)} records")

    n_with_meta = 0
    n_with_fullpage = 0
    n_with_annotations = 0

    with open(output_path, "w") as out:
        for i, rec in enumerate(records):
            title = rec["title"]
            m = meta.get(title, {})
            fp = full_pages.get(title, {})

            # Build export record
            export = {
                "id": f"{split_name}_{i}",
                "split": split_name,
                "title": title,
                "question": rec["question"],
                "answer": rec["answer"],
                "answer_model": rec["answer_model"],
                "answer_length": len(rec["answer"]),
                "paragraph_context": rec.get("context", ""),
                "full_page_context": fp.get("text", ""),
                "context_id": rec["context_id"],
                # Wiki metadata
                "wiki_url": m.get("fullurl", ""),
                "wiki_pageid": m.get("pageid"),
                "wiki_categories": m.get("categories", []),
                "wiki_qid": m.get("qid", ""),
                "wiki_article_length": m.get("length"),
                # Pageview stats
                "pv_mean": rec.get("pv_mean") or m.get("pv_mean"),
                "pv_total": m.get("pv_total"),
                "pv_p50": m.get("pv_p50"),
                "pv_p95": m.get("pv_p95"),
                "popularity_tier": popularity_tier(
                    rec.get("pv_mean") or m.get("pv_mean")
                ),
            }

            if m:
                n_with_meta += 1
            if fp.get("text"):
                n_with_fullpage += 1

            # Annotations (test only)
            if annotations is not None:
                ann_key = (rec["context_id"], rec["answer_model"])
                ann = annotations.get(ann_key)
                if ann:
                    n_with_annotations += 1
                    export["n_facts"] = ann["n_facts"]
                    export["n_hallucinated"] = ann["n_hallucinated"]
                    export["hall_rate"] = round(ann["hall_rate"], 6)
                    export["mean_hall_prob"] = round(ann["mean_hall_prob"], 6)
                    export["max_hall_prob"] = round(ann["max_hall_prob"], 6)
                    export["facts"] = [clean_fact(fs) for fs in ann["fact_scores"]]

            out.write(json.dumps(export, ensure_ascii=False) + "\n")

    print(f"    Meta joined: {n_with_meta}/{len(records)}")
    print(f"    Full pages joined: {n_with_fullpage}/{len(records)}")
    if annotations is not None:
        print(f"    Annotations joined: {n_with_annotations}/{len(records)}")
    print(f"    Written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export EnokiQA dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "enokiqa_export",
        help="Output directory",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load shared resources
    print("Loading Wikipedia metadata...")
    meta = load_wiki_meta(ROOT / "data" / "wiki" / "en" / "meta_en.sampled20k.jsonl")
    print(f"  {len(meta)} entries")

    print("Loading full page texts...")
    full_pages = load_full_pages(ROOT / "data" / "wiki" / "en" / "full_pages.jsonl")
    print(f"  {len(full_pages)} entries")

    print("Loading annotations...")
    ann_path = (
        ROOT
        / "fact_extractor"
        / "predictions"
        / "enoki_test_fullpage_all.jsonl"
    )
    annotations = load_annotations(ann_path)
    print(f"  {len(annotations)} entries")

    # Export test (with annotations)
    print("\nExporting test split...")
    export_split(
        "test",
        ROOT / "data" / "splits" / "enoki_test.jsonl",
        meta,
        full_pages,
        annotations,
        args.output_dir / "enokiqa_test.jsonl",
    )

    # Export train (no annotations)
    print("\nExporting train split...")
    export_split(
        "train",
        ROOT / "data" / "splits" / "enoki_train.jsonl",
        meta,
        full_pages,
        None,
        args.output_dir / "enokiqa_train.jsonl",
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
