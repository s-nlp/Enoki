#!/usr/bin/env python3
"""
Create balanced train/test split for Enoki dataset.

Requirements:
- Train/test contexts do NOT intersect
- Test: ~2k balanced examples (from contexts with all models)
- Train: ALL remaining examples
- Test balanced by: models, pageviews, answer length, NER entity count
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
import numpy as np

def load_metadata(meta_path: Path) -> dict:
    """Load pageview metadata by pageid."""
    meta = {}
    with open(meta_path) as f:
        for line in f:
            d = json.loads(line)
            meta[d["pageid"]] = {
                "pv_total": d.get("pv_total", 0),
                "pv_mean": d.get("pv_mean", 0),
                "title": d.get("title", ""),
            }
    return meta

def load_ner_data(ner_path: Path) -> dict:
    """Load NER entity counts by question text."""
    ner = {}
    if not ner_path.exists():
        return ner
    with open(ner_path) as f:
        for line in f:
            d = json.loads(line)
            key = d.get("question", "")
            if key:
                ner[key] = {
                    "answer_entities": len(d.get("answer_entities", [])),
                    "context_entities": len(d.get("context_entities", [])),
                }
    return ner

def load_answers(answers_path: Path) -> list:
    """Load filtered answers."""
    answers = []
    with open(answers_path) as f:
        for line in f:
            d = json.loads(line)
            answers.append(d)
    return answers

def get_pv_bucket(pv_mean: float) -> str:
    """Bucket pageviews into categories."""
    if pv_mean < 100:
        return "low"
    elif pv_mean < 1000:
        return "medium"
    else:
        return "high"

def get_length_bucket(length: int) -> str:
    """Bucket answer length into categories."""
    if length < 500:
        return "short"
    elif length < 1500:
        return "medium"
    else:
        return "long"

def get_ner_bucket(count: int) -> str:
    """Bucket NER entity count into categories."""
    if count < 3:
        return "few"
    elif count < 8:
        return "some"
    else:
        return "many"

def main():
    parser = argparse.ArgumentParser(description="Create balanced train/test split")
    parser.add_argument("--data-dir", type=Path, default=Path("data/wiki/en/hallucinations"))
    parser.add_argument("--meta-path", type=Path, default=Path("data/wiki/en/meta_en.sampled20k.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--test-size", type=int, default=2000, help="Target test set size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-models", type=int, default=7, help="Min models per context for test set")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Find model directories
    models = sorted([
        d.name for d in args.data_dir.iterdir()
        if d.is_dir() and not d.name.startswith("sample")
        and (d / "answers_no_context.filtered.jsonl").exists()
    ])
    print(f"Found {len(models)} models: {models}")

    # Load metadata
    print(f"Loading metadata from {args.meta_path}...")
    meta = load_metadata(args.meta_path)
    print(f"  Loaded {len(meta)} pages")

    # Load all answers and group by context
    print("Loading answers...")
    context_data = defaultdict(lambda: {"models": set(), "examples": []})
    all_examples = []  # Store all examples for train set

    for model in models:
        answers_path = args.data_dir / model / "answers_no_context.filtered.jsonl"
        ner_path = args.data_dir / model / "ner_gliner.no_context.jsonl"

        ner_data = load_ner_data(ner_path)
        answers = load_answers(answers_path)

        for ans in answers:
            source = ans.get("source", {})
            ctx_id = source.get("context_id")
            if not ctx_id:
                continue

            pageid = source.get("pageid")
            question = ans.get("question", "")
            answer_text = ans.get("answer", "")

            # Get metadata
            page_meta = meta.get(pageid, {})
            pv_mean = page_meta.get("pv_mean", 0)

            # Get NER stats
            ner_stats = ner_data.get(question, {})
            answer_entities = ner_stats.get("answer_entities", 0)

            example = {
                "model": model,
                "question": question,
                "answer": answer_text,
                "answer_length": len(answer_text),
                "answer_entities": answer_entities,
                "context_id": ctx_id,
                "context_text": source.get("context_text", ""),
                "pv_mean": pv_mean,
                "title": source.get("title", ""),
                "source": source,
            }

            # Store in context data
            context_data[ctx_id]["models"].add(model)
            context_data[ctx_id]["pv_mean"] = pv_mean
            context_data[ctx_id]["title"] = source.get("title", "")
            context_data[ctx_id]["examples"].append(example)

            # Store all examples
            all_examples.append(example)

    print(f"  Total contexts: {len(context_data)}")
    print(f"  Total examples: {len(all_examples)}")

    # Filter contexts with all models (for balanced test set)
    full_coverage_contexts = [
        ctx_id for ctx_id, data in context_data.items()
        if len(data["models"]) >= args.min_models
    ]
    print(f"  Contexts with >= {args.min_models} models: {len(full_coverage_contexts)}")

    # Compute features for stratified sampling of test set
    context_features = []
    for ctx_id in full_coverage_contexts:
        data = context_data[ctx_id]
        avg_length = np.mean([e["answer_length"] for e in data["examples"]])
        avg_entities = np.mean([e["answer_entities"] for e in data["examples"]])

        context_features.append({
            "ctx_id": ctx_id,
            "pv_bucket": get_pv_bucket(data["pv_mean"]),
            "length_bucket": get_length_bucket(avg_length),
            "ner_bucket": get_ner_bucket(avg_entities),
            "pv_mean": data["pv_mean"],
            "avg_length": avg_length,
            "avg_entities": avg_entities,
        })

    # Create stratification key
    for cf in context_features:
        cf["strata"] = f"{cf['pv_bucket']}_{cf['length_bucket']}_{cf['ner_bucket']}"

    # Group by strata
    strata_groups = defaultdict(list)
    for cf in context_features:
        strata_groups[cf["strata"]].append(cf["ctx_id"])

    print(f"\nStrata distribution (for test sampling):")
    for strata, ctx_ids in sorted(strata_groups.items(), key=lambda x: -len(x[1])):
        print(f"  {strata}: {len(ctx_ids)}")

    # Calculate how many contexts needed for test (7 examples per context)
    test_contexts_needed = args.test_size // len(models)
    print(f"\nTest contexts needed: {test_contexts_needed} (for ~{args.test_size} examples)")

    # Stratified sampling for test set
    test_contexts = []
    for strata, ctx_ids in strata_groups.items():
        n_strata = len(ctx_ids)
        n_sample = int(np.ceil(n_strata / len(full_coverage_contexts) * test_contexts_needed))
        n_sample = min(n_sample, n_strata)
        sampled = random.sample(ctx_ids, n_sample)
        test_contexts.extend(sampled)

    # Trim to exact count
    random.shuffle(test_contexts)
    test_contexts = test_contexts[:test_contexts_needed]
    test_contexts_set = set(test_contexts)

    print(f"Sampled {len(test_contexts)} test contexts")

    # Build test set (1 example per model per context)
    test_examples = []
    for ctx_id in test_contexts:
        data = context_data[ctx_id]
        model_examples = defaultdict(list)
        for ex in data["examples"]:
            model_examples[ex["model"]].append(ex)

        for model in models:
            if model in model_examples:
                ex = random.choice(model_examples[model])
                test_examples.append({
                    "question": ex["question"],
                    "context": ex["context_text"],
                    "context_id": ctx_id,
                    "answer": ex["answer"],
                    "answer_model": model,
                    "answer_length": ex["answer_length"],
                    "answer_entities": ex["answer_entities"],
                    "pv_mean": ex["pv_mean"],
                    "title": ex["title"],
                    "mode": "no_context",
                })

    # Build train set (ALL examples from non-test contexts)
    train_examples = []
    for ex in all_examples:
        if ex["context_id"] not in test_contexts_set:
            train_examples.append({
                "question": ex["question"],
                "context": ex["context_text"],
                "context_id": ex["context_id"],
                "answer": ex["answer"],
                "answer_model": ex["model"],
                "answer_length": ex["answer_length"],
                "answer_entities": ex["answer_entities"],
                "pv_mean": ex["pv_mean"],
                "title": ex["title"],
                "mode": "no_context",
            })

    print(f"\nFinal counts:")
    print(f"  Test: {len(test_examples)} examples from {len(test_contexts)} contexts")
    print(f"  Train: {len(train_examples)} examples from {len(set(e['context_id'] for e in train_examples))} contexts")

    # Verify no overlap
    train_ctx = set(e["context_id"] for e in train_examples)
    test_ctx = set(e["context_id"] for e in test_examples)
    overlap = train_ctx & test_ctx
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping contexts!"
    print("  No context overlap between train/test")

    # Print stats per model
    print(f"\nPer-model distribution:")
    for split_name, examples in [("train", train_examples), ("test", test_examples)]:
        model_counts = defaultdict(int)
        for ex in examples:
            model_counts[ex["answer_model"]] += 1
        print(f"  {split_name}:")
        for model in models:
            print(f"    {model}: {model_counts[model]}")

    # Print stats comparison
    print(f"\nStats comparison (train vs test):")
    for split_name, examples in [("train", train_examples), ("test", test_examples)]:
        lengths = [ex["answer_length"] for ex in examples]
        entities = [ex["answer_entities"] for ex in examples]
        pvs = [ex["pv_mean"] for ex in examples]
        print(f"  {split_name}:")
        print(f"    answer_length: mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}, std={np.std(lengths):.0f}")
        print(f"    answer_entities: mean={np.mean(entities):.1f}, median={np.median(entities):.0f}, std={np.std(entities):.1f}")
        print(f"    pv_mean: mean={np.mean(pvs):.0f}, median={np.median(pvs):.0f}")

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.output_dir / "enoki_train.jsonl"
    test_path = args.output_dir / "enoki_test.jsonl"

    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(test_path, "w") as f:
        for ex in test_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nSaved:")
    print(f"  {train_path}")
    print(f"  {test_path}")

    # Save context lists for reference
    with open(args.output_dir / "train_contexts.json", "w") as f:
        json.dump(list(train_ctx), f)
    with open(args.output_dir / "test_contexts.json", "w") as f:
        json.dump(test_contexts, f)

if __name__ == "__main__":
    main()
