"""
Random false-positive error analysis for span-level predictions.

A "false positive" here is a predicted hallucination span (hall_prob > threshold)
that does NOT overlap with any gold span.  These are the interesting cases:
either the model is wrong, OR the annotation missed a real hallucination.

Output CSV columns:
  sample_id      – index in dataset
  hall_prob      – model score for this span
  predicted_text – the predicted span text
  gold_texts     – gold hallucination spans in this sample (pipe-separated)
  answer_marked  – full answer with [[ ]] around predicted span, {{ }} around gold spans
  context_head   – first 400 chars of the context (enough to spot the issue)
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end <= b_start or a_start >= b_end)


def _mark_answer(answer: str, pred_span, gold_spans) -> str:
    """Return answer text with predicted span in [[ ]] and gold spans in {{ }}."""
    # Collect all markers sorted by position
    events = []
    ps, pe = pred_span
    events.append((ps, "open",  "pred"))
    events.append((pe, "close", "pred"))
    for gs, ge in gold_spans:
        events.append((gs, "open",  "gold"))
        events.append((ge, "close", "gold"))

    # Sort: position asc; within same position close before open so tags don't cross
    events.sort(key=lambda x: (x[0], 0 if x[1] == "close" else 1))

    result = []
    prev = 0
    for pos, kind, tag in events:
        result.append(answer[prev:pos])
        prev = pos
        if kind == "open":
            result.append("[[" if tag == "pred" else "{{")
        else:
            result.append("]]" if tag == "pred" else "}}")
    result.append(answer[prev:])
    return "".join(result)


def collect_false_positives(
    pred_samples: List[Dict],
    dataset: List[Dict],
    threshold: float,
) -> List[Dict[str, Any]]:
    """Return one record per predicted-hall span with no gold overlap."""
    results = []
    for idx, (pred, row) in enumerate(zip(pred_samples, dataset)):
        answer = row["answer"]
        gold_spans = pred.get("gold_spans", [])
        fact_spans = pred.get("fact_spans", [])

        gold_texts = [answer[gs:ge] for gs, ge in gold_spans if gs < ge <= len(answer)]

        for fs in fact_spans:
            hp = float(fs.get("hall_prob", 0.0))
            if hp <= threshold:
                continue
            ps, pe = int(fs["start"]), int(fs["end"])
            if ps >= pe or pe > len(answer):
                continue

            overlaps_any_gold = any(
                _overlaps(ps, pe, gs[0], gs[1]) for gs in gold_spans
            )
            if overlaps_any_gold:
                continue

            results.append({
                "sample_id":      idx,
                "hall_prob":      round(hp, 4),
                "predicted_text": answer[ps:pe],
                "gold_texts":     " | ".join(gold_texts) if gold_texts else "(none)",
                "answer_marked":  _mark_answer(answer, (ps, pe), gold_spans),
                "context_head":   row.get("context", "")[:400],
            })

    return results


def run_error_analysis(
    preds_path: Path,
    dataset_loader: Callable[[], List[Dict]],
    threshold: float,
    n_samples: int,
    output_path: Path,
    seed: int = 42,
) -> None:
    """
    Sample false-positive spans and write a CSV for manual review.

    Args:
        preds_path:     Path to *_preds.json file.
        dataset_loader: Zero-arg callable that returns the original dataset list.
        threshold:      hall_prob threshold used to call a span a hallucination.
        n_samples:      How many false-positive spans to include in the output.
        output_path:    Where to write the CSV.
        seed:           Random seed for reproducibility.
    """
    from evaluation.predictions_io import load_raw_predictions

    raw = load_raw_predictions(preds_path)
    pred_samples = raw["samples"]

    print(f"Loading dataset for original texts...")
    dataset = dataset_loader()

    if len(dataset) != len(pred_samples):
        print(
            f"Warning: dataset size ({len(dataset)}) != preds size ({len(pred_samples)}). "
            "Truncating to shorter."
        )
        n = min(len(dataset), len(pred_samples))
        dataset = dataset[:n]
        pred_samples = pred_samples[:n]

    fps = collect_false_positives(pred_samples, dataset, threshold)
    print(f"Found {len(fps)} false-positive spans at threshold={threshold}")

    random.seed(seed)
    if len(fps) > n_samples:
        sampled = random.sample(fps, n_samples)
    else:
        sampled = fps
        print(f"(fewer than {n_samples} available, using all {len(sampled)})")

    # Sort by hall_prob descending so highest-confidence errors come first
    sampled.sort(key=lambda x: x["hall_prob"], reverse=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["sample_id", "hall_prob", "predicted_text",
                  "gold_texts", "answer_marked", "context_head"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled)

    print(f"Saved {len(sampled)} examples to {output_path}")
