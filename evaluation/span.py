"""
Span-level hallucination detection evaluation.

Supports: PsiloQA, Mushroom, RAGTruth datasets.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
from functools import partial

import pandas as pd
from tqdm import tqdm

from evaluation.common import setup_logging, load_fact_extractor, load_decontextualizer, print_header
from evaluation.metrics import calculate_span_f1, print_span_metrics_summary
from evaluation.dataset_loaders import load_psiloqa_dataset, load_mushroom_dataset, load_ragtruth_dataset
from nli import check_nli_batch_fast, score_facts_with_nli
from fact_alignment import normalize_fact_spans_to_orig, dedupe_fact_scores_norm


def evaluate_span_dataset(
    data: List[Dict],
    extractor,
    decontextualizer: Optional,
    nli_method: str,
    max_length: int,
    threshold: float,
    hall_prob_mode: str,
) -> tuple[List[List[List[int]]], List[List[List[int]]]]:
    """
    Run fact extraction and NLI scoring on span-level dataset.

    Returns:
        (golds, preds) - Lists of span lists
    """
    golds = []
    preds = []

    progress_bar = tqdm(
        data,
        desc=f"Evaluating with {nli_method}",
        unit="sample",
        ncols=100,
        disable=False,
        file=sys.stderr
    )
    for row in progress_bar:
        # Decontextualize answer
        if decontextualizer is not None:
            decontext_result = decontextualizer.decontextualize(row["answer"])
        else:
            decontext_result = {"resolved_text": row["answer"], "replacements": []}

        # Extract granular facts
        try:
            granular_facts = extractor.extract_granular_facts(decontext_result["resolved_text"])
        except Exception as e:
            print(f"Warning: Failed to extract facts: {e}")
            granular_facts = []

        # Update progress with info
        progress_bar.set_postfix({
            'facts': len(granular_facts),
            'gold_spans': len(row["labels"]) if "labels" in row else 0
        })

        # Score facts with NLI
        if granular_facts:
            fact_scores = score_facts_with_nli(
                context=row["context"],
                granular_facts=granular_facts,
                check_nli_batch_fn=partial(
                    check_nli_batch_fast,
                    max_length=max_length,
                    method=nli_method
                ),
                chunk_size=32,
                hall_prob_mode=hall_prob_mode,
            )

            # Filter out predicates
            fact_scores = [f for f in fact_scores if f.get('span_kind') != 'predicate']

            # Normalize spans back to original text
            fact_scores_norm, _ = normalize_fact_spans_to_orig(
                orig_text=row["answer"],
                resolved_text=decontext_result["resolved_text"],
                replacements=decontext_result["replacements"],
                fact_scores=fact_scores,
            )
            fact_scores_norm = dedupe_fact_scores_norm(fact_scores_norm)
        else:
            fact_scores_norm = []

        # Convert to span predictions based on threshold
        pred_spans = []
        for fs in fact_scores_norm:
            if fs.get('hall_prob', 0) > threshold:
                pred_spans.append([fs['orig_span_start'], fs['orig_span_end']])

        golds.append(row["labels"])
        preds.append(pred_spans)

    return golds, preds


def run_span_evaluation(
    dataset: Optional[str],
    method: str,
    data_dir: str,
    output_dir: str,
    max_length: int,
    threshold: float,
    hall_prob_mode: str,
    coref: bool,
    all_datasets: bool,
):
    """Run span-level evaluation."""
    setup_logging()

    # Determine datasets to evaluate
    if all_datasets:
        datasets_to_run = ["psiloqa", "mushroom", "ragtruth"]
    else:
        datasets_to_run = [dataset]

    print_header("Span-Level Hallucination Detection Evaluation")
    print(f"NLI method: {method}")
    print(f"Coreference resolution: {'ENABLED' if coref else 'DISABLED'}")
    print(f"Will evaluate {len(datasets_to_run)} dataset(s): {', '.join(datasets_to_run)}")
    print("=" * 70)

    # Load models
    extractor = load_fact_extractor()
    decontextualizer = load_decontextualizer(coref)

    # Dataset loaders
    loaders = {
        "psiloqa": lambda: load_psiloqa_dataset(),
        "mushroom": lambda: load_mushroom_dataset(data_dir),
        "ragtruth": lambda: load_ragtruth_dataset(),
    }

    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run evaluation on each dataset
    for ds_name in datasets_to_run:
        print_header(f"Evaluating on {ds_name}")

        # Load dataset
        try:
            data = loaders[ds_name]()
            print(f"Loaded {len(data)} examples")
        except Exception as e:
            print(f"Error loading {ds_name}: {e}")
            continue

        # Run evaluation
        golds, preds = evaluate_span_dataset(
            data=data,
            extractor=extractor,
            decontextualizer=decontextualizer,
            nli_method=method,
            max_length=max_length,
            threshold=threshold,
            hall_prob_mode=hall_prob_mode,
        )

        # Save predictions CSV
        # Format: method[_coref][_mode]_dataset.csv
        method_name = method
        if coref:
            method_name += "_coref"
        if hall_prob_mode != "default":
            method_name += f"_{hall_prob_mode}"

        output_name = f"{method_name}_{ds_name}.csv"
        output_file = output_path / output_name

        df = pd.DataFrame({'gold': golds, 'pred': preds})
        df.to_csv(output_file, index=False)
        print(f"Saved predictions to {output_file}")

        # Calculate and print metrics
        metrics = calculate_span_f1(golds, preds)
        print_header(f"Metrics for {ds_name}")
        print_span_metrics_summary(metrics)
