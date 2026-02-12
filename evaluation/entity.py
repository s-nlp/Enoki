"""
Entity-level hallucination detection evaluation.

Supports: HalluEntity dataset.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
from functools import partial

import numpy as np
from tqdm import tqdm

from evaluation.common import setup_logging, load_fact_extractor, load_decontextualizer, print_header
from evaluation.metrics import calculate_entity_metrics, print_entity_metrics_summary
from evaluation.dataset_loaders import load_halluentity_dataset
from nli import check_nli_batch_fast, score_facts_with_nli
from fact_alignment import normalize_fact_spans_to_orig, dedupe_fact_scores_norm, align_fact_scores_to_entities_orig


def evaluate_entity_level(
    samples: list[dict],
    extractor,
    decontextualizer: Optional,
    nli_method: str,
    max_length: int,
    chunk_size: int,
) -> Dict[str, Any]:
    """
    Evaluate at entity level (HalluEntity).

    Args:
        samples: List of samples with 'text', 'context', 'entities', 'entity_labels'
        extractor: FactExtractor instance
        decontextualizer: Optional decontextualizer
        nli_method: NLI method to use
        max_length: Max sequence length for NLI
        chunk_size: Batch size for NLI

    Returns:
        Dict with predictions, labels, and metrics
    """
    auroc_list = []
    auprc_list = []
    all_preds = []
    all_labels = []
    fact_details = []
    skipped = 0

    print(f"Starting entity-level evaluation on {len(samples)} samples...")
    print(f"  NLI method: {nli_method}")
    print(f"  Decontextualization: {'enabled' if decontextualizer is not None else 'disabled'}")
    print(f"  Max length: {max_length}, Chunk size: {chunk_size}")
    print()

    progress_bar = tqdm(
        samples,
        desc="Evaluating samples",
        unit="sample",
        ncols=100,
        disable=False,
        file=sys.stderr
    )

    for idx, sample in enumerate(progress_bar):
        orig_text = sample['text']
        context = sample['context']
        entities = sample['entities']
        entity_labels = sample['entity_labels']

        # Update progress bar with sample info
        progress_bar.set_postfix({
            'entities': len(entities),
            'context_len': len(context),
            'skipped': skipped
        })

        # Decontextualize if needed
        if 'resolved_text' in sample and 'replacements' in sample:
            # Use pre-computed decontextualization
            resolved_text = sample['resolved_text']
            replacements = sample['replacements']
        else:
            # Decontextualize on-the-fly (if enabled)
            if decontextualizer is not None:
                decontext_result = decontextualizer.decontextualize(orig_text)
                resolved_text = decontext_result['resolved_text']
                replacements = decontext_result['replacements']
            else:
                # No decontextualization
                resolved_text = orig_text
                replacements = []

        # Extract facts from resolved text
        facts = extractor.extract_granular_facts(resolved_text)

        if not facts:
            auroc_list.append(0.0)
            auprc_list.append(0.0)
            fact_details.append({'text': orig_text, 'facts': [], 'scores': []})
            continue

        # Score with NLI
        fact_scores = score_facts_with_nli(
            context=context,
            granular_facts=facts,
            check_nli_batch_fn=partial(check_nli_batch_fast, max_length=max_length, method=nli_method),
            chunk_size=chunk_size,
            incremental_stop_threshold=1.0,
        )

        # Normalize to original text coordinates
        try:
            fact_scores_norm, _ = normalize_fact_spans_to_orig(
                orig_text=orig_text,
                resolved_text=resolved_text,
                replacements=replacements,
                fact_scores=fact_scores,
            )
            fact_scores_norm = dedupe_fact_scores_norm(fact_scores_norm)

            # Align to entities
            y_true = [0 if label else 1 for label in entity_labels]  # True=factual->0, False=hallucination->1

            ent_probs = align_fact_scores_to_entities_orig(
                orig_text=orig_text,
                ds_entities=entities,
                fact_scores_norm=fact_scores_norm,
            )
            y_score = ent_probs.tolist()

            auroc, auprc = calculate_entity_metrics(y_true, y_score)
            auroc_list.append(auroc)
            auprc_list.append(auprc)

            all_preds.extend(y_score)
            all_labels.extend(y_true)

        except Exception as e:
            skipped += 1
            auroc_list.append(0.0)
            auprc_list.append(0.0)
            if skipped <= 3:
                print(f"Skipped sample: {e}")

        fact_details.append({
            'text': orig_text,
            'facts': [str(f.facts[-1]) for f in facts] if facts else [],
            'scores': fact_scores,
        })

    return {
        'auroc_per_sample': auroc_list,
        'auprc_per_sample': auprc_list,
        'mean_auroc': np.mean(auroc_list),
        'mean_auprc': np.mean(auprc_list),
        'all_predictions': all_preds,
        'all_labels': all_labels,
        'skipped': skipped,
        'fact_details': fact_details,
    }


def run_entity_evaluation(
    dataset: str,
    method: str,
    data_dir: str,
    output_dir: str,
    cache_dir: str,
    max_length: int,
    chunk_size: int,
    coref: bool,
    force_recompute: bool,
):
    """Run entity-level evaluation."""
    setup_logging()

    print_header("Entity-Level Hallucination Detection Evaluation")
    print(f"Dataset: {dataset}")
    print(f"NLI method: {method}")
    print(f"Coreference resolution: {'ENABLED' if coref else 'DISABLED'}")
    print("=" * 70)

    # Load models
    extractor = load_fact_extractor()
    decontextualizer = load_decontextualizer(coref)

    # Load dataset
    print(f"\nLoading {dataset} dataset...")
    samples, eval_type = load_halluentity_dataset(data_dir)
    print(f"Loaded {len(samples)} samples")

    # Run evaluation
    results = evaluate_entity_level(
        samples=samples,
        extractor=extractor,
        decontextualizer=decontextualizer,
        nli_method=method,
        max_length=max_length,
        chunk_size=chunk_size,
    )

    # Print results
    print_header("Results")
    print_entity_metrics_summary(
        auroc_list=results['auroc_per_sample'],
        auprc_list=results['auprc_per_sample'],
        skipped=results['skipped']
    )

    # Save results if output directory specified
    if output_dir:
        import json
        from datetime import datetime

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_path / f"{dataset}_{method}_{timestamp}.json"

        # Prepare results for saving (remove non-serializable items)
        results_to_save = {
            'dataset': dataset,
            'method': method,
            'coref_enabled': coref,
            'max_length': max_length,
            'chunk_size': chunk_size,
            'n_samples': len(samples),
            'mean_auroc': float(results['mean_auroc']),
            'mean_auprc': float(results['mean_auprc']),
            'skipped': results['skipped'],
            'timestamp': timestamp,
        }

        with open(result_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        print(f"\nSaved results to {result_file}")
