"""
Sentence-level hallucination detection evaluation.

Supports: FELM, FactCheckBench datasets.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
from tqdm import tqdm

from evaluation.common import setup_logging, load_fact_extractor, load_decontextualizer, print_header
from evaluation.metrics import calculate_sentence_metrics, print_sentence_metrics_summary
from evaluation.dataset_loaders import load_felm_dataset, load_factcheckbench_dataset
from nli import check_nli_batch_fast, hallucination_prob_from_nli, is_context_valid


# =============================================================================
# CACHING UTILITIES
# =============================================================================

def get_cache_path(cache_dir: Path, dataset_name: str, cache_type: str, **kwargs) -> Path:
    """Get cache file path for a specific dataset and cache type.

    Args:
        cache_dir: Base cache directory
        dataset_name: Name of the dataset
        cache_type: Type of cache ('facts' or 'nli')
        **kwargs: Additional parameters for cache key (method, max_length, use_decontextualizer)

    Returns:
        Path to cache file
    """
    cache_dir = cache_dir / dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    if cache_type == 'facts':
        # Include decontextualizer in cache name
        use_decontext = kwargs.get('use_decontextualizer', False)
        suffix = "_decontext" if use_decontext else ""
        return cache_dir / f"facts_extracted{suffix}.json"
    elif cache_type == 'nli':
        method = kwargs.get('method', 'unknown')
        max_length = kwargs.get('max_length', 2048)
        return cache_dir / f"nli_scores_{method}_maxlen{max_length}.json"
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")


def extract_facts_cached(
    samples: List[Dict],
    extractor,
    decontextualizer,
    cache_path: Path,
    force_recompute: bool = False,
) -> List[Dict]:
    """Extract facts from samples with caching.

    Args:
        samples: List of samples
        extractor: FactExtractor instance
        decontextualizer: Optional decontextualizer
        cache_path: Path to cache file
        force_recompute: Force recomputation even if cache exists

    Returns:
        List of dicts with extracted facts for each sample
    """
    if cache_path.exists() and not force_recompute:
        print(f"Loading cached facts from {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f)

    print("Extracting facts...")
    results = []

    for sample in tqdm(samples, desc="Extracting facts"):
        text = sample['text']
        context = sample['context']
        label = sample['label']

        # Convert label: 1 = factual -> 0, 0 = hallucination -> 1
        gold = 1 - label if isinstance(label, int) else (0 if label else 1)

        result = {
            'text': sample['text'],
            'context': context,
            'gold': gold,
            'invalid_context': False,
            'no_facts': False,
            'facts': [],
            'decontextualized_text': text,
        }

        # Check context validity
        if not is_context_valid(context, min_words=10):
            result['invalid_context'] = True
            results.append(result)
            continue

        # Decontextualize if available
        if decontextualizer is not None:
            try:
                decontext_result = decontextualizer.decontextualize(text)
                text = decontext_result['resolved_text']
                result['decontextualized_text'] = text
            except Exception as e:
                print(f"Warning: Decontextualization failed: {e}")

        # Extract facts
        try:
            facts = extractor.extract_granular_facts(text)
            if not facts:
                result['no_facts'] = True
            else:
                # Take last fact from each incremental group and convert to strings
                facts_to_check = [f.facts[-1] for f in facts]
                result['facts'] = [str(f) for f in facts_to_check]
        except Exception as e:
            print(f"Warning: Fact extraction failed: {e}")
            result['no_facts'] = True

        results.append(result)

    # Save to cache
    print(f"Saving facts to cache: {cache_path}")
    with open(cache_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def score_facts_with_nli_cached(
    fact_samples: List[Dict],
    nli_method: str,
    max_length: int,
    chunk_size: int,
    cache_path: Path,
    force_recompute: bool = False,
) -> List[Dict]:
    """Score facts with NLI with caching.

    Args:
        fact_samples: List of samples with extracted facts
        nli_method: NLI method to use
        max_length: Max sequence length
        chunk_size: Batch size
        cache_path: Path to cache file
        force_recompute: Force recomputation even if cache exists

    Returns:
        List of dicts with NLI scores for each sample
    """
    if cache_path.exists() and not force_recompute:
        print(f"Loading cached NLI scores from {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f)

    print(f"Computing NLI scores with {nli_method}...")
    results = []

    for sample in tqdm(fact_samples, desc="Scoring with NLI"):
        result = {
            'text': sample['text'],
            'context': sample['context'],
            'gold': sample['gold'],
            'invalid_context': sample['invalid_context'],
            'no_facts': sample['no_facts'],
            'facts': sample.get('facts', []),
            'fact_scores': [],
        }

        # Skip if invalid context or no facts
        if sample['invalid_context'] or sample['no_facts']:
            results.append(result)
            continue

        if not sample['facts']:
            result['no_facts'] = True
            results.append(result)
            continue

        # Score each fact directly with NLI
        try:
            hypotheses = sample['facts']
            context = sample['context']

            # Split into chunks
            all_nli_scores = []
            for i in range(0, len(hypotheses), chunk_size):
                batch = hypotheses[i:i+chunk_size]
                batch_scores = check_nli_batch_fast(
                    premise=context,
                    hypotheses=batch,
                    max_length=max_length,
                    method=nli_method
                )
                all_nli_scores.extend(batch_scores)

            # Convert to serializable format
            serializable_scores = []
            for score_dict in all_nli_scores:
                serializable_scores.append({
                    'entailment': float(score_dict.get('entailment', 0)),
                    'neutral': float(score_dict.get('neutral', 0)),
                    'contradiction': float(score_dict.get('contradiction', 0)),
                    'span_kind': 'argument',
                })

            result['fact_scores'] = serializable_scores

        except Exception as e:
            print(f"Warning: NLI scoring failed: {e}")
            import traceback
            traceback.print_exc()
            result['no_facts'] = True

        results.append(result)

    # Save to cache
    print(f"Saving NLI scores to cache: {cache_path}")
    with open(cache_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def compute_metrics_for_combination(
    nli_samples: List[Dict],
    hal_prob_mode: str,
    aggregation: str,
    invalid_context_prob: float = 0.5,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute metrics for a specific combination of hal_prob_mode and aggregation.

    Args:
        nli_samples: Samples with NLI scores
        hal_prob_mode: Hallucination probability mode
        aggregation: Aggregation method
        invalid_context_prob: Probability for invalid context
        threshold: Classification threshold

    Returns:
        Dict with metrics
    """
    preds = []
    golds = []

    for sample in nli_samples:
        golds.append(sample['gold'])

        # Handle invalid context
        if sample['invalid_context']:
            preds.append(invalid_context_prob)
            continue

        # Handle no facts
        if sample['no_facts']:
            preds.append(0.0)
            continue

        # Compute hallucination probabilities
        arg_scores = []
        for fs in sample['fact_scores']:
            if fs.get('span_kind') == 'argument':
                hal_prob = hallucination_prob_from_nli(fs, mode=hal_prob_mode)
                arg_scores.append(hal_prob)

        # Aggregate scores
        if not arg_scores:
            sentence_hal_prob = 0.0
        elif aggregation == "max":
            sentence_hal_prob = max(arg_scores)
        elif aggregation == "min":
            sentence_hal_prob = min(arg_scores)
        elif aggregation == "median":
            sentence_hal_prob = float(np.median(arg_scores))
        else:  # mean
            sentence_hal_prob = float(np.mean(arg_scores))

        preds.append(sentence_hal_prob)

    # Compute metrics
    metrics = calculate_sentence_metrics(golds, preds, threshold)
    metrics['hal_prob_mode'] = hal_prob_mode
    metrics['aggregation'] = aggregation

    return metrics


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def evaluate_sentence_level(
    samples: List[Dict],
    extractor,
    nli_method: str,
    max_length: int,
    chunk_size: int,
    decontextualizer: Optional,
    hal_prob_mode: str,
    invalid_context_prob: float,
    aggregation: str,
    cache_dir: Path,
    dataset_name: str,
    force_recompute: bool,
    test_all_combinations: bool,
) -> Dict[str, Any]:
    """
    Evaluate at sentence level (FELM, Factcheck-Bench) with caching.

    Args:
        samples: List of sample dicts with 'text', 'context', 'label'
        extractor: FactExtractor instance
        nli_method: NLI method to use
        max_length: Max sequence length for NLI
        chunk_size: Batch size for NLI
        decontextualizer: Optional decontextualizer
        hal_prob_mode: Mode for hallucination probability calculation
        invalid_context_prob: Probability to assign when context is invalid
        aggregation: How to aggregate fact scores
        cache_dir: Directory for cache files
        dataset_name: Name of dataset for cache
        force_recompute: Force recomputation of cache
        test_all_combinations: If True, test all hal_prob_mode and aggregation combinations

    Returns dict with predictions, labels, and metrics.
    """
    # Step 1: Extract facts (with caching)
    facts_cache_path = get_cache_path(
        cache_dir,
        dataset_name,
        'facts',
        use_decontextualizer=(decontextualizer is not None)
    )

    if decontextualizer is not None:
        print(f"Using decontextualized facts cache: {facts_cache_path.name}")

    fact_samples = extract_facts_cached(
        samples=samples,
        extractor=extractor,
        decontextualizer=decontextualizer,
        cache_path=facts_cache_path,
        force_recompute=force_recompute,
    )

    # Step 2: Score with NLI (with caching)
    nli_cache_path = get_cache_path(cache_dir, dataset_name, 'nli', method=nli_method, max_length=max_length)
    nli_samples = score_facts_with_nli_cached(
        fact_samples=fact_samples,
        nli_method=nli_method,
        max_length=max_length,
        chunk_size=chunk_size,
        cache_path=nli_cache_path,
        force_recompute=force_recompute,
    )

    # Count invalid contexts
    invalid_context_count = sum(1 for s in nli_samples if s['invalid_context'])
    print(f"Invalid context samples: {invalid_context_count}/{len(samples)}")

    # Step 3: Test all combinations or single combination
    if test_all_combinations:
        print("\n" + "="*70)
        print("Testing all combinations of hal_prob_mode and aggregation")
        print("="*70)

        # Define all combinations to test
        hal_prob_modes = [
            ("contradiction_only", "contradiction"),
            ("neutral_only", "neutral"),
            ("default", "contradiction+neutral"),
        ]
        aggregations = ["mean", "max", "min"]

        results_table = []
        for mode_key, mode_display in hal_prob_modes:
            for agg in aggregations:
                metrics = compute_metrics_for_combination(
                    nli_samples=nli_samples,
                    hal_prob_mode=mode_key,
                    aggregation=agg,
                    invalid_context_prob=invalid_context_prob,
                    threshold=0.5,
                )
                # Replace mode key with display name for output
                metrics['hal_prob_mode'] = mode_display
                metrics['hal_prob_mode_key'] = mode_key
                results_table.append(metrics)

        # Sort by F1 macro (descending)
        results_table.sort(key=lambda x: x['f1_macro'], reverse=True)

        # Print results table
        print(f"\n{'Hal Prob Mode':<22} {'Aggregation':<12} {'AUROC':<10} {'F1 Macro':<10}")
        print("-" * 56)
        for res in results_table:
            print(f"{res['hal_prob_mode']:<22} {res['aggregation']:<12} {res['auroc']:<10.4f} {res['f1_macro']:<10.4f}")

        # Return best result
        best_result = results_table[0]
        print(f"\nBest combination: {best_result['hal_prob_mode']} + {best_result['aggregation']}")
        print(f"  AUROC: {best_result['auroc']:.4f}")
        print(f"  F1 Macro: {best_result['f1_macro']:.4f}")

        return {
            'all_combinations': results_table,
            'best_result': best_result,
            'invalid_context_count': invalid_context_count,
            'nli_samples': nli_samples,
        }
    else:
        # Single combination
        metrics = compute_metrics_for_combination(
            nli_samples=nli_samples,
            hal_prob_mode=hal_prob_mode,
            aggregation=aggregation,
            invalid_context_prob=invalid_context_prob,
            threshold=0.5,
        )

        return {
            'metrics': metrics,
            'invalid_context_count': invalid_context_count,
            'nli_samples': nli_samples,
        }


def run_sentence_evaluation(
    dataset: str,
    method: str,
    data_dir: str,
    output_dir: str,
    cache_dir: str,
    subset: str,
    max_length: int,
    chunk_size: int,
    coref: bool,
    force_recompute: bool,
    all_datasets: bool,
):
    """Run sentence-level evaluation."""
    setup_logging()

    # Determine datasets to evaluate
    if all_datasets:
        datasets_to_run = [
            ("felm", "wk"),
            ("felm", "science"),
            ("felm", "writing_rec"),
            ("factcheckbench", None),
        ]
    elif dataset == "felm":
        if subset:
            datasets_to_run = [(dataset, subset)]
        else:
            # Run all FELM subsets
            datasets_to_run = [
                ("felm", "wk"),
                ("felm", "science"),
                ("felm", "writing_rec"),
            ]
    else:
        datasets_to_run = [(dataset, None)]

    print_header("Sentence-Level Hallucination Detection Evaluation")
    print(f"NLI method: {method}")
    print(f"Coreference resolution: {'ENABLED' if coref else 'DISABLED'}")
    print(f"Will evaluate {len(datasets_to_run)} dataset(s)")
    print("=" * 70)

    # Load models
    extractor = load_fact_extractor()
    decontextualizer = load_decontextualizer(coref)

    # Setup output directory
    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    # Run evaluation on each dataset
    for ds_name, ds_subset in datasets_to_run:
        dataset_key = f"{ds_name}_{ds_subset}" if ds_subset else ds_name
        print_header(f"Evaluating on {dataset_key}")

        # Load dataset
        try:
            if ds_name == "felm":
                samples, eval_type = load_felm_dataset(ds_subset, data_dir)
            elif ds_name == "factcheckbench":
                samples, eval_type = load_factcheckbench_dataset(data_dir)
            else:
                raise ValueError(f"Unknown dataset: {ds_name}")

            print(f"Loaded {len(samples)} samples")
        except Exception as e:
            print(f"Error loading {dataset_key}: {e}")
            continue

        # Run evaluation (test all combinations by default)
        results = evaluate_sentence_level(
            samples=samples,
            extractor=extractor,
            nli_method=method,
            max_length=max_length,
            chunk_size=chunk_size,
            decontextualizer=decontextualizer,
            hal_prob_mode="default",  # Not used when test_all_combinations=True
            invalid_context_prob=0.5,
            aggregation="mean",  # Not used when test_all_combinations=True
            cache_dir=Path(cache_dir),
            dataset_name=dataset_key,
            force_recompute=force_recompute,
            test_all_combinations=True,
        )

        # Print results
        print_header("Results")
        best = results['best_result']
        print(f"Best combination: {best['hal_prob_mode']} + {best['aggregation']}")
        print_sentence_metrics_summary(best)

        # Save results if output directory specified
        if output_path:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = output_path / f"{dataset_key}_{method}_{timestamp}.json"

            # Prepare results for saving
            results_to_save = {
                'dataset': dataset_key,
                'method': method,
                'coref_enabled': coref,
                'max_length': max_length,
                'chunk_size': chunk_size,
                'n_samples': len(samples),
                'all_combinations': results['all_combinations'],
                'best_result': results['best_result'],
                'invalid_context_count': results['invalid_context_count'],
                'timestamp': timestamp,
            }

            with open(result_file, 'w') as f:
                json.dump(results_to_save, f, indent=2, default=str)
            print(f"\nSaved results to {result_file}")
