"""Sentence-level hallucination detection evaluation."""

import contextlib
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor

_null_ctx = contextlib.nullcontext()

import numpy as np
from tqdm import tqdm

from evaluation.common import setup_logging, load_fact_extractor, load_decontextualizer, print_header
from evaluation.metrics import calculate_sentence_metrics, print_sentence_metrics_summary
from evaluation.dataset_loaders import load_factcheckbench_dataset, load_sampled_jsonl
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

    extractor_method = kwargs.get('extractor', 'unknown')

    if cache_type == 'facts':
        # Include decontextualizer in cache name
        use_decontext = kwargs.get('use_decontextualizer', False)
        suffix = "_decontext" if use_decontext else ""
        return cache_dir / f"extractor_method_{extractor_method}_facts_extracted{suffix}.json"
    elif cache_type == 'nli':
        method = kwargs.get('method', 'unknown')
        max_length = kwargs.get('max_length', 2048)
        chunk_overlap = kwargs.get('chunk_overlap', 0)
        use_decontext = kwargs.get('use_decontextualizer', False)
        suffix = "_decontext" if use_decontext else ""
        return cache_dir / f"extractor_method_{extractor_method}_nli_scores_{method}_maxlen{max_length}_overlap{chunk_overlap}{suffix}.json"
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")


def _select_largest_per_chain(pairs):
    """Keep only the largest (last) triplet per incremental chain.

    A chain is a run of consecutive triplets sharing the same (subject, predicate)
    where each successive span strictly contains the previous one.  When span
    containment breaks, the chain ends and the last triplet is kept.
    """
    if not pairs:
        return pairs
    result = []
    for i in range(len(pairs)):
        triplet, span = pairs[i]
        if i + 1 < len(pairs):
            next_triplet, next_span = pairs[i + 1]
            same_sp = (
                len(triplet) >= 2 and len(next_triplet) >= 2
                and triplet[0] == next_triplet[0]
                and triplet[1] == next_triplet[1]
            )
            span_grows = (
                span and next_span
                and next_span[0] <= span[0] and next_span[1] >= span[1]
                and next_span != span
            )
            if same_sp and span_grows:
                continue  # not the largest in this chain yet
        result.append(pairs[i])
    return result


def _extract_one_sentence_sample(sample, extractor, decontextualizer):
    """Extract facts for one sentence sample — runs in a thread for parallel extraction.

    Acquires extractor._extraction_lock when present (e.g. MinIE / pyjnius-based
    extractors that are not safe for concurrent JVM calls).
    """
    from fact_extractor.enoki_llm_extractor import PreExtractedFactExtractor

    text = sample['text']
    context = sample['context']
    label = sample['label']

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

    if not is_context_valid(context, min_words=10):
        result['invalid_context'] = True
        return result

    if isinstance(extractor, PreExtractedFactExtractor):
        sample_id = sample.get('id', '')
        pairs = extractor.get_facts_by_id(sample_id) if sample_id else None
        if not pairs:
            result['no_facts'] = True
        else:
            pairs = _select_largest_per_chain(pairs)
            result['facts'] = [
                " ".join(p.strip() for p in triplet if p and p.strip())
                for triplet, _ in pairs
            ]
        return result

    if decontextualizer is not None:
        try:
            answer = sample.get('answer')
            if answer:
                decontext_result = decontextualizer.decontextualize_in_document(answer, text)
            else:
                decontext_result = decontextualizer.decontextualize(text)
            text = decontext_result['resolved_text']
            result['decontextualized_text'] = text
        except Exception as e:
            print(f"Warning: Decontextualization failed: {e}")

    lock = getattr(extractor, '_extraction_lock', None)
    try:
        with lock if lock else _null_ctx:
            facts = extractor.extract_granular_facts(text)
        if not facts:
            result['no_facts'] = True
        else:
            facts_to_check = [f.facts[-1] for f in facts]
            result['facts'] = [str(f) for f in facts_to_check]
            result['fact_confs'] = [g.confidence for g in facts]
    except Exception as e:
        print(f"Warning: Fact extraction failed: {e}")
        result['no_facts'] = True

    return result


def extract_facts_cached(
    samples: List[Dict],
    extractor,
    decontextualizer,
    cache_path: Path,
    force_recompute: bool = False,
    extraction_workers: int = 1,
) -> List[Dict]:
    """Extract facts from samples with caching.

    Args:
        samples: List of samples
        extractor: FactExtractor instance
        decontextualizer: Optional decontextualizer
        cache_path: Path to cache file
        force_recompute: Force recomputation even if cache exists
        extraction_workers: Parallel threads for extraction (>1 pre-extracts in parallel)

    Returns:
        List of dicts with extracted facts for each sample
    """
    if cache_path.exists() and not force_recompute:
        print(f"Loading cached facts from {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f)

    print("Extracting facts...")

    if extraction_workers > 1:
        print(f"Pre-extracting facts with {extraction_workers} threads...", flush=True)
        pool = ThreadPoolExecutor(max_workers=extraction_workers)
        futures = [
            pool.submit(_extract_one_sentence_sample, s, extractor, decontextualizer)
            for s in samples
        ]
        try:
            results = [
                f.result()
                for f in tqdm(futures, desc="Extracting facts", unit="sample", ncols=100, file=sys.stderr)
            ]
        except KeyboardInterrupt:
            for f in futures:
                f.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            pool.shutdown(wait=False)
        print(f"Saving facts to cache: {cache_path}")
        with open(cache_path, 'w') as f:
            json.dump(results, f, indent=2)
        return results

    from fact_extractor.enoki_llm_extractor import PreExtractedFactExtractor

    results = []

    # Fast path: batch GPU inference via nlp.pipe() when the extractor supports it
    # and no decontextualization is needed.
    if decontextualizer is None and hasattr(extractor, "extract_granular_facts_batch"):
        texts_for_batch = []
        batch_indices = []  # which result slots need batch extraction

        for i, sample in enumerate(samples):
            context = sample['context']
            label = sample['label']
            gold = 1 - label if isinstance(label, int) else (0 if label else 1)
            result = {
                'text': sample['text'],
                'context': context,
                'gold': gold,
                'invalid_context': False,
                'no_facts': False,
                'facts': [],
                'decontextualized_text': sample['text'],
            }
            results.append(result)
            if not is_context_valid(context, min_words=10):
                result['invalid_context'] = True
            else:
                texts_for_batch.append(sample['text'])
                batch_indices.append(i)

        if texts_for_batch:
            print(f"Batch-extracting facts for {len(texts_for_batch)} samples...", flush=True)
            all_facts = extractor.extract_granular_facts_batch(texts_for_batch)
            for idx, facts in zip(batch_indices, all_facts):
                result = results[idx]
                if not facts:
                    result['no_facts'] = True
                else:
                    facts_to_check = [f.facts[-1] for f in facts]
                    result['facts'] = [str(f) for f in facts_to_check]
                    result['fact_confs'] = [g.confidence for g in facts]

        print(f"Saving facts to cache: {cache_path}")
        with open(cache_path, 'w') as f:
            json.dump(results, f, indent=2)
        return results

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

        if isinstance(extractor, PreExtractedFactExtractor):
            sample_id = sample.get('id', '')
            pairs = extractor.get_facts_by_id(sample_id) if sample_id else None
            if not pairs:
                result['no_facts'] = True
            else:
                result['facts'] = [
                    " ".join(p.strip() for p in triplet if p and p.strip())
                    for triplet, _ in pairs
                ]
            results.append(result)
            continue

        # Decontextualize if available
        if decontextualizer is not None:
            try:
                answer = sample.get('answer')
                if answer:
                    decontext_result = decontextualizer.decontextualize_in_document(answer, text)
                else:
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
                facts_to_check = [f.facts[-1] for f in facts]
                result['facts'] = [str(f) for f in facts_to_check]
                result['fact_confs'] = [g.confidence for g in facts]
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
    chunk_overlap: int = 0
) -> List[Dict]:
    """Score facts with NLI with caching.

    Batches (premise_chunk, hypothesis) pairs across samples so the GPU
    forward pass size equals *chunk_size* regardless of how many facts
    each individual sentence has.

    Args:
        fact_samples: List of samples with extracted facts
        nli_method: NLI method to use
        max_length: Max sequence length
        chunk_size: GPU batch size (pairs per forward pass, across samples)
        cache_path: Path to cache file
        force_recompute: Force recomputation even if cache exists

    Returns:
        List of dicts with NLI scores for each sample
    """
    if cache_path.exists() and not force_recompute:
        print(f"Loading cached NLI scores from {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f)

    from nli.utils import get_nli_checker
    from nli.base import BaseNLIChecker

    print(f"Computing NLI scores with {nli_method}...")
    checker: BaseNLIChecker = get_nli_checker(nli_method)

    # Build result skeletons
    results = []
    for sample in fact_samples:
        results.append({
            'text': sample['text'],
            'context': sample['context'],
            'gold': sample['gold'],
            'invalid_context': sample['invalid_context'],
            'no_facts': sample['no_facts'] or not sample.get('facts'),
            'facts': sample.get('facts', []),
            'fact_scores': [],
        })

    # --- Phase 1: pre-compute premise chunks for each scoreable sample ---
    # Entry: (sample_idx, fact_idx, chunk_idx) -> position in flat list
    # We build a flat list of (premise_chunk, hypothesis) pairs, then score
    # in batches of chunk_size, then fold scores back.

    # For each sample: [(chunk0, chunk1, ...), fact0, fact1, ...]
    # Scoring matrix per sample: score[fact_idx] = best over chunks

    # flat_pairs[k] = (premise_chunk_str, hypothesis_str)
    flat_pairs: List[Tuple[str, str]] = []
    # flat_meta[k] = (sample_idx, fact_idx, chunk_idx, n_chunks)
    flat_meta: List[Tuple[int, int, int, int]] = []

    # Cache chunks per (premise, max_length, overlap) — many ANAH sentences share
    # the same Wikipedia article context.
    _chunk_cache: Dict[Tuple[str, int, int], List[str]] = {}

    print("Pre-chunking premises...", flush=True)
    for s_idx, sample in enumerate(tqdm(fact_samples, desc="Chunking", ncols=100, file=sys.stderr)):
        if sample['invalid_context'] or sample['no_facts'] or not sample.get('facts'):
            continue
        context = sample['context']
        hypotheses = sample['facts']
        cache_key = (context, max_length, chunk_overlap)
        if cache_key in _chunk_cache:
            chunks = _chunk_cache[cache_key]
        else:
            try:
                chunks = checker.get_premise_chunks(
                    context,
                    hypotheses,
                    max_length=max_length,
                    overlap_sents=chunk_overlap,
                )
            except Exception:
                chunks = [context]
            _chunk_cache[cache_key] = chunks
        n_chunks = len(chunks)
        for f_idx, hyp in enumerate(hypotheses):
            for c_idx, chunk in enumerate(chunks):
                flat_pairs.append((chunk, hyp))
                flat_meta.append((s_idx, f_idx, c_idx, n_chunks))

    # --- Phase 2: batched forward passes across all samples ---
    flat_scores: List[Optional[Dict[str, float]]] = [None] * len(flat_pairs)

    for batch_start in tqdm(
        range(0, len(flat_pairs), chunk_size),
        desc=f"Scoring with NLI (batch={chunk_size})",
        unit="batch",
        ncols=100,
        file=sys.stderr,
    ):
        batch_end = min(batch_start + chunk_size, len(flat_pairs))
        b_premises = [flat_pairs[k][0] for k in range(batch_start, batch_end)]
        b_hyps = [flat_pairs[k][1] for k in range(batch_start, batch_end)]
        try:
            batch_result = checker.check_batch_flat(
                b_premises, b_hyps, max_length=max_length
            )
            for k, score in zip(range(batch_start, batch_end), batch_result):
                flat_scores[k] = score
        except Exception as e:
            print(f"Warning: NLI batch failed: {e}")
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    # --- Phase 3: fold flat scores back into per-sample fact_scores ---
    # For multi-chunk premises: pick chunk with minimum hall_prob per hypothesis
    from nli.utils import hallucination_prob_from_nli

    # best_score[s_idx][f_idx] = best NLI dict so far
    best_score: Dict[int, Dict[int, Dict]] = {}
    best_hall: Dict[int, Dict[int, float]] = {}

    for k, (s_idx, f_idx, c_idx, n_chunks) in enumerate(flat_meta):
        score = flat_scores[k]
        if score is None:
            continue
        hall = hallucination_prob_from_nli(score)
        if s_idx not in best_score or f_idx not in best_score[s_idx] or hall < best_hall[s_idx][f_idx]:
            best_score.setdefault(s_idx, {})[f_idx] = score
            best_hall.setdefault(s_idx, {})[f_idx] = hall

    for s_idx, fact_map in best_score.items():
        confs = fact_samples[s_idx].get('fact_confs') or []
        serializable = []
        for f_idx in sorted(fact_map):
            sc = fact_map[f_idx]
            entry = {
                'entailment': float(sc.get('entailment', 0)),
                'neutral': float(sc.get('neutral', 0)),
                'contradiction': float(sc.get('contradiction', 0)),
                'span_kind': 'argument',
            }
            if f_idx < len(confs):
                entry['triple_conf'] = float(confs[f_idx])
            serializable.append(entry)
        results[s_idx]['fact_scores'] = serializable

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
    chunk_overlap: int,
    extraction_workers: int = 1,
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
        use_decontextualizer=(decontextualizer is not None),
        extractor=extractor
    )

    if decontextualizer is not None:
        print(f"Using decontextualized facts cache: {facts_cache_path.name}")

    fact_samples = extract_facts_cached(
        samples=samples,
        extractor=extractor,
        decontextualizer=decontextualizer,
        cache_path=facts_cache_path,
        force_recompute=force_recompute,
        extraction_workers=extraction_workers,
    )

    # Step 2: Score with NLI (with caching)
    nli_cache_path = get_cache_path(
        cache_dir, dataset_name, 'nli',
        method=nli_method, max_length=max_length,
        chunk_overlap=chunk_overlap,
        use_decontextualizer=(decontextualizer is not None),
        extractor=extractor
    )
    nli_samples = score_facts_with_nli_cached(
        fact_samples=fact_samples,
        nli_method=nli_method,
        max_length=max_length,
        chunk_size=chunk_size,
        cache_path=nli_cache_path,
        force_recompute=force_recompute,
        chunk_overlap=chunk_overlap
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
            ("default",            "contradiction+neutral"),
            ("contradiction_only", "contradiction"),
            ("neutral_only",       "neutral"),
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

def build_samples(anah_samples: List[Dict]) -> List[Dict]:
    """Adapt ANAH records to the sentence-evaluator's expected shape.

    Target shape: {'text': str, 'context': str, 'label': int}
        where label=1 means FACTUAL (consistent with FELM/FCB convention,
        evaluator will invert to gold=hallucination=1 internally).
    """
    out = []
    for s in anah_samples:
        record = {
            "text": s["sentence"],
            "context": s["context"],
            "label": 1 - int(s["label"]),  # invert: our label=1 is hall, evaluator wants 1=factual
        }
        if s.get("id"):
            record["id"] = s["id"]
        if s.get("answer"):
            record["answer"] = s["answer"]
        out.append(record)
    return out

def run_sentence_evaluation(
    dataset: str,
    method: str,
    extractor_method: str,
    data_dir: str,
    output_dir: str,
    cache_dir: str,
    subset: str,
    max_length: int,
    chunk_size: int,
    coref: bool,
    force_recompute: bool,
    all_datasets: bool,
    incremental: bool = True,
    use_preprocessing: bool = False,
    chunk_overlap: int = 0,
    extraction_workers: int = 1,
    checkpoint: Optional[str] = None,
    pre_extracted_facts_file: Optional[str] = None,
    limit: Optional[int] = None,
    filter_by_pre_extracted: bool = False,
):
    """Run sentence-level evaluation."""
    setup_logging()

    # Determine datasets to evaluate
    if all_datasets:
        datasets_to_run = [
            ("factcheckbench", None),
            ("anah", None),
            ("ragtruth", None),
        ]
    else:
        datasets_to_run = [(dataset, None)]

    print_header("Sentence-Level Hallucination Detection Evaluation")
    print(f"NLI method: {method}")
    print(f"Coreference resolution: {'ENABLED' if coref else 'DISABLED'}")
    print(f"Will evaluate {len(datasets_to_run)} dataset(s)")
    print("=" * 70)

    # For pre_extracted_refchecker the extractor is dataset-specific and loaded
    # inside the loop below; for all other methods load once here.
    _shared_extractor = None
    if extractor_method != 'pre_extracted_refchecker':
        _shared_extractor = load_fact_extractor(
            extractor_method,
            incremental=incremental,
            use_preprocessing=use_preprocessing,
            checkpoint=checkpoint,
            pre_extracted_facts_file=pre_extracted_facts_file,
        )
    decontextualizer = load_decontextualizer(coref)

    # Setup output directory
    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    # Run evaluation on each dataset
    for ds_name, ds_subset in datasets_to_run:
        dataset_key = f"{ds_name}_{ds_subset}" if ds_subset else ds_name
        print_header(f"Evaluating on {dataset_key}")

        if extractor_method == 'pre_extracted_refchecker':
            extractor = load_fact_extractor(
                extractor_method,
                incremental=incremental,
                use_preprocessing=use_preprocessing,
                checkpoint=checkpoint,
                dataset=ds_name,
                pre_extracted_facts_file=pre_extracted_facts_file,
            )
        else:
            extractor = _shared_extractor

        # Load dataset
        try:
            if ds_name == "factcheckbench":
                samples, eval_type = load_factcheckbench_dataset(data_dir)
            elif ds_name == "anah":
                anah_sampled = Path(data_dir) / "anah_sampled" / "anah_250_sample.jsonl"
                samples = load_sampled_jsonl(str(anah_sampled))
            elif ds_name == "ragtruth":
                ragtruth_sampled = Path(data_dir) / "ragtruth_sampled" / "ragtruth_250_sample_test.jsonl"
                samples = load_sampled_jsonl(str(ragtruth_sampled))
            else:
                raise ValueError(f"Unknown dataset: {ds_name}")

            if filter_by_pre_extracted and hasattr(extractor, 'has_sample'):
                before = len(samples)
                samples = [s for s in samples if extractor.has_sample(s.get('id', ''))]
                print(f"Filtered to {len(samples)}/{before} samples present in pre-extracted file")
                if not samples:
                    print(f"WARNING: 0 samples matched — the pre-extracted file may cover a different split or dataset. Skipping {dataset_key}.")
                    continue
            if limit is not None:
                samples = samples[:limit]
            print(f"Loaded {len(samples)} samples")
        except Exception as e:
            print(f"Error loading {dataset_key}: {e}")
            continue

        # Run evaluation with fixed combination: contradiction+neutral, mean, threshold=0.5
        results = evaluate_sentence_level(
            samples=samples,
            extractor=extractor,
            nli_method=method,
            max_length=max_length,
            chunk_size=chunk_size,
            decontextualizer=decontextualizer,
            hal_prob_mode="default",
            invalid_context_prob=0.5,
            aggregation="mean",
            cache_dir=Path(cache_dir),
            dataset_name=dataset_key,
            force_recompute=force_recompute,
            test_all_combinations=False,
            chunk_overlap=chunk_overlap,
            extraction_workers=extraction_workers,
        )

        # Print results
        print_header("Results")
        print_sentence_metrics_summary(results['metrics'])

        # Save results if output directory specified
        if output_path:
            from datetime import datetime
            from evaluation.predictions_io import save_raw_predictions

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_suffix = f"_{Path(checkpoint).stem}" if checkpoint else ""
            result_file = output_path / f"{dataset_key}_{method}{ckpt_suffix}_{extractor_method}_{timestamp}.json"

            # Prepare results for saving
            results_to_save = {
                'dataset': dataset_key,
                'method': method,
                'extractor_method': extractor_method,
                'coref_enabled': coref,
                'max_length': max_length,
                'chunk_size': chunk_size,
                'n_samples': len(samples),
                'result': results['metrics'],
                'invalid_context_count': results['invalid_context_count'],
                'timestamp': timestamp,
            }

            with open(result_file, 'w') as f:
                json.dump(results_to_save, f, indent=2, default=str)
            print(f"\nSaved results to {result_file}")

            # Save raw predictions for threshold analysis (derived from NLI cache, no extra inference)
            coref_suffix = "_coref" if coref else ""
            preds_file = output_path / f"{dataset_key}_{method}{ckpt_suffix}{coref_suffix}_preds.json"
            nli_samples = results.get('nli_samples', [])
            if nli_samples and not preds_file.exists():
                save_raw_predictions(preds_file, {
                    "type": "sentence",
                    "dataset": dataset_key,
                    "method": method,
                    "coref_enabled": coref,
                    "samples": [
                        {
                            "gold": s["gold"],
                            "invalid_context": s["invalid_context"],
                            "no_facts": s["no_facts"],
                            "facts": s.get("facts", []),
                            "fact_nli_scores": s.get("fact_scores", []),
                        }
                        for s in nli_samples
                    ],
                })


# =============================================================================
# EFFICIENCY BENCHMARK
# =============================================================================

def _load_efficiency_samples(dataset: str, data_dir: str) -> tuple:
    """Return (samples, dataset_label) for the efficiency benchmark.

    Each sample must have 'text' (one sentence) and 'context' keys.
    For RAGTruth, multi-sentence responses are split into individual sentences
    so the reported latency is per-sentence (consistent with other datasets).
    """
    import nltk
    import json as _json
    from evaluation.dataset_loaders import load_factcheckbench_dataset

    if dataset == "factcheckbench":
        samples, _ = load_factcheckbench_dataset(data_dir)
        valid = [s for s in samples if is_context_valid(s["context"], min_words=10)]
        return valid, "FactCheckBench"

    if dataset == "anah":
        anah_file = Path(data_dir) / "anah_sampled" / "anah_250_sample (1).jsonl"
        valid = []
        with open(anah_file, encoding="utf-8") as fh:
            for line in fh:
                s = _json.loads(line)
                ctx = s.get("reference") or s.get("context") or ""
                if is_context_valid(ctx, min_words=10):
                    valid.append({"text": s["sentence"], "context": ctx})
        return valid, "ANAH (250-sample)"

    if dataset == "ragtruth":
        ragtruth_file = Path(data_dir) / "ragtruth_sampled" / "ragtruth_250_sample_test.jsonl"
        valid = []
        with open(ragtruth_file, encoding="utf-8") as fh:
            for line in fh:
                s = _json.loads(line)
                if s.get("task_type") != "QA":
                    continue
                ctx = s.get("context") or s.get("reference") or ""
                if is_context_valid(ctx, min_words=10):
                    valid.append({"text": s["sentence"], "context": ctx})
        return valid, "RAGTruth (QA, 84-sample)"

    raise ValueError(f"Unknown dataset for efficiency benchmark: {dataset!r}. "
                     "Choose from: factcheckbench, anah, ragtruth")


def run_efficiency_benchmark(
    extractor,
    nli_method: str,
    dataset: str = "factcheckbench",
    data_dir: str = "data",
    max_length: int = 2048,
    chunk_size: int = 16,
    chunk_overlap: int = 1,
    n_warmup: int = 5,
    limit: Optional[int] = None,
) -> None:
    """Run efficiency benchmark on FactCheckBench, ANAH, or RAGTruth.

    Measures per-sentence latency, FLOPs, and TFLOPs throughput for
    enoki_encoder (OIE extraction + NLI verification) and enoki_encoder_hal
    (joint extraction + hallucination scoring via hal head).

    Reported metrics match the efficiency table in the Enoki paper:
      Avg. Claims/Sent | Extract Time/Sent | Verify Time/Sent |
      Total Latency/Sent | Total FLOPs/Sent | TFLOPs Throughput
    """
    import nltk
    from time import perf_counter
    from fact_extractor.enoki_encoder_extractor import ModernOpenIEExtractor, UNUSED_TOKENS

    if not isinstance(extractor, ModernOpenIEExtractor):
        raise ValueError("Efficiency benchmark is only implemented for the enoki_encoder extractor.")

    # ---- Load dataset ----
    valid_samples, dataset_label = _load_efficiency_samples(dataset, data_dir)
    print(f"{dataset_label}: {len(valid_samples)} valid samples (warmup={n_warmup})")

    bench_samples = valid_samples[n_warmup:]
    if limit is not None:
        bench_samples = bench_samples[:limit]

    # ---- Load NLI checker ----
    from nli.utils import get_nli_checker
    nli_checker = get_nli_checker(nli_method)
    nli_checker._load_model()
    nli_params = sum(p.numel() for p in nli_checker.model.parameters())
    print(f"NLI model parameters: {nli_params:,}")

    extractor_params = sum(p.numel() for p in extractor.model.parameters())
    print(f"Extractor parameters: {extractor_params:,}")

    # ---- Token counting helpers ----

    def _count_extract_tokens(text: str) -> int:
        """Tokens for one IGL forward pass (sentence only, no context)."""
        words = nltk.word_tokenize(text) + UNUSED_TOKENS
        enc = extractor.tokenizer(
            [words], is_split_into_words=True,
            truncation=True, max_length=extractor.max_length,
        )
        return len(enc["input_ids"][0])

    def _count_hal_tokens(text: str, context: str) -> int:
        """Tokens for one hal forward pass ([CLS] ctx [SEP] sent [SEP])."""
        words = nltk.word_tokenize(text) + UNUSED_TOKENS
        sent_ids = extractor.tokenizer(
            words, is_split_into_words=True, add_special_tokens=False,
        )["input_ids"]
        ctx_ids = extractor.tokenizer(context, add_special_tokens=False)["input_ids"]
        trimmed_ctx = ctx_ids[: max(0, extractor.hal_max_length - 3 - len(sent_ids))]
        return min(1 + len(trimmed_ctx) + 1 + len(sent_ids) + 1, extractor.hal_max_length)

    def _count_nli_tokens(context: str, facts: List[str]) -> int:
        """Total tokens across all (premise_chunk, fact) pairs."""
        if not facts:
            return 0
        chunks = nli_checker.get_premise_chunks(
            context, facts, max_length=max_length, overlap_sents=chunk_overlap
        )
        total = 0
        for chunk in chunks:
            for fact in facts:
                enc = nli_checker.tokenizer(
                    chunk, fact,
                    add_special_tokens=True, truncation=False,
                    return_attention_mask=False,
                )
                total += len(enc["input_ids"])
        return total

    # ---- Warmup ----
    print(f"Warming up on {n_warmup} samples …", flush=True)
    for sample in valid_samples[:n_warmup]:
        try:
            text, context = sample["text"], sample["context"]
            groups = extractor.extract_granular_facts(text)
            facts = [str(g.facts[-1]) for g in groups] if groups else []
            if facts:
                nli_checker.check_batch(
                    context, facts,
                    max_length=max_length,
                    premise_chunk_overlap_sents=chunk_overlap,
                )
        except Exception as e:
            print(f"  warmup error: {e}")

    # ---- Benchmark ----
    n_claims_list: List[int] = []
    extract_times: List[float] = []
    verify_times: List[float] = []
    extract_tokens_list: List[int] = []
    verify_tokens_list: List[int] = []

    print(f"Benchmarking {len(bench_samples)} samples …", flush=True)
    for sample in tqdm(bench_samples, desc="Efficiency benchmark", ncols=100, file=sys.stderr):
        text, context = sample["text"], sample["context"]
        try:
            # --- Extraction phase ---
            t0 = perf_counter()
            groups = extractor.extract_granular_facts(text)
            extract_t = perf_counter() - t0

            facts = [str(g.facts[-1]) for g in groups] if groups else []
            ex_tokens = _count_extract_tokens(text)

            # --- Verification phase ---
            if facts:
                t0 = perf_counter()
                nli_checker.check_batch(
                    context, facts,
                    max_length=max_length,
                    premise_chunk_overlap_sents=chunk_overlap,
                )
                verify_t = perf_counter() - t0
                v_tokens = _count_nli_tokens(context, facts)
            else:
                verify_t = 0.0
                v_tokens = 0

            n_claims_list.append(len(facts))
            extract_times.append(extract_t)
            verify_times.append(verify_t)
            extract_tokens_list.append(ex_tokens)
            verify_tokens_list.append(v_tokens)

        except Exception as e:
            print(f"  sample error: {e}")

    if not n_claims_list:
        print("No samples processed — cannot compute metrics.")
        return

    import numpy as np

    n = len(n_claims_list)
    avg_claims = float(np.mean(n_claims_list))
    avg_extract_t = float(np.mean(extract_times))
    avg_verify_t = float(np.mean(verify_times))
    avg_total_t = float(np.mean([e + v for e, v in zip(extract_times, verify_times)]))

    # FLOPs ≈ 2 * P * T  (extraction + verification)
    avg_ex_tokens = float(np.mean(extract_tokens_list))
    avg_v_tokens = float(np.mean(verify_tokens_list))

    extract_flops = 2.0 * extractor_params * avg_ex_tokens
    verify_flops = 2.0 * nli_params * avg_v_tokens if not is_hal else 0.0
    avg_flops = extract_flops + verify_flops

    tflops_throughput = avg_flops / (avg_total_t * 1e12) if avg_total_t > 0 else 0.0

    method_name = str(extractor)
    model_name = getattr(extractor.model.hparams, "model_name", "ModernBERT-large")

    print("\n" + "=" * 80)
    print(f"EFFICIENCY BENCHMARK  —  {dataset_label}")
    print(f"Method: {method_name}   |   Model: {model_name}   |   N={n} sentences")
    print("=" * 80)
    print(f"  Avg. Claims / Sentence  :  {avg_claims:.2f}")
    print(f"  Extract Time / Sent     :  {avg_extract_t:.3f}s")
    print(f"  Verify  Time / Sent     :  {avg_verify_t:.3f}s")
    print(f"  Total Latency / Sent    :  {avg_total_t:.3f}s")
    print(f"  Total FLOPs / Sent      :  {avg_flops:.3e}")
    print(f"  TFLOPs Throughput       :  {tflops_throughput:.2f}")
    print("=" * 80)
    print()
    print("LaTeX table row:")
    print(
        f"  {method_name!r}  &  {model_name}  "
        f"&  {avg_claims:.2f}  "
        f"&  {avg_extract_t:.2f}s  "
        f"&  {avg_verify_t:.2f}s  "
        f"&  {avg_total_t:.2f}s  "
        f"&  ${avg_flops / 1e16:.2f} \\times 10^{{16}}$  "
        f"&  {tflops_throughput:.2f}  \\\\"
    )
