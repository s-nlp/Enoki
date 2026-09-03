"""
Entity-level hallucination detection evaluation.

Supports: HalluEntity dataset.
"""

import contextlib
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from functools import partial
from concurrent.futures import ThreadPoolExecutor

_null_ctx = contextlib.nullcontext()

import numpy as np
from tqdm import tqdm

from evaluation.common import setup_logging, load_fact_extractor, load_decontextualizer, print_header
from evaluation.metrics import calculate_entity_metrics, print_entity_metrics_summary
from evaluation.dataset_loaders import load_halluentity_dataset
from nli import check_nli_batch_fast, score_facts_with_nli
from evaluation.fact_alignment import (
    align_fact_scores_to_entities_orig,
    dedupe_fact_scores_norm,
    normalize_fact_spans_to_orig,
)


def _extract_for_sample(sample, extractor, decontextualizer):
    """Decontextualize + extract facts for one sample — runs in a thread for parallel extraction.

    Acquires extractor._extraction_lock when present for extractors that are not
    safe for concurrent calls.
    """
    if 'resolved_text' in sample and 'replacements' in sample:
        resolved_text = sample['resolved_text']
        replacements = sample['replacements']
    elif decontextualizer is not None:
        result = decontextualizer.decontextualize(sample['text'])
        resolved_text = result['resolved_text']
        replacements = result['replacements']
    else:
        resolved_text = sample['text']
        replacements = []
    lock = getattr(extractor, '_extraction_lock', None)
    try:
        with lock if lock else _null_ctx:
            facts = extractor.extract_granular_facts(resolved_text)
    except Exception as e:
        print(f"Warning: Fact extraction failed: {e}")
        facts = []
    return resolved_text, replacements, facts


def evaluate_entity_level(
    samples: list[dict],
    extractor,
    decontextualizer: Optional,
    nli_method: str,
    max_length: int,
    chunk_size: int,
    chunk_overlap: int = 1,
    extraction_workers: int = 1,
) -> Dict[str, Any]:
    """
    Evaluate at entity level (HalluEntity).

    Args:
        samples: List of samples with 'text', 'context', 'entities', 'entity_labels'
        extractor: Fact extractor instance
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
    per_sample_preds = []
    fact_details = []
    skipped = 0

    print(f"Starting entity-level evaluation on {len(samples)} samples...")
    print(f"  NLI method: {nli_method}")
    print(f"  Decontextualization: {'enabled' if decontextualizer is not None else 'disabled'}")
    print(f"  Max length: {max_length}, Chunk size: {chunk_size}")
    print()

    # Pre-extract all facts in parallel when multiple workers are requested.
    # Each worker issues independent Graphene HTTP requests; NLI stays sequential on GPU.
    if extraction_workers > 1:
        print(f"Pre-extracting facts with {extraction_workers} threads...", flush=True)
        pool = ThreadPoolExecutor(max_workers=extraction_workers)
        futures = [
            pool.submit(_extract_for_sample, s, extractor, decontextualizer)
            for s in samples
        ]
        try:
            extracted_results = [
                f.result()
                for f in tqdm(futures, desc="Extracting", unit="sample", ncols=100, file=sys.stderr)
            ]
        except KeyboardInterrupt:
            for f in futures:
                f.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            pool.shutdown(wait=False)
    else:
        extracted_results = None

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

        pronoun_map = []
        if extracted_results is not None:
            resolved_text, replacements, facts = extracted_results[idx]
            # Hypothesis-level substitution must include possessives that the
            # rewrite path skipped; they're encoded as hyp_only=True entries.
            pronoun_map = list(replacements)
            replacements = [r for r in replacements if not r.get('hyp_only')]
        else:
            # Decontextualize if needed
            if 'resolved_text' in sample and 'replacements' in sample:
                resolved_text = sample['resolved_text']
                replacements = sample['replacements']
                pronoun_map = list(replacements)
                replacements = [r for r in replacements if not r.get('hyp_only')]
            elif decontextualizer is not None:
                decontext_result = decontextualizer.decontextualize(orig_text)
                # OIE runs on the *resolved* text so pronoun-subject sentences
                # produce well-formed triples (e.g. "Lanny Flaherty has appeared
                # in ..." instead of "He has appeared in ..."). Span coordinates
                # are mapped back to orig_text via normalize_fact_spans_to_orig
                # below, using `replacements`. Possessive (hyp_only) entries
                # are excluded from the text rewrite — they substitute only
                # inside the NLI hypothesis via pronoun_map.
                resolved_text = decontext_result['resolved_text']
                pronoun_map = decontext_result['replacements']
                replacements = [r for r in decontext_result['replacements']
                                if not r.get('hyp_only')]
            else:
                resolved_text = orig_text
                pronoun_map = []
                replacements = []

            # Extract facts from resolved text so OIE sees the canonical
            # name in subject position; spans are normalized back to orig_text
            # coordinates below for entity alignment.
            facts = extractor.extract_granular_facts(resolved_text)

        if not facts:
            auroc_list.append(0.0)
            auprc_list.append(0.0)
            per_sample_preds.append({
                "gold_labels": [int(l) for l in entity_labels],
                "entity_scores": [0.0] * len(entity_labels),
            })
            fact_details.append({'text': orig_text, 'facts': [], 'scores': []})
            continue

        # Score with NLI; pronoun_map substitutes pronoun subjects in hypotheses
        # only (OIE already ran on orig_text, so spans are unchanged)
        fact_scores = score_facts_with_nli(
            context=context,
            granular_facts=facts,
            check_nli_batch_fn=partial(
                check_nli_batch_fast,
                max_length=max_length,
                method=nli_method,
                premise_chunk_overlap_sents=chunk_overlap,
            ),
            chunk_size=chunk_size,
            incremental_stop_threshold=0.5,
            pronoun_map=pronoun_map or None,
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
            per_sample_preds.append({
                "gold_labels": y_true,
                "entity_scores": y_score,
                "facts": [
                    {
                        "fact": fs.get("fact", ""),
                        "orig_span_start": int(fs["orig_span_start"]),
                        "orig_span_end": int(fs["orig_span_end"]),
                        "hall_prob": float(fs.get("hall_prob", 0.0)),
                        "entailment": float(fs.get("entailment", 0.0)),
                        "neutral": float(fs.get("neutral", 0.0)),
                        "contradiction": float(fs.get("contradiction", 0.0)),
                        "span_kind": fs.get("span_kind", ""),
                    }
                    for fs in fact_scores_norm
                    if fs.get("span_kind") != "predicate"
                ],
            })

        except Exception as e:
            skipped += 1
            auroc_list.append(0.0)
            auprc_list.append(0.0)
            per_sample_preds.append({
                "gold_labels": [int(l) for l in entity_labels],
                "entity_scores": [0.0] * len(entity_labels),
            })
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
        'per_sample_preds': per_sample_preds,
        'skipped': skipped,
        'fact_details': fact_details,
    }


def run_entity_evaluation(
    dataset: str,
    method: str,
    extractor_method: str,
    data_dir: str,
    output_dir: str,
    cache_dir: str,
    max_length: int,
    chunk_size: int,
    coref: bool,
    force_recompute: bool,
    chunk_overlap: int = 1,
    incremental: bool = True,
    use_preprocessing: bool = True,
    extraction_workers: int = 1,
    checkpoint: Optional[str] = None,
    llm_model: Optional[str] = None,
):
    """Run entity-level evaluation."""
    setup_logging()

    print_header("Entity-Level Hallucination Detection Evaluation")
    print(f"Dataset: {dataset}")
    print(f"NLI method: {method}")
    print(f"Coreference resolution: {'ENABLED' if coref else 'DISABLED'}")
    print(f"Chunk overlap: {chunk_overlap} sentence(s)")
    print("=" * 70)

    import json
    from datetime import datetime
    from evaluation.predictions_io import load_raw_predictions, save_raw_predictions

    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    coref_suffix = "_coref" if coref else ""
    ckpt_suffix = f"_{Path(checkpoint).stem}" if checkpoint else ""
    preds_file = output_path / f"{dataset}_{method}{ckpt_suffix}{coref_suffix}_overlap{chunk_overlap}_extractor_{extractor_method}_preds.json" if output_path else None

    if preds_file and preds_file.exists() and not force_recompute:
        # Fast path: load cached raw predictions, recompute aggregate metrics
        print(f"Loading cached raw predictions from {preds_file}")
        raw_data = load_raw_predictions(preds_file)
        per_sample_preds = raw_data["samples"]

        all_labels = [g for s in per_sample_preds for g in s["gold_labels"]]
        all_preds = [sc for s in per_sample_preds for sc in s["entity_scores"]]

        from evaluation.metrics import calculate_entity_metrics
        auroc_list = []
        auprc_list = []
        for s in per_sample_preds:
            auroc, auprc = calculate_entity_metrics(s["gold_labels"], s["entity_scores"])
            auroc_list.append(auroc)
            auprc_list.append(auprc)

        results = {
            'auroc_per_sample': auroc_list,
            'auprc_per_sample': auprc_list,
            'mean_auroc': float(np.mean(auroc_list)),
            'mean_auprc': float(np.mean(auprc_list)),
            'per_sample_preds': per_sample_preds,
            'skipped': 0,
        }
        n_samples = len(per_sample_preds)
    else:
        # Load models and run inference
        extractor = load_fact_extractor(extractor_method=extractor_method, incremental=incremental, use_preprocessing=use_preprocessing, checkpoint=checkpoint, llm_model=llm_model)
        decontextualizer = load_decontextualizer(coref)

        # Load dataset and run inference
        print(f"\nLoading {dataset} dataset...")
        samples, eval_type = load_halluentity_dataset(data_dir)
        print(f"Loaded {len(samples)} samples")
        n_samples = len(samples)

        results = evaluate_entity_level(
            samples=samples,
            extractor=extractor,
            decontextualizer=decontextualizer,
            nli_method=method,
            max_length=max_length,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            extraction_workers=extraction_workers,
        )

        # Save raw predictions for future re-use
        if preds_file:
            save_raw_predictions(preds_file, {
                "type": "entity",
                "dataset": dataset,
                "method": method,
                "coref_enabled": coref,
                "samples": results['per_sample_preds'],
            })

    # Print results
    print_header("Results")
    print_entity_metrics_summary(
        auroc_list=results['auroc_per_sample'],
        auprc_list=results['auprc_per_sample'],
        skipped=results['skipped']
    )

    # Save summary results
    if output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_path / f"{dataset}_{method}{ckpt_suffix}_{timestamp}.json"

        results_to_save = {
            'dataset': dataset,
            'method': method,
            'coref_enabled': coref,
            'max_length': max_length,
            'chunk_size': chunk_size,
            'n_samples': n_samples,
            'mean_auroc': float(results['mean_auroc']),
            'mean_auprc': float(results['mean_auprc']),
            'skipped': results['skipped'],
            'timestamp': timestamp,
        }

        with open(result_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        print(f"\nSaved results to {result_file}")
