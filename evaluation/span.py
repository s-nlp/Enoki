"""
Span-level hallucination detection evaluation.

Supports: PsiloQA, Mushroom, RAGTruth, MUCH datasets.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
from functools import partial

import pandas as pd
from tqdm import tqdm

from evaluation.common import setup_logging, load_fact_extractor, load_decontextualizer, print_header
from evaluation.metrics import calculate_span_f1, print_span_metrics_summary
from evaluation.dataset_loaders import load_psiloqa_dataset, load_mushroom_dataset, load_ragtruth_dataset  # , load_much_dataset
from nli import check_nli_batch_fast, score_facts_with_nli, score_preextracted_with_nli
from evaluation.fact_alignment import dedupe_fact_scores_norm, normalize_fact_spans_to_orig


def _extract_for_row(row, extractor, decontextualizer, postfilter):
    """Extract facts for one row — runs in a thread for parallel extraction.

    Returns (decontext_result, granular_facts).
    OIE always runs on the original answer text so character spans are stable.
    Coref replacements are stored in decontext_result for downstream NLI use.
    """
    if decontextualizer is not None:
        decontext_result = decontextualizer.decontextualize(row["answer"])
        # Keep replacements for NLI hypothesis substitution, but override
        # resolved_text so that normalize_fact_spans_to_orig is a no-op.
        decontext_result = {
            "resolved_text": row["answer"],
            "replacements": [],
            "pronoun_map": decontext_result["replacements"],
        }
    else:
        decontext_result = {"resolved_text": row["answer"], "replacements": [], "pronoun_map": []}
    try:
        facts = extractor.extract_granular_facts(row["answer"])
    except Exception as e:
        print(f"Warning: Failed to extract facts: {e}")
        facts = []
    if postfilter and facts:
        from fact_extractor.postfilter import filter_fact_groups
        facts, _ = filter_fact_groups(facts)
    return decontext_result, facts


def evaluate_span_dataset(
    data: List[Dict],
    extractor,
    decontextualizer: Optional,
    nli_method: str,
    max_length: int,
    threshold: float,
    hall_prob_mode: str,
    chunk_overlap: int = 1,
    postfilter: bool = False,
    extraction_workers: int = 1,
) -> tuple[List[List[List[int]]], List[List[List[int]]], List[Dict]]:
    """
    Run fact extraction and NLI scoring on span-level dataset.

    extraction_workers > 1 pre-computes all facts in a thread pool before NLI
    scoring. Useful when the extractor is a remote service (Graphene) with
    multiple server instances — each thread hits a different instance.

    Returns:
        (golds, preds, raw_samples) where raw_samples contains gold spans and
        un-thresholded fact scores for later re-thresholding.
    """
    from concurrent.futures import ThreadPoolExecutor

    golds = []
    preds = []
    raw_samples = []

    # Pre-extract all facts in parallel when multiple workers are available.
    if extraction_workers > 1:
        print(f"Pre-extracting facts with {extraction_workers} threads...", flush=True)
        with ThreadPoolExecutor(max_workers=extraction_workers) as pool:
            futures = [
                pool.submit(_extract_for_row, row, extractor, decontextualizer, postfilter)
                for row in data
            ]
            extracted = [f.result() for f in tqdm(futures, desc="Extracting",
                                                   unit="sample", ncols=100, file=sys.stderr)]
    else:
        extracted = None  # will extract inline below

    progress_bar = tqdm(
        enumerate(data),
        total=len(data),
        desc=f"Evaluating with {nli_method}",
        unit="sample",
        ncols=100,
        disable=False,
        file=sys.stderr
    )
    for idx, row in progress_bar:
        if extracted is not None:
            decontext_result, granular_facts = extracted[idx]
        else:
            # Inline extraction (serial path).
            if decontextualizer is not None:
                decontext_result_raw = decontextualizer.decontextualize(row["answer"])
                decontext_result = {
                    "resolved_text": row["answer"],
                    "replacements": [],
                    "pronoun_map": decontext_result_raw["replacements"],
                }
            else:
                decontext_result = {"resolved_text": row["answer"], "replacements": [], "pronoun_map": []}

            try:
                granular_facts = extractor.extract_granular_facts(row["answer"])
            except Exception as e:
                print(f"Warning: Failed to extract facts: {e}")
                granular_facts = []

            if postfilter and granular_facts:
                from fact_extractor.postfilter import filter_fact_groups
                granular_facts, _ = filter_fact_groups(granular_facts)

        # Update progress with info
        progress_bar.set_postfix({
            'facts': len(granular_facts),
            'gold_spans': len(row["labels"]) if "labels" in row else 0
        })

        # Score facts with NLI; pronoun_map substitutes pronouns in hypotheses only
        if granular_facts:
            fact_scores = score_facts_with_nli(
                context=row["context"],
                granular_facts=granular_facts,
                check_nli_batch_fn=partial(
                    check_nli_batch_fast,
                    max_length=max_length,
                    method=nli_method,
                    premise_chunk_overlap_sents=chunk_overlap,
                ),
                chunk_size=32,
                hall_prob_mode=hall_prob_mode,
                pronoun_map=decontext_result.get("pronoun_map") or None,
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

        gold_spans = row["labels"]
        golds.append(gold_spans)
        preds.append(pred_spans)
        answer_text = row["answer"]
        raw_samples.append({
            "answer": answer_text,
            "context": row["context"],
            "question": row.get("question", ""),
            "gold_spans": gold_spans,
            "fact_spans": [
                {
                    "start": int(fs["orig_span_start"]),
                    "end": int(fs["orig_span_end"]),
                    "hall_prob": float(fs.get("hall_prob", 0.0)),
                    "entailment": float(fs.get("entailment", 0.0)),
                    "neutral": float(fs.get("neutral", 0.0)),
                    "contradiction": float(fs.get("contradiction", 0.0)),
                    "fact": fs.get("fact", ""),
                    "span_text": answer_text[int(fs["orig_span_start"]):int(fs["orig_span_end"])],
                    "group_info": list(fs["group_info"]) if fs.get("group_info") is not None else None,
                    "clause_type": fs.get("clause_type"),
                    **( {"triple_conf": float(fs["triple_conf"])} if "triple_conf" in fs else {} ),
                }
                for fs in fact_scores_norm
            ],
        })

    return golds, preds, raw_samples


def _hal_groups_to_fact_scores(granular_facts) -> List[Dict]:
    """Convert hal-scored IncrementalFactGroups to score_facts_with_nli output format.

    group.confidence is used directly as hall_prob (hal sigmoid, high = hallucinated).
    """
    from nli.utils import argument_core_span, predicate_core_span

    results = []
    for group_idx, group in enumerate(granular_facts):
        per_fact_confs = getattr(group, "per_fact_confidences", None)
        group_conf = float(getattr(group, "confidence", 0.0))

        for fact_idx, fact in enumerate(group.facts):
            hal_prob = (
                float(per_fact_confs[fact_idx])
                if per_fact_confs and fact_idx < len(per_fact_confs)
                else group_conf
            )
            ent = 1.0 - hal_prob
            hyp = str(fact)
            delta = (group.deltas[fact_idx]
                     if group.deltas and fact_idx < len(group.deltas) else None)
            arg_span = delta if delta is not None else getattr(fact, "argument", None)

            if arg_span is not None:
                arg_core = argument_core_span(arg_span) or arg_span
                s = int(getattr(arg_core, "start_char", -1))
                e = int(getattr(arg_core, "end_char", -1))
                if s >= 0 and e > s and arg_core.text.strip():
                    results.append({
                        "fact": hyp,
                        "span_kind": "argument",
                        "span_start": s,
                        "span_end": e,
                        "span_text": arg_core.text,
                        "source_text": arg_core.doc.text,
                        "fact_idx": len(results),
                        "group_info": (group_idx, fact_idx),
                        "clause_type": getattr(group, "clause_type", None),
                        "entailment": ent,
                        "neutral": 0.0,
                        "contradiction": hal_prob,
                        "hall_prob": hal_prob,
                    })

            pred = getattr(fact, "predicate", None)
            if pred is not None:
                pred_core = predicate_core_span(pred)
                if pred_core is not None and pred_core.text.strip():
                    s = int(getattr(pred_core, "start_char", -1))
                    e = int(getattr(pred_core, "end_char", -1))
                    if s >= 0 and e > s:
                        results.append({
                            "fact": hyp,
                            "span_kind": "predicate",
                            "span_start": s,
                            "span_end": e,
                            "span_text": pred_core.text,
                            "source_text": pred_core.doc.text,
                            "fact_idx": len(results),
                            "group_info": (group_idx, fact_idx),
                            "clause_type": getattr(group, "clause_type", None),
                            "entailment": ent,
                            "neutral": 0.0,
                            "contradiction": hal_prob,
                            "hall_prob": hal_prob,
                        })

    return results


def evaluate_span_dataset_hal(
    data: List[Dict],
    extractor,
    decontextualizer: Optional,
    threshold: float,
    postfilter: bool = False,
) -> tuple[List[List[List[int]]], List[List[List[int]]], List[Dict]]:
    """Like evaluate_span_dataset but uses the hal head instead of NLI.

    Calls extractor.extract_granular_facts(text, context=context) so the
    hal head scores each triple against the reference in one forward pass.
    """
    from tqdm import tqdm
    from evaluation.fact_alignment import dedupe_fact_scores_norm, normalize_fact_spans_to_orig

    golds, preds, raw_samples = [], [], []

    for row in tqdm(data, desc="Evaluating with hal head", unit="sample",
                    ncols=100, file=__import__("sys").stderr):
        if decontextualizer is not None:
            decontext_result = decontextualizer.decontextualize(row["answer"])
        else:
            decontext_result = {"resolved_text": row["answer"], "replacements": []}

        try:
            granular_facts = extractor.extract_granular_facts(
                decontext_result["resolved_text"],
                context=row["context"],
            )
        except Exception as exc:
            print(f"Warning: extraction failed: {exc}")
            granular_facts = []

        if postfilter and granular_facts:
            from fact_extractor.postfilter import filter_fact_groups
            granular_facts, _ = filter_fact_groups(granular_facts)

        if granular_facts:
            fact_scores = _hal_groups_to_fact_scores(granular_facts)
            fact_scores = [f for f in fact_scores if f.get("span_kind") != "predicate"]
            fact_scores_norm, _ = normalize_fact_spans_to_orig(
                orig_text=row["answer"],
                resolved_text=decontext_result["resolved_text"],
                replacements=decontext_result["replacements"],
                fact_scores=fact_scores,
            )
            fact_scores_norm = dedupe_fact_scores_norm(fact_scores_norm)
        else:
            fact_scores_norm = []

        answer_text = row["answer"]

        # Build the saved fact_spans list (uses "start"/"end" keys for threshold replay).
        saved_fact_spans = [
            {
                "start": int(fs["orig_span_start"]),
                "end": int(fs["orig_span_end"]),
                "hall_prob": float(fs.get("hall_prob", 0.0)),
                "entailment": float(fs.get("entailment", 0.0)),
                "neutral": 0.0,
                "contradiction": float(fs.get("contradiction", 0.0)),
                "fact": fs.get("fact", ""),
                "span_text": answer_text[int(fs["orig_span_start"]):int(fs["orig_span_end"])],
                "group_info": list(fs["group_info"]) if fs.get("group_info") else None,
                "clause_type": fs.get("clause_type"),
            }
            for fs in fact_scores_norm
        ]

        has_incremental = any(
            (fs.get("group_info") or [None, -1])[1] > 0
            for fs in saved_fact_spans
        )
        if has_incremental:
            from evaluation.predictions_io import _apply_incremental_group_threshold
            pred_spans = _apply_incremental_group_threshold(saved_fact_spans, threshold)
        else:
            pred_spans = [
                [fs["start"], fs["end"]]
                for fs in saved_fact_spans
                if fs.get("hall_prob", 0) > threshold
            ]

        gold_spans = row["labels"]
        golds.append(gold_spans)
        preds.append(pred_spans)
        raw_samples.append({
            "answer": answer_text,
            "context": row["context"],
            "question": row.get("question", ""),
            "gold_spans": gold_spans,
            "fact_spans": saved_fact_spans,
        })

    return golds, preds, raw_samples


def evaluate_span_dataset_preextracted(
    data: List[Dict],
    extractor,
    nli_method: str,
    max_length: int,
    threshold: float,
    hall_prob_mode: str,
    chunk_overlap: int = 1,
    incremental_preextracted: bool = False,
) -> tuple[List[List[List[int]]], List[List[List[int]]], List[Dict]]:
    """
    NLI scoring for pre-extracted facts on span-level datasets.

    Bypasses fact extraction and span normalization — triplets and spans
    come directly from the extractor's JSONL file.

    Returns:
        (golds, preds, raw_samples)
    """
    golds = []
    preds = []
    raw_samples = []

    progress_bar = tqdm(
        enumerate(data),
        total=len(data),
        desc=f"Evaluating with {nli_method} (pre-extracted)",
        unit="sample",
        ncols=100,
        file=sys.stderr,
    )

    for idx, row in progress_bar:
        sample_id = row.get("id", "")
        context = row["context"]
        answer = row["answer"]
        gold_spans = row["labels"]

        triplet_span_pairs = extractor.get_facts_by_id(sample_id) if sample_id else None

        if triplet_span_pairs:
            fact_scores = score_preextracted_with_nli(
                context=context,
                triplet_span_pairs=triplet_span_pairs,
                check_nli_batch_fn=partial(
                    check_nli_batch_fast,
                    max_length=max_length,
                    method=nli_method,
                    premise_chunk_overlap_sents=chunk_overlap,
                ),
                hall_prob_mode=hall_prob_mode,
                answer=answer,
                incremental=incremental_preextracted,
            )
            if not incremental_preextracted:
                fact_scores = dedupe_fact_scores_norm(fact_scores)
        else:
            fact_scores = []

        if incremental_preextracted and any(fs.get("group_info") is not None for fs in fact_scores):
            from evaluation.predictions_io import _apply_incremental_group_threshold
            pred_spans = _apply_incremental_group_threshold(
                [{"start": fs["orig_span_start"], "end": fs["orig_span_end"], **fs}
                 for fs in fact_scores],
                threshold,
                hall_prob_mode,
            )
        else:
            pred_spans = [
                [fs["orig_span_start"], fs["orig_span_end"]]
                for fs in fact_scores
                if fs.get("hall_prob", 0) > threshold
            ]

        golds.append(gold_spans)
        preds.append(pred_spans)
        raw_samples.append({
            "answer": answer,
            "context": context,
            "question": row.get("question", ""),
            "gold_spans": gold_spans,
            "fact_spans": [
                {
                    "start": int(fs["orig_span_start"]),
                    "end": int(fs["orig_span_end"]),
                    "hall_prob": float(fs.get("hall_prob", 0.0)),
                    "entailment": float(fs.get("entailment", 0.0)),
                    "neutral": float(fs.get("neutral", 0.0)),
                    "contradiction": float(fs.get("contradiction", 0.0)),
                    "fact": fs.get("fact", ""),
                    "span_text": answer[int(fs["orig_span_start"]):int(fs["orig_span_end"])],
                    "group_info": list(fs["group_info"]) if fs.get("group_info") is not None else None,
                    "clause_type": None,
                }
                for fs in fact_scores
            ],
        })

    return golds, preds, raw_samples


def save_facts_json(raw_samples: List[Dict], output_path: Path) -> None:
    """Write a JSON list of samples, each with facts and NLI scores, for analysis."""
    import json
    out = []
    for sample_idx, sample in enumerate(raw_samples):
        gold_spans = sample.get("gold_spans", [])
        facts = []
        for fs in sample.get("fact_spans", []):
            start, end = fs.get("start", 0), fs.get("end", 0)
            is_gold = any(s < end and start < e for s, e in gold_spans)
            group_info = fs.get("group_info")
            facts.append({
                "fact": fs.get("fact", ""),
                "span_text": fs.get("span_text", ""),
                "start": start,
                "end": end,
                "hall_prob": fs.get("hall_prob", 0.0),
                "entailment": fs.get("entailment", 0.0),
                "neutral": fs.get("neutral", 0.0),
                "contradiction": fs.get("contradiction", 0.0),
                "group_idx": group_info[0] if group_info is not None else None,
                "fact_in_group": group_info[1] if group_info is not None else None,
                "is_gold": is_gold,
            })
        out.append({
            "sample_idx": sample_idx,
            "question": sample.get("question", ""),
            "answer": sample.get("answer", ""),
            "context": sample.get("context", ""),
            "gold_spans": gold_spans,
            "facts": facts,
        })
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved facts JSON to {output_path}")


ALL_HAL_PROB_MODES = ("default",)


def _raw_samples_from_nli_predictions_file(
    nli_predictions_file: str,
    data_fn,
    ds_name: str,
    limit: Optional[int] = None,
) -> List[Dict]:
    """Build raw_samples from a pre-scored NLI predictions JSONL.

    The JSONL format (nli_results/raw_predictions/*.jsonl):
      {"id": "...", "n_triplets": N, "triplet_scores": [
          {"triplet": [...], "span": [start, end],
           "hall_prob": float, "entailment": float,
           "neutral": float, "contradiction": float}, ...]}

    Returns raw_samples in the standard span-sample schema so the rest of
    run_span_evaluation (threshold application, CSV output, metrics) works
    without any changes.
    """
    import json

    nli_preds: Dict[str, Dict] = {}
    with open(nli_predictions_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            nli_preds[str(p["id"])] = p

    data = data_fn()
    if limit is not None:
        data = data[:limit]

    missing = 0
    raw_samples: List[Dict] = []
    for row in data:
        sid = str(row.get("id", ""))
        pred = nli_preds.get(sid)
        if pred is None:
            missing += 1

        answer = row["answer"]
        gold_spans = row["labels"] if row["labels"] else []
        fact_spans = []
        if pred:
            for t in pred.get("triplet_scores", []):
                start, end = int(t["span"][0]), int(t["span"][1])
                fact_spans.append({
                    "start": start,
                    "end": end,
                    "hall_prob": float(t.get("hall_prob", 0.0)),
                    "entailment": float(t.get("entailment", 0.0)),
                    "neutral": float(t.get("neutral", 0.0)),
                    "contradiction": float(t.get("contradiction", 0.0)),
                    "fact": " ".join(t["triplet"]) if "triplet" in t else "",
                    "span_text": answer[start:end],
                    "group_info": None,
                    "clause_type": None,
                })

        raw_samples.append({
            "answer": answer,
            "context": row.get("context", ""),
            "question": row.get("question", ""),
            "gold_spans": gold_spans,
            "fact_spans": fact_spans,
        })

    if missing:
        print(
            f"  WARNING: {missing}/{len(data)} {ds_name} samples not found "
            f"in {nli_predictions_file}",
            file=sys.stderr,
        )
    print(f"  Loaded {len(raw_samples)} samples from pre-scored NLI file (no NLI inference)")
    return raw_samples


def _get_or_compute_raw_predictions(
    preds_file: Path,
    ds_name: str,
    data_fn,
    extractor,
    decontextualizer,
    method: str,
    max_length: int,
    threshold: float,
    chunk_overlap: int,
    coref: bool,
    postfilter: bool,
    extraction_workers: int,
    limit: Optional[int],
    force_recompute: bool,
    incremental_preextracted: bool = False,
    nli_predictions_file: Optional[str] = None,
) -> List[Dict]:
    """Return cached raw predictions or run inference and cache the result."""
    from evaluation.predictions_io import load_raw_predictions, save_raw_predictions

    if preds_file.exists() and not force_recompute:
        print(f"Loading cached raw predictions from {preds_file}")
        return load_raw_predictions(preds_file)["samples"]

    if nli_predictions_file is not None:
        raw_samples = _raw_samples_from_nli_predictions_file(
            nli_predictions_file=nli_predictions_file,
            data_fn=data_fn,
            ds_name=ds_name,
            limit=limit,
        )
        save_raw_predictions(preds_file, {
            "type": "span",
            "dataset": ds_name,
            "method": "nli_file",
            "nli_predictions_file": nli_predictions_file,
            "samples": raw_samples,
        })
        return raw_samples

    try:
        data = data_fn()
        if limit is not None:
            data = data[:limit]
        print(f"Loaded {len(data)} examples")
    except Exception as e:
        raise RuntimeError(f"Error loading {ds_name}: {e}") from e

    if hasattr(extractor, 'get_facts_by_id'):
        _, _, raw_samples = evaluate_span_dataset_preextracted(
            data=data,
            extractor=extractor,
            nli_method=method,
            max_length=max_length,
            threshold=threshold,
            hall_prob_mode="default",
            chunk_overlap=chunk_overlap,
            incremental_preextracted=incremental_preextracted,
        )
    else:
        _, _, raw_samples = evaluate_span_dataset(
            data=data,
            extractor=extractor,
            decontextualizer=decontextualizer,
            nli_method=method,
            max_length=max_length,
            threshold=threshold,
            hall_prob_mode="default",
            chunk_overlap=chunk_overlap,
            postfilter=postfilter,
            extraction_workers=extraction_workers,
        )

    save_raw_predictions(preds_file, {
        "type": "span",
        "dataset": ds_name,
        "method": method,
        "coref_enabled": coref,
        "chunk_overlap": chunk_overlap,
        "samples": raw_samples,
    })
    return raw_samples


def run_span_evaluation(
    dataset: Optional[str],
    method: str,
    extractor_method: str,
    data_dir: str,
    output_dir: str,
    max_length: int,
    threshold: float,
    hall_prob_mode: str,
    coref: bool,
    all_datasets: bool,
    force_recompute: bool = False,
    parse_table: bool = False,
    ragtruth_split: str = "test",
    chunk_overlap: int = 1,
    limit: Optional[int] = None,
    save_facts: bool = False,
    incremental: bool = True,
    all_modes: bool = True,
    use_preprocessing: bool = True,
    postfilter: bool = False,
    max_workers: int = 1,
    extraction_workers: int = 1,
    checkpoint: Optional[str] = None,
    pre_extracted_facts_file: Optional[str] = None,
    train_pre_extracted_facts_file: Optional[str] = None,
    calibrate: bool = False,
    incremental_preextracted: bool = False,
    nli_predictions_file: Optional[str] = None,
    nli_batch_size: int = 32,
):
    """Run span-level evaluation."""
    setup_logging()

    # Determine datasets to evaluate
    if all_datasets:
        datasets_to_run = ["psiloqa", "mushroom", "ragtruth"]  # , "much"]
    else:
        datasets_to_run = [dataset]

    print_header("Span-Level Hallucination Detection Evaluation")
    print(f"NLI method: {method}")
    print(f"Coreference resolution: {'ENABLED' if coref else 'DISABLED'}")
    print(f"Chunk overlap: {chunk_overlap} sentence(s)")
    print(f"Will evaluate {len(datasets_to_run)} dataset(s): {', '.join(datasets_to_run)}")
    print("=" * 70)

    # Models are loaded lazily — only if at least one dataset needs inference
    extractor = None
    decontextualizer = None

    import atexit
    def _close_extractor():
        if extractor is not None and hasattr(extractor, 'close'):
            extractor.close()
    atexit.register(_close_extractor)

    # Dataset loaders
    loaders = {
        "psiloqa": lambda: load_psiloqa_dataset(),
        "mushroom": lambda: load_mushroom_dataset(data_dir),
        "ragtruth": lambda: load_ragtruth_dataset(split=ragtruth_split),
        # "much": lambda: load_much_dataset(data_dir),
    }

    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    from evaluation.predictions_io import load_raw_predictions, save_raw_predictions

    # Run evaluation on each dataset
    for ds_name in datasets_to_run:
        print_header(f"Evaluating on {ds_name}")

        # Cache filename is mode-agnostic (NLI scores don't depend on hal_prob_mode)
        cache_method_name = method
        if checkpoint:
            cache_method_name += f"_{Path(checkpoint).stem}"
        if coref:
            cache_method_name += "_coref"
        cache_method_name += f"_overlap{chunk_overlap}"

        ds_key = ds_name
        if ds_name == "ragtruth" and ragtruth_split != "test":
            ds_key = f"{ds_name}_{ragtruth_split}"

        preds_file = output_path / f"{cache_method_name}_{ds_key}_{extractor_method}_preds.json"

        from nli.utils import hallucination_prob_from_nli

        # Ensure models are loaded when inference will be needed
        if not (preds_file.exists() and not force_recompute) and extractor is None:
            extractor = load_fact_extractor(extractor_method, incremental=incremental, use_preprocessing=use_preprocessing, max_workers=max_workers, dataset=ds_name, checkpoint=checkpoint, pre_extracted_facts_file=pre_extracted_facts_file)
            decontextualizer = load_decontextualizer(coref)

        try:
            raw_samples = _get_or_compute_raw_predictions(
                preds_file=preds_file,
                ds_name=ds_name,
                data_fn=loaders[ds_name],
                extractor=extractor,
                decontextualizer=decontextualizer,
                method=method,
                max_length=max_length,
                threshold=threshold,
                chunk_overlap=chunk_overlap,
                coref=coref,
                postfilter=postfilter,
                extraction_workers=extraction_workers,
                limit=limit,
                force_recompute=force_recompute,
                incremental_preextracted=incremental_preextracted,
            )
        except RuntimeError as e:
            print(f"Error loading {ds_name}: {e}")
            continue
        golds = [s["gold_spans"] for s in raw_samples]

        # Threshold calibration on train split
        effective_threshold = threshold
        if calibrate:
            from evaluation.predictions_io import (
                calibrate_span_threshold,
                save_calibrated_threshold,
                load_calibrated_threshold,
            )
            threshold_file = output_path / f"{cache_method_name}_{ds_name}_{extractor_method}_threshold.json"

            if threshold_file.exists() and not force_recompute:
                cal_data = load_calibrated_threshold(threshold_file)
                effective_threshold = cal_data["threshold"]
                train_iou = cal_data.get("train_iou")
                iou_label = (
                    f", train IoU={train_iou:.4f}" if train_iou is not None else ""
                )
                print(
                    f"Loaded calibrated threshold: {effective_threshold:.4f}"
                    f"  (train F1={cal_data.get('train_f1', float('nan')):.4f}"
                    f"{iou_label}, mode={cal_data.get('hall_prob_mode', '?')})"
                )
            elif ds_name == "mushroom":
                effective_threshold = 0.5
                print("MuSHROOM has no labelled training set — using threshold=0.5")
            else:
                # Determine train-split loader and cache key
                if ds_name == "ragtruth":
                    train_key = "ragtruth_train"
                    train_fn = lambda: load_ragtruth_dataset(split="train")
                else:  # psiloqa
                    train_key = "psiloqa_train"
                    train_fn = lambda: load_psiloqa_dataset(split="train")

                train_preds_file = output_path / f"{cache_method_name}_{train_key}_{extractor_method}_preds.json"

                # Ensure models are loaded for train inference
                cal_facts_file = train_pre_extracted_facts_file or pre_extracted_facts_file
                train_extractor = load_fact_extractor(extractor_method, incremental=incremental, use_preprocessing=use_preprocessing, max_workers=max_workers, dataset=ds_name, checkpoint=checkpoint, pre_extracted_facts_file=cal_facts_file)
                if extractor is None:
                    extractor = train_extractor
                    decontextualizer = load_decontextualizer(coref)

                print_header(f"Calibrating threshold on {train_key}")
                try:
                    train_raw_samples = _get_or_compute_raw_predictions(
                        preds_file=train_preds_file,
                        ds_name=train_key,
                        data_fn=train_fn,
                        extractor=train_extractor,
                        decontextualizer=decontextualizer,
                        method=method,
                        max_length=max_length,
                        threshold=threshold,
                        chunk_overlap=chunk_overlap,
                        coref=coref,
                        postfilter=postfilter,
                        extraction_workers=extraction_workers,
                        limit=limit,
                        force_recompute=force_recompute,
                        incremental_preextracted=incremental_preextracted,
                    )
                except RuntimeError as e:
                    print(f"Calibration failed for {ds_name}: {e} — using default threshold {threshold}")
                    train_raw_samples = None

                if train_raw_samples is not None:
                    effective_threshold, best_row = calibrate_span_threshold(
                        train_raw_samples, n_thresholds=200, hall_prob_mode=hall_prob_mode
                    )
                    print(
                        f"Calibrated threshold: {effective_threshold:.4f}"
                        f"  (train P={best_row['precision']:.4f}"
                        f" R={best_row['recall']:.4f}"
                        f" F1={best_row['f1']:.4f}"
                        f" IoU={best_row['iou']:.4f})"
                    )
                    save_calibrated_threshold(threshold_file, {
                        "threshold": effective_threshold,
                        "hall_prob_mode": hall_prob_mode,
                        "train_source": train_key,
                        "train_precision": best_row["precision"],
                        "train_recall": best_row["recall"],
                        "train_f1": best_row["f1"],
                        "train_iou": best_row["iou"],
                    })

        # Emit one CSV per mode
        modes_to_emit = ALL_HAL_PROB_MODES if all_modes else (hall_prob_mode,)
        for mode in modes_to_emit:
            output_method_name = cache_method_name
            if mode != "default":
                output_method_name = method
                if checkpoint:
                    output_method_name += f"_{Path(checkpoint).stem}"
                if coref:
                    output_method_name += "_coref"
                output_method_name += f"_{mode}_overlap{chunk_overlap}"

            # Use calibrated threshold for the mode that was calibrated; fixed threshold otherwise.
            t = effective_threshold if (calibrate and mode == hall_prob_mode) else threshold

            _has_multi_fact_groups = any(
                any(
                    (fs.get("group_info") or [None, -1])[1] > 0
                    for fs in s.get("fact_spans", [])
                )
                for s in raw_samples
            )
            _use_incremental = (incremental_preextracted or _has_multi_fact_groups) and any(
                any(fs.get("group_info") is not None for fs in s.get("fact_spans", []))
                for s in raw_samples
            )
            if _use_incremental:
                from evaluation.predictions_io import _apply_incremental_group_threshold
                mode_preds = [
                    _apply_incremental_group_threshold(s.get("fact_spans", []), t, mode)
                    for s in raw_samples
                ]
            else:
                mode_preds = [
                    [
                        [fs["start"], fs["end"]]
                        for fs in s.get("fact_spans", [])
                        if hallucination_prob_from_nli(fs, mode=mode) > t
                    ]
                    for s in raw_samples
                ]

            output_file = output_path / f"{output_method_name}_{ds_key}_{extractor_method}.csv"
            pd.DataFrame({'gold': golds, 'pred': mode_preds}).to_csv(output_file, index=False)
            print(f"Saved predictions to {output_file}")

            threshold_label = f"t={t:.4f}" + (" [calibrated]" if calibrate and mode == hall_prob_mode else "")
            metrics = calculate_span_f1(golds, mode_preds)
            print_header(f"Metrics for {ds_key} [{mode}] ({threshold_label})")
            print_span_metrics_summary(metrics)

        # Facts CSV (optional)
        if save_facts:
            facts_missing = raw_samples and "fact" not in (raw_samples[0].get("fact_spans") or [{}])[0]
            if facts_missing:
                print("Warning: cached preds lack fact text — rerun with --force-recompute to generate facts CSV")
            else:
                facts_file = output_path / f"{cache_method_name}_{ds_key}_{extractor_method}_facts.json"
                save_facts_json(raw_samples, facts_file)

        # Parse table (optional, psiloqa and ragtruth only)
        if parse_table and ds_name in ("psiloqa", "ragtruth"):
            from evaluation.parse_table import build_parse_table
            # Ensure models are loaded (may have used cache path above)
            if extractor is None:
                extractor = load_fact_extractor(extractor_method, incremental=incremental, dataset=ds_name, checkpoint=checkpoint, pre_extracted_facts_file=pre_extracted_facts_file)
                decontextualizer = load_decontextualizer(coref)
            # Load dataset if not already loaded (cache path doesn't load it)
            try:
                pt_data = loaders[ds_name]()
            except Exception as e:
                print(f"Warning: Could not load {ds_name} for parse table: {e}")
                pt_data = None
            if pt_data is not None:
                pt_file = output_path / f"{cache_method_name}_{ds_key}_{extractor_method}_parse_table.csv"
                build_parse_table(
                    data=pt_data,
                    extractor=extractor,
                    decontextualizer=decontextualizer,
                    coref=coref,
                    output_file=pt_file,
                )
