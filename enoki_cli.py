#!/usr/bin/env python3
"""
Enoki CLI - Unified interface for hallucination detection evaluation.

Usage:
    python enoki_cli.py evaluate sentence --dataset factcheckbench --method modernbert
    python enoki_cli.py evaluate entity --dataset halluentity --method modernbert
    python enoki_cli.py evaluate span --dataset psiloqa --method modernbert
"""

import typer
from typing import Optional
from pathlib import Path
from enum import Enum

app = typer.Typer(
    name="enoki",
    help="Enoki: Hallucination Detection Tool",
    add_completion=False,
)

evaluate_app = typer.Typer(help="Evaluation commands")
app.add_typer(evaluate_app, name="evaluate")

train_app = typer.Typer(help="Training commands")
app.add_typer(train_app, name="train")


class NLIMethod(str, Enum):
    modernbert = "modernbert"
    alignscore = "alignscore"
    qwen_06b = "qwen_06b"
    qwen_4b = "qwen_4b"
    qwen_8b = "qwen_8b"
    llm = "llm"


class ExtractorMethod(str, Enum):
    stanford = "stanford"
    minie = "minie"
    minie_safe = "minie_safe"
    minie_complete = "minie_complete"
    minie_aggressive = "minie_aggressive"
    minie_dictionary = "minie_dictionary"
    enoki_encoder = "enoki_encoder"
    cycleoie = "cycleoie"
    enoki_rules = "enoki_rules"


class SentenceDataset(str, Enum):
    factcheckbench = "factcheckbench"
    anah = "anah"
    ragtruth = "ragtruth"


class EntityDataset(str, Enum):
    halluentity = "halluentity"


class SpanDataset(str, Enum):
    psiloqa = "psiloqa"
    mushroom = "mushroom"
    ragtruth = "ragtruth"


class HallProbMode(str, Enum):
    default = "default"
    contradiction_only = "contradiction_only"
    neutral_only = "neutral_only"


@evaluate_app.command("sentence")
def evaluate_sentence(
    dataset: Optional[SentenceDataset] = typer.Option(None, help="Dataset to evaluate"),
    method: NLIMethod = typer.Option(NLIMethod.modernbert, help="NLI method"),
    extractor_method: ExtractorMethod = typer.Option(ExtractorMethod.stanford, help="Fact extractor method"),
    data_dir: Path = typer.Option("data", help="Directory containing dataset files"),
    output_dir: Path = typer.Option("eval_results", help="Output directory"),
    cache_dir: Path = typer.Option("cache", help="Cache directory"),
    max_length: int = typer.Option(2048, help="Max sequence length"),
    chunk_size: int = typer.Option(16, help="Batch size for NLI"),
    coref: bool = typer.Option(False, help="Enable coreference resolution"),
    force_recompute: bool = typer.Option(False, help="Force recomputation"),
    all_datasets: bool = typer.Option(False, "--all", help="Run on all datasets"),
    incremental: bool = typer.Option(True, help="Build incremental NP-modifier chains"),
    preprocessing: bool = typer.Option(False, help="Apply markdown masking + GLiNER to non-enoki extractors"),
    chunk_overlap: int = typer.Option(1, help="Sentence overlap between premise chunks (0=no overlap, 1=default, ...)"),
    extraction_workers: int = typer.Option(1, help="Parallel threads for fact extraction across samples"),
    checkpoint: Optional[str] = typer.Option(None, help="Checkpoint path for enoki_encoder extractor"),
    pre_extracted_facts_file: Optional[str] = typer.Option(None, help="Path to pre-extracted facts file (overrides default for pre_extracted_refchecker)"),
    first: Optional[int] = typer.Option(None, "--first", help="Limit to first N samples (for quick testing)"),
    filter_by_pre_extracted: bool = typer.Option(False, "--filter-by-pre-extracted", help="Evaluate only on samples present in the pre-extracted facts file"),
):
    """Evaluate sentence-level hallucination detection (FactCheckBench, ANAH, RAGTruth)."""
    if not all_datasets and dataset is None:
        typer.echo("Error: Must specify either --dataset or --all", err=True)
        raise typer.Exit(1)

    from evaluation.sentence import run_sentence_evaluation

    run_sentence_evaluation(
        dataset=dataset.value if dataset else None,
        method=method.value,
        extractor_method=extractor_method.value,
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        cache_dir=str(cache_dir),
        subset=None,
        max_length=max_length,
        chunk_size=chunk_size,
        coref=coref,
        force_recompute=force_recompute,
        all_datasets=all_datasets,
        incremental=incremental,
        use_preprocessing=preprocessing,

        chunk_overlap=chunk_overlap,
        extraction_workers=extraction_workers,
        checkpoint=checkpoint,
        pre_extracted_facts_file=pre_extracted_facts_file,
        limit=first,
        filter_by_pre_extracted=filter_by_pre_extracted,
    )


@evaluate_app.command("entity")
def evaluate_entity(
    dataset: EntityDataset = typer.Option(EntityDataset.halluentity, help="Dataset to evaluate"),
    extractor_method: ExtractorMethod = typer.Option(ExtractorMethod.stanford, help="Fact extractor method"),
    method: NLIMethod = typer.Option(NLIMethod.modernbert, help="NLI method"),
    data_dir: Path = typer.Option("data", help="Directory containing dataset files"),
    output_dir: Path = typer.Option("eval_results", help="Output directory"),
    cache_dir: Path = typer.Option("cache", help="Cache directory"),
    max_length: int = typer.Option(2048, help="Max sequence length"),
    chunk_size: int = typer.Option(16, help="Batch size for NLI"),
    coref: bool = typer.Option(False, help="Enable coreference resolution"),
    force_recompute: bool = typer.Option(False, help="Force recomputation"),
    chunk_overlap: int = typer.Option(1, help="Sentence overlap between premise chunks (0=no overlap, 1=default, ...)"),
    incremental: bool = typer.Option(True, help="Build incremental NP-modifier chains"),
    preprocessing: bool = typer.Option(False, help="Apply markdown masking + GLiNER to non-enoki extractors"),
    extraction_workers: int = typer.Option(1, help="Parallel threads for fact extraction across samples"),
    checkpoint: Optional[str] = typer.Option(None, help="Checkpoint path for enoki_encoder extractor"),
    pre_extracted_facts_file: Optional[str] = typer.Option(None, help="Path to pre-extracted facts file (overrides default for pre_extracted_refchecker)"),
    incremental_preextracted: bool = typer.Option(False, "--incremental-preextracted", help="Treat pre-extracted facts as incremental chains"),
):
    """Evaluate entity-level hallucination detection (HalluEntity)."""
    from evaluation.entity import run_entity_evaluation

    run_entity_evaluation(
        dataset=dataset.value,
        method=method.value,
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        cache_dir=str(cache_dir),
        max_length=max_length,
        chunk_size=chunk_size,
        coref=coref,
        force_recompute=force_recompute,
        chunk_overlap=chunk_overlap,
        incremental=incremental,
        use_preprocessing=preprocessing,
        extractor_method=extractor_method.value,
        extraction_workers=extraction_workers,
        checkpoint=checkpoint,
        pre_extracted_facts_file=pre_extracted_facts_file,
        incremental_preextracted=incremental_preextracted,
    )


@evaluate_app.command("span")
def evaluate_span(
    dataset: Optional[SpanDataset] = typer.Option(None, help="Dataset to evaluate"),
    method: NLIMethod = typer.Option(NLIMethod.modernbert, help="NLI method"),
    extractor_method: ExtractorMethod = typer.Option(ExtractorMethod.stanford, help="Fact extractor method"),
    data_dir: Path = typer.Option("data", help="Directory containing dataset files"),
    output_dir: Path = typer.Option("predictions", help="Output directory"),
    max_length: int = typer.Option(2048, help="Max sequence length"),
    threshold: float = typer.Option(0.5, help="Hallucination probability threshold"),
    hall_prob_mode: HallProbMode = typer.Option(HallProbMode.default, help="Hallucination probability mode"),
    coref: bool = typer.Option(False, help="Enable coreference resolution"),
    all_datasets: bool = typer.Option(False, "--all", help="Run on all span datasets"),
    force_recompute: bool = typer.Option(False, "--force-recompute", help="Bypass cached predictions and rerun inference"),
    parse_table: bool = typer.Option(False, "--parse-table", help="Generate per-sentence parse+facts CSV"),
    ragtruth_split: str = typer.Option("test", help="RAGTruth HuggingFace split (test or train)"),
    chunk_overlap: int = typer.Option(1, help="Sentence overlap between premise chunks (0=no overlap, 1=default, ...)"),
    limit: Optional[int] = typer.Option(None, help="Limit to first N examples (for debugging)"),
    save_facts: bool = typer.Option(True, "--save-facts", help="Save per-fact CSV with NLI scores and answer/context"),
    incremental: bool = typer.Option(False, help="Build incremental NP-modifier chains"),
    preprocessing: bool = typer.Option(False, help="Apply markdown masking + GLiNER to non-enoki extractors"),
    postfilter: bool = typer.Option(False, help="Drop boilerplate/discourse facts after extraction"),
    minie_workers: int = typer.Option(1, help="Parallel threads for MinIE per-sentence calls (>1 requires pyjnius>=1.4)"),
    extraction_workers: int = typer.Option(1, help="Parallel threads for fact extraction across rows"),
    checkpoint: Optional[str] = typer.Option(None, help="Checkpoint path for enoki_encoder extractor"),
    pre_extracted_facts_file: Optional[str] = typer.Option(None, help="Path to pre-extracted facts file (overrides default for pre_extracted_refchecker)"),
    train_pre_extracted_facts_file: Optional[str] = typer.Option(None, help="Pre-extracted facts file for the train split used during --calibrate"),
    calibrate: bool = typer.Option(False, "--calibrate", help="Calibrate threshold on train split (RAGTruth QA train, PsiloQA en train); MuSHROOM uses 0.5"),
    incremental_preextracted: bool = typer.Option(False, "--incremental-preextracted", help="Treat pre-extracted facts as incremental chains"),
):
    """Evaluate span-level hallucination detection (PsiloQA, Mushroom, RAGTruth)."""
    from evaluation.span import run_span_evaluation

    if not all_datasets and dataset is None:
        typer.echo("Error: Must specify either --dataset or --all", err=True)
        raise typer.Exit(1)

    run_span_evaluation(
        dataset=dataset.value if dataset else None,
        extractor_method=extractor_method.value,
        method=method.value,
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        max_length=max_length,
        threshold=threshold,
        hall_prob_mode=hall_prob_mode.value,
        coref=coref,
        all_datasets=all_datasets,
        force_recompute=force_recompute,
        parse_table=parse_table,
        ragtruth_split=ragtruth_split,
        chunk_overlap=chunk_overlap,
        limit=limit,
        save_facts=save_facts,
        incremental=incremental,
        use_preprocessing=preprocessing,
        postfilter=postfilter,
        max_workers=minie_workers,
        extraction_workers=extraction_workers,
        checkpoint=checkpoint,
        pre_extracted_facts_file=pre_extracted_facts_file,
        train_pre_extracted_facts_file=train_pre_extracted_facts_file,
        calibrate=calibrate,
        incremental_preextracted=incremental_preextracted,
    )


@app.command()
def threshold(
    predictions_file: Path = typer.Argument(..., help="Raw predictions file (*_preds.json)"),
    n_thresholds: int = typer.Option(50, help="Number of threshold grid points"),
    output: Optional[Path] = typer.Option(None, help="Save P/R/F1 curve data to this CSV file"),
    full: bool = typer.Option(False, "--full", help="Print all threshold points (not just best-F1)"),
    hall_prob_mode: HallProbMode = typer.Option(HallProbMode.default, help="How to compute hall_prob from NLI scores: default=contradiction+neutral, contradiction_only, neutral_only"),
):
    """Compute P/R/F1 curves across thresholds from a saved raw predictions file.

    The *_preds.json files are created automatically by the evaluate commands.
    Re-running the threshold command with different --n-thresholds or --hall-prob-mode
    does not require re-running inference.
    """
    from evaluation.predictions_io import run_threshold_analysis
    run_threshold_analysis(predictions_file, n_thresholds=n_thresholds, output_path=output, full=full, hall_prob_mode=hall_prob_mode.value)


@app.command()
def analyze(
    predictions_file: Path = typer.Argument(..., help="Raw predictions file (*_preds.json)"),
    threshold: float = typer.Option(0.5, help="hall_prob threshold to call a span a hallucination"),
    n_samples: int = typer.Option(50, help="Number of false-positive spans to sample"),
    output: Optional[Path] = typer.Option(None, help="Output CSV path (default: <preds_stem>_fp_analysis.csv)"),
    ragtruth_split: str = typer.Option("test", help="RAGTruth split to reload (test or train)"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Sample false-positive hallucination spans for manual review."""
    from evaluation.predictions_io import load_raw_predictions
    from evaluation.error_analysis import run_error_analysis

    raw = load_raw_predictions(predictions_file)
    dataset_name = raw.get("dataset", "")

    if dataset_name == "ragtruth":
        from evaluation.dataset_loaders import load_ragtruth_dataset
        loader = lambda: load_ragtruth_dataset(split=ragtruth_split)
    elif dataset_name == "psiloqa":
        from evaluation.dataset_loaders import load_psiloqa_dataset
        loader = load_psiloqa_dataset
    elif dataset_name == "mushroom":
        from evaluation.dataset_loaders import load_mushroom_dataset
        loader = lambda: load_mushroom_dataset("data")
    else:
        typer.echo(f"Error: unknown dataset '{dataset_name}' in preds file.", err=True)
        raise typer.Exit(1)

    if output is None:
        output = predictions_file.parent / (predictions_file.stem + "_fp_analysis.csv")

    run_error_analysis(
        preds_path=predictions_file,
        dataset_loader=loader,
        threshold=threshold,
        n_samples=n_samples,
        output_path=output,
        seed=seed,
    )


@app.command()
def version():
    """Show Enoki version."""
    typer.echo("Enoki v0.1.0")


# ── Train commands ────────────────────────────────────────────────────────────

@train_app.command("encoder")
def train_encoder(
    train: str = typer.Option("data/openie4_labels", help="OIE4 label file (train split)"),
    dev: Optional[str] = typer.Option(None, help="Separate dev label file (auto-split from train if omitted)"),
    dev_split: float = typer.Option(0.002, help="Fraction of training data for auto dev split"),
    no_negatives: bool = typer.Option(False, "--no-negatives", help="Filter out sentences with no extractions"),
    model: str = typer.Option("answerdotai/ModernBERT-large", help="Encoder model name or path"),
    iter_layers: int = typer.Option(2, help="Number of iterative transformer layers"),
    dropout: float = typer.Option(0.1, help="Dropout after each iterative step"),
    batch_size: int = typer.Option(32, help="Training batch size"),
    epochs: int = typer.Option(15, help="Number of training epochs"),
    lr: float = typer.Option(5e-5, help="Learning rate"),
    max_depth: int = typer.Option(14, help="Maximum extraction depth"),
    max_length: int = typer.Option(128, help="Maximum token length"),
    patience: int = typer.Option(3, help="Early stopping patience (epochs)"),
    warmup_steps: int = typer.Option(500, help="Linear warmup steps"),
    hungarian: bool = typer.Option(True, help="Use Hungarian matching for optimal depth-to-triple assignment"),
    empty_weight: float = typer.Option(0.1, help="Weight for EMPTY-matched depths in Hungarian loss"),
    workers: int = typer.Option(4, help="DataLoader worker threads"),
    seed: int = typer.Option(42, help="Random seed"),
    gpus: int = typer.Option(1, help="Number of GPUs (0 for CPU)"),
    out: str = typer.Option("checkpoints/", help="Checkpoint output directory"),
    checkpoint: Optional[str] = typer.Option(None, help="Resume from or transfer weights from this checkpoint"),
    save_weights_only: bool = typer.Option(True, help="Save only model weights (no optimizer state)"),
):
    """Train the IGL OpenIE encoder on OIE4-format label files.

    Trains on a single label file (stage 3 objective). Hungarian matching
    (--hungarian) finds the optimal depth-to-triple assignment per sentence
    via scipy.optimize.linear_sum_assignment; disabled by default.
    """
    from train_encoder import run_training

    run_training(
        train_fp=train,
        dev_fp=dev,
        dev_split=dev_split,
        allow_negatives=not no_negatives,
        model_name=model,
        iter_layers=iter_layers,
        dropout=dropout,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        max_depth=max_depth,
        max_length=max_length,
        patience=patience,
        warmup_steps=warmup_steps,
        hungarian=hungarian,

        empty_weight=empty_weight,

        workers=workers,
        seed=seed,
        gpus=gpus,
        out=out,
        checkpoint=checkpoint,
        save_weights_only=save_weights_only,
    )



@app.command("extract-triplets")
def extract_triplets(
    dataset: str = typer.Option(..., help=(
        "Dataset to process: ragtruth, ragtruth-sentence, psiloqa, halluentity, "
        "factcheckbench, mushroom, bench, anah"
    )),
    output: Path = typer.Option(..., help="Output JSONL path"),
    input_path: Optional[str] = typer.Option(None, "--input-path", help="Local JSON/JSONL input (required for mushroom, factcheckbench, bench, anah, ragtruth-sentence)"),
    model: str = typer.Option("gpt-oss-120b", help="Model name"),
    prompt: str = typer.Option("incremental", help="Prompt variant: 'incremental' (cycleoie_with_incrementality) or 'original' (cycleoie_original)"),
    temperature: float = typer.Option(0.0, help="Sampling temperature"),
    workers: int = typer.Option(1, help="Parallel source rows"),
    max_in_flight: Optional[int] = typer.Option(None, "--max-in-flight", help="Max queued futures (default: workers * 4)"),
    lang: str = typer.Option("en", help="Language for PsiloQA filter; use 'all' to disable"),
    text_field: Optional[str] = typer.Option(None, "--text-field", help="Override source text field name"),
    cache_dir: Optional[str] = typer.Option(None, "--cache-dir", help="HuggingFace dataset cache directory"),
    limit: Optional[int] = typer.Option(None, help="Limit to first N rows (for debugging)"),
    max_retries: int = typer.Option(8, help="Max retries per model call"),
    request_timeout: float = typer.Option(120.0, help="Timeout per model request (seconds)"),
    model_params: float = typer.Option(120e9, help="Parameter count for FLOPs approximation (2 * P * T)"),
    save_sentence_metrics: bool = typer.Option(False, "--save-sentence-metrics", help="Save per-sentence timing/token/FLOPs metrics"),
    no_resume: bool = typer.Option(False, "--no-resume", help="Do not skip already written IDs"),
    no_fsync: bool = typer.Option(False, "--no-fsync", help="Faster writes, less crash-safe"),
    ragtruth_task_type: str = typer.Option("QA", "--ragtruth-task-type", help="RAGTruth task type filter; use 'all' to disable"),
    ragtruth_sentence_text_field: str = typer.Option("sentence", "--ragtruth-sentence-text-field", help="Text field for ragtruth-sentence rows"),
    halluentity_split: str = typer.Option("train", "--halluentity-split", help="HF split for HalluEntity"),
    factcheckbench_text_field: str = typer.Option("decontext", "--factcheckbench-text-field", help="Nested field for factcheckbench: decontext, text, or revised_decontext"),
    factcheckbench_fallback_to_text: bool = typer.Option(False, "--factcheckbench-fallback-to-text", help="Fall back to .text when selected factcheckbench field is empty"),
    bench_text_field: str = typer.Option("sentence", "--bench-text-field", help="Text field for bench rows"),
    anah_text_field: str = typer.Option("sentence", "--anah-text-field", help="Text field for ANAH rows"),
    verbose: bool = typer.Option(False, "--verbose", help="Print abstain/parse/span diagnostics to stderr"),
    keep_unlocalized: bool = typer.Option(False, "--keep-unlocalized", help="Keep triples whose span cannot be mapped (stored as [-1,-1])"),
):
    """Extract pre-computed triplets via an OpenAI-compatible LLM backend.

    Reads credentials from environment variables or a .env file:
      OPENAI_API_KEY   (or LLM_PROXY_API_KEY / API_KEY)
      OPENAI_BASE_URL  (or LLM_PROXY_BASE_URL) -- optional

    Writes one JSON object per source example to the output JSONL file.
    Supports resuming: already-written IDs are skipped unless --no-resume is set.
    """
    from factextractor_backend import run_extraction

    run_extraction(
        dataset=dataset,
        output=str(output),
        input_path=input_path,
        model=model,
        prompt=prompt,
        temperature=temperature,
        workers=workers,
        max_in_flight=max_in_flight,
        lang=lang,

        text_field=text_field,
        cache_dir=cache_dir,
        limit=limit,
        max_retries=max_retries,
        request_timeout=request_timeout,
        model_params=model_params,
        save_sentence_metrics=save_sentence_metrics,
        no_resume=no_resume,
        no_fsync=no_fsync,
        ragtruth_task_type=ragtruth_task_type,
        ragtruth_sentence_text_field=ragtruth_sentence_text_field,
        halluentity_split=halluentity_split,
        factcheckbench_text_field=factcheckbench_text_field,
        factcheckbench_fallback_to_text=factcheckbench_fallback_to_text,
        bench_text_field=bench_text_field,
        anah_text_field=anah_text_field,
        verbose=verbose,
        keep_unlocalized=keep_unlocalized,
    )


if __name__ == "__main__":
    app()
