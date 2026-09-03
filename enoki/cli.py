#!/usr/bin/env python3
"""
Enoki CLI - Unified interface for hallucination detection evaluation.

Usage:
    enoki evaluate sentence --dataset factcheckbench --method modernbert
    enoki evaluate entity --dataset halluentity --method modernbert
    enoki evaluate span --dataset psiloqa --method modernbert
"""

import json
import sys

import typer
from typing import List, Optional
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
    enoki_encoder = "enoki-encoder"
    enoki_llm = "enoki-llm"
    enoki_rules = "enoki-rules"


class InferenceMethod(str, Enum):
    encoder = "encoder"
    llm = "llm"
    rules = "rules"


def _evaluation_extractor(method: ExtractorMethod) -> str:
    """Map public CLI names to the evaluation module's identifiers."""
    return method.value.replace("-", "_")


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


@app.command("extract")
def extract(
    method: InferenceMethod = typer.Option(
        InferenceMethod.encoder,
        help="Enoki extraction backend: encoder, llm, or rules",
    ),
    text: Optional[List[str]] = typer.Option(
        None,
        "--text",
        "-t",
        help="Text to process; repeat the option for multiple inputs",
    ),
    input_path: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        help="UTF-8 file with one input per non-empty line",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write JSON to this file instead of stdout",
    ),
    model: Optional[str] = typer.Option(
        None,
        help="HF model ID for encoder or API model name for llm",
    ),
    device: str = typer.Option("auto", help="Encoder device: auto, cpu, cuda, or mps"),
    min_confidence: float = typer.Option(0.7, help="Minimum encoder confidence"),
    top_k: int = typer.Option(10, help="Maximum encoder triples per input"),
    temperature: float = typer.Option(0.0, help="LLM sampling temperature"),
    prompt: str = typer.Option("incremental", help="LLM prompt: incremental or original"),
    max_retries: int = typer.Option(3, help="LLM request retries"),
    request_timeout: float = typer.Option(120.0, help="LLM request timeout in seconds"),
    enable_thinking: bool = typer.Option(False, help="Enable reasoning mode for compatible LLMs"),
    max_tokens: Optional[int] = typer.Option(None, help="Maximum LLM completion tokens"),
):
    """Extract OpenIE triples with Enoki-Encoder, Enoki-LLM, or Enoki-Rules."""
    if text and input_path:
        raise typer.BadParameter("Use either --text or --input, not both")

    if text:
        inputs = text
    elif input_path:
        inputs = [
            line.strip()
            for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif not sys.stdin.isatty():
        stdin_text = sys.stdin.read().strip()
        inputs = [stdin_text] if stdin_text else []
    else:
        raise typer.BadParameter("Pass --text/--input or pipe text through stdin")

    if not inputs:
        raise typer.BadParameter("No non-empty input text found")

    from enoki import EnokiPipeline

    pipeline = EnokiPipeline(
        method=method.value,
        model=model,
        device=device,
        min_confidence=min_confidence,
        top_k=top_k,
        temperature=temperature,
        prompt=prompt,
        max_retries=max_retries,
        request_timeout=request_timeout,
        enable_thinking=enable_thinking,
        max_tokens=max_tokens,
    )
    rendered = json.dumps(pipeline.extract(inputs), ensure_ascii=False, indent=2) + "\n"
    if output is None:
        typer.echo(rendered, nl=False)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")


@evaluate_app.command("sentence")
def evaluate_sentence(
    dataset: Optional[SentenceDataset] = typer.Option(None, help="Dataset to evaluate"),
    method: NLIMethod = typer.Option(NLIMethod.modernbert, help="NLI method"),
    extractor_method: ExtractorMethod = typer.Option(ExtractorMethod.enoki_rules, help="Fact extractor method"),
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
    llm_model: Optional[str] = typer.Option(None, help="API model for enoki-llm"),
    first: Optional[int] = typer.Option(None, "--first", help="Limit to first N samples (for quick testing)"),
):
    """Evaluate sentence-level hallucination detection (FactCheckBench, ANAH, RAGTruth)."""
    if not all_datasets and dataset is None:
        typer.echo("Error: Must specify either --dataset or --all", err=True)
        raise typer.Exit(1)

    from evaluation.sentence import run_sentence_evaluation

    run_sentence_evaluation(
        dataset=dataset.value if dataset else None,
        method=method.value,
        extractor_method=_evaluation_extractor(extractor_method),
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
        llm_model=llm_model,
        limit=first,
    )


@evaluate_app.command("entity")
def evaluate_entity(
    dataset: EntityDataset = typer.Option(EntityDataset.halluentity, help="Dataset to evaluate"),
    extractor_method: ExtractorMethod = typer.Option(ExtractorMethod.enoki_rules, help="Fact extractor method"),
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
    llm_model: Optional[str] = typer.Option(None, help="API model for enoki-llm"),
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
        extractor_method=_evaluation_extractor(extractor_method),
        extraction_workers=extraction_workers,
        checkpoint=checkpoint,
        llm_model=llm_model,
    )


@evaluate_app.command("span")
def evaluate_span(
    dataset: Optional[SpanDataset] = typer.Option(None, help="Dataset to evaluate"),
    method: NLIMethod = typer.Option(NLIMethod.modernbert, help="NLI method"),
    extractor_method: ExtractorMethod = typer.Option(ExtractorMethod.enoki_rules, help="Fact extractor method"),
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
    extraction_workers: int = typer.Option(1, help="Parallel threads for fact extraction across rows"),
    checkpoint: Optional[str] = typer.Option(None, help="Checkpoint path for enoki_encoder extractor"),
    llm_model: Optional[str] = typer.Option(None, help="API model for enoki-llm"),
    calibrate: bool = typer.Option(False, "--calibrate", help="Calibrate threshold on train split (RAGTruth QA train, PsiloQA en train); MuSHROOM uses 0.5"),
):
    """Evaluate span-level hallucination detection (PsiloQA, Mushroom, RAGTruth)."""
    from evaluation.span import run_span_evaluation

    if not all_datasets and dataset is None:
        typer.echo("Error: Must specify either --dataset or --all", err=True)
        raise typer.Exit(1)

    run_span_evaluation(
        dataset=dataset.value if dataset else None,
        extractor_method=_evaluation_extractor(extractor_method),
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
        extraction_workers=extraction_workers,
        checkpoint=checkpoint,
        llm_model=llm_model,
        calibrate=calibrate,
    )


@app.command()
def threshold(
    predictions_file: Path = typer.Argument(..., help="Raw predictions file (*_preds.json)"),
    n_thresholds: int = typer.Option(50, help="Number of threshold grid points"),
    output: Optional[Path] = typer.Option(
        None,
        help="Save threshold metrics to this CSV file (span runs also include IoU)",
    ),
    full: bool = typer.Option(False, "--full", help="Print all threshold points (not just best-F1)"),
    hall_prob_mode: HallProbMode = typer.Option(HallProbMode.default, help="How to compute hall_prob from NLI scores: default=contradiction+neutral, contradiction_only, neutral_only"),
):
    """Compute threshold curves from a saved raw predictions file.

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
    from model.train import run_training

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
    prompt: str = typer.Option("incremental", help="Prompt variant: 'incremental' or 'original'"),
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
    enable_thinking: bool = typer.Option(False, "--enable-thinking", help=(
        "Let hybrid-thinking models (e.g. Qwen3.6-35B-A3B) emit chain-of-thought before "
        "the KG output. OFF by default."
    )),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="Cap generated tokens per extraction call"),
):
    """Extract pre-computed triplets via an OpenAI-compatible LLM backend.

    Reads credentials from environment variables or a .env file:
      OPENAI_API_KEY   (or LLM_PROXY_API_KEY / API_KEY)
      OPENAI_BASE_URL  (or LLM_PROXY_BASE_URL) -- optional

    Writes one JSON object per source example to the output JSONL file.
    Supports resuming: already-written IDs are skipped unless --no-resume is set.
    """
    from fact_extractor.llm_backend import run_extraction

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
        enable_thinking=enable_thinking,
        max_tokens=max_tokens,
    )


if __name__ == "__main__":
    app()
