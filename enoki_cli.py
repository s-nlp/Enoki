#!/usr/bin/env python3
"""
Enoki CLI - Unified interface for hallucination detection evaluation.

Usage:
    python enoki_cli.py evaluate sentence --dataset felm --method modernbert
    python enoki_cli.py evaluate entity --dataset halluentity --method alignscore
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


# Enums for choices
class NLIMethod(str, Enum):
    modernbert = "modernbert"
    alignscore = "alignscore"
    qwen_06b = "qwen_06b"
    qwen_4b = "qwen_4b"
    qwen_8b = "qwen_8b"


class SentenceDataset(str, Enum):
    felm = "felm"
    factcheckbench = "factcheckbench"


class EntityDataset(str, Enum):
    halluentity = "halluentity"


class SpanDataset(str, Enum):
    psiloqa = "psiloqa"
    mushroom = "mushroom"
    ragtruth = "ragtruth"


class FELMSubset(str, Enum):
    wk = "wk"
    science = "science"
    writing_rec = "writing_rec"


class HallProbMode(str, Enum):
    default = "default"
    contradiction_only = "contradiction_only"
    neutral_only = "neutral_only"


@evaluate_app.command("sentence")
def evaluate_sentence(
    dataset: SentenceDataset = typer.Option(..., help="Dataset to evaluate"),
    method: NLIMethod = typer.Option(NLIMethod.modernbert, help="NLI method"),
    data_dir: Path = typer.Option("data", help="Directory containing dataset files"),
    output_dir: Path = typer.Option("eval_results", help="Output directory"),
    cache_dir: Path = typer.Option("cache", help="Cache directory"),
    subset: Optional[FELMSubset] = typer.Option(None, help="FELM subset (wk/science/writing_rec)"),
    max_length: int = typer.Option(2048, help="Max sequence length"),
    chunk_size: int = typer.Option(16, help="Batch size for NLI"),
    coref: bool = typer.Option(True, help="Enable coreference resolution"),
    force_recompute: bool = typer.Option(False, help="Force recomputation"),
    all_datasets: bool = typer.Option(False, "--all", help="Run on all datasets"),
):
    """Evaluate sentence-level hallucination detection (FELM, FactCheckBench)."""
    from evaluation.sentence import run_sentence_evaluation

    run_sentence_evaluation(
        dataset=dataset.value,
        method=method.value,
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        cache_dir=str(cache_dir),
        subset=subset.value if subset else None,
        max_length=max_length,
        chunk_size=chunk_size,
        coref=coref,
        force_recompute=force_recompute,
        all_datasets=all_datasets,
    )


@evaluate_app.command("entity")
def evaluate_entity(
    dataset: EntityDataset = typer.Option(EntityDataset.halluentity, help="Dataset to evaluate"),
    method: NLIMethod = typer.Option(NLIMethod.modernbert, help="NLI method"),
    data_dir: Path = typer.Option("data", help="Directory containing dataset files"),
    output_dir: Path = typer.Option("eval_results", help="Output directory"),
    cache_dir: Path = typer.Option("cache", help="Cache directory"),
    max_length: int = typer.Option(2048, help="Max sequence length"),
    chunk_size: int = typer.Option(16, help="Batch size for NLI"),
    coref: bool = typer.Option(True, help="Enable coreference resolution"),
    force_recompute: bool = typer.Option(False, help="Force recomputation"),
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
    )


@evaluate_app.command("span")
def evaluate_span(
    dataset: Optional[SpanDataset] = typer.Option(None, help="Dataset to evaluate"),
    method: NLIMethod = typer.Option(NLIMethod.modernbert, help="NLI method"),
    data_dir: Path = typer.Option("data", help="Directory containing dataset files"),
    output_dir: Path = typer.Option("predictions", help="Output directory"),
    max_length: int = typer.Option(2048, help="Max sequence length"),
    threshold: float = typer.Option(0.5, help="Hallucination probability threshold"),
    hall_prob_mode: HallProbMode = typer.Option(HallProbMode.default, help="Hallucination probability mode"),
    coref: bool = typer.Option(True, help="Enable coreference resolution"),
    all_datasets: bool = typer.Option(False, "--all", help="Run on all span datasets"),
):
    """Evaluate span-level hallucination detection (PsiloQA, Mushroom, RAGTruth)."""
    from evaluation.span import run_span_evaluation

    if not all_datasets and dataset is None:
        typer.echo("Error: Must specify either --dataset or --all", err=True)
        raise typer.Exit(1)

    run_span_evaluation(
        dataset=dataset.value if dataset else None,
        method=method.value,
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        max_length=max_length,
        threshold=threshold,
        hall_prob_mode=hall_prob_mode.value,
        coref=coref,
        all_datasets=all_datasets,
    )


@app.command()
def version():
    """Show Enoki version."""
    typer.echo("Enoki v0.1.0")


if __name__ == "__main__":
    app()
