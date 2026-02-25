# Baselines

Baselines for claim extraction / factuality evaluation over **FactBench** and **FELM** using offline evidence.

## Data
- FactBench JSONL: `data/factcheck-GPT-benchmark.jsonl`
- FELM dataset: `data/felm_with_ref_text` (HuggingFace `datasets` format, with subset folders)

## Methods
- `AlignScore/`: AlignScore scoring of (context, claim) pairs on FactBench.
- `Claimify/`: Claimify extraction with optional verification (supports vLLM, OpenRouter, OpenAI backends).
- `SAFE/`: SAFE-like extraction → decontextualize → relevance → verify pipeline.
- `VeriScore/`: VeriScore-like extraction + BM25 passage selection + verification.
- `factowl/`: FactOwl-based pipeline with offline evidence.

Each method has its own README with exact flags and run examples.

## Outputs
Most runners write `metrics.json` plus a per-sentence JSONL in their output folder. See each method README for the exact filenames.
