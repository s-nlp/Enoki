# Enoki

![Enoki OpenIE](assets/enoki-openie-banner.png)

Open Information Extraction for multi-level hallucination detection. Enoki
turns text into anchored relational facts, verifies them against evidence, and
maps unsupported facts back to hallucinated spans.

Enoki ships three interchangeable extraction methods behind one interface:

| Method | Best for | Runtime requirement |
| --- | --- | --- |
| **Enoki-Encoder** | Fast, local inference | [enoki-openie-encoder](https://huggingface.co/s-nlp/enoki-openie-encoder) |
| **Enoki-LLM** | Flexible extraction through an OpenAI-compatible API | API endpoint and credentials |
| **Enoki-Rules** | Deterministic, model-free extraction | spaCy English pipeline |

## Quick start

Clone the repository and install only the backend you need:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[encoder]"  # or .[llm], .[rules], .[all]
```

### Enoki-Encoder

The encoder is the default method and downloads
[`s-nlp/enoki-openie-encoder`](https://huggingface.co/s-nlp/enoki-openie-encoder)
on first use:

```bash
enoki extract \
  --method encoder \
  --text "Apple acquired Beats Electronics in 2014."
```

### Enoki-LLM

Configure an OpenAI-compatible endpoint, then select its model name:

```bash
export OPENAI_API_KEY="..."
# export OPENAI_BASE_URL="https://your-endpoint.example/v1"  # optional

enoki extract \
  --method llm \
  --model gpt-4o \
  --text "Apple acquired Beats Electronics in 2014."
```

### Enoki-Rules

Install the English transformer pipeline once after installing the `rules`
extra:

```bash
python -m spacy download en_core_web_trf
enoki extract \
  --method rules \
  --text "Apple acquired Beats Electronics in 2014."
```

All methods emit the same JSON fields: `text`, `triples`, `subject`,
`predicate`, `object`, and `confidence`. Pass `--input sentences.txt` for one
input per line, `--output triples.json` to save the result, or pipe text through
stdin. Run `enoki extract --help` for backend-specific options.

## Python API

```python
from enoki import EnokiPipeline

pipeline = EnokiPipeline(method="encoder")
results = pipeline.extract([
    "Barack Obama was born in Honolulu.",
    "Apple acquired Beats Electronics in 2014.",
])
```

Change `method` to `"llm"` or `"rules"`; the result schema remains the same.
Backend dependencies are imported lazily.

## Train Enoki-Encoder

Training remains available through the same CLI:

```bash
pip install -e ".[train]"
enoki train encoder \
  --train data/enoki_encoder_train/train_labels \
  --dev data/enoki_encoder_train/val_labels \
  --model answerdotai/ModernBERT-large \
  --epochs 15 \
  --batch-size 32 \
  --out checkpoints/
```

Fine-tune from an existing checkpoint with `--checkpoint`. The training CLI
reads OIE4-style label files. Run `enoki train encoder --help` for all training
and resume options. Local Hugging Face export directories and publication
tooling are intentionally excluded from Git.

## Validate on benchmarks

### Span-level results

Span Coverage F1 (%) from [Table 3 of the paper](https://arxiv.org/abs/2609.00581);
higher is better, and bold marks the best displayed result for each benchmark.
Selected strong baselines from the same table are included for context.

| Method | Extractor | MuSHROOM | RAGTruth | PsiloQA |
| --- | --- | ---: | ---: | ---: |
| **Enoki-LLM** | GPT-OSS-120B | **52.07** | 37.32 | **71.15** |
| **Enoki-Rules** | Rule-based | 49.18 | 27.87 | 65.73 |
| **Enoki-Encoder** | ModernBERT-large | 46.96 | 34.84 | 65.51 |
| OpenIE | MinIE | 44.12 | 28.09 | 64.06 |
| FT on RAGTruth | Qwen3-8B | 4.33 | **42.20** | 23.81 |
| haldetect | ModernBERT-base-32k | 11.43 | 41.54 | 27.70 |
| ZS RAGTruth Prompt | GPT-5.2 | 5.67 | 35.97 | 39.17 |

```bash
pip install -e ".[benchmark]"

enoki evaluate sentence \
  --dataset factcheckbench \
  --extractor-method enoki-rules

enoki evaluate span \
  --dataset ragtruth \
  --extractor-method enoki-encoder \
  --checkpoint checkpoints/best.ckpt
```

Enoki-LLM facts are precomputed so interrupted runs can resume and one
extraction can be reused across NLI settings:

```bash
enoki extract-triplets \
  --dataset ragtruth \
  --output data/pre_extracted/ragtruth_test.jsonl \
  --model gpt-4o

enoki evaluate span \
  --dataset ragtruth \
  --extractor-method enoki-llm \
  --pre-extracted-facts-file data/pre_extracted/ragtruth_test.jsonl
```

Use `enoki evaluate --help` and the level-specific `--help` commands for all
datasets, NLI methods, caching, and output options. Additional reproducibility
tools live in `scripts/benchmarks/`.

## EnokiQA

[EnokiQA](https://huggingface.co/datasets/s-nlp/EnokiQA)
is a long-form QA benchmark with claim-level verification labels aligned to
span-level localization. It contains 3,990 labeled examples across seven
generator models and 19,594 unlabeled question-answer-context triples.

## Repository map

```text
enoki/                 Public Python inference API and CLI
fact_extractor/        Encoder, LLM, and rules extraction implementations
model/                 Enoki-Encoder training model and data code
evaluation/            Sentence, span, and entity benchmark runners
nli/                   Verification backends
data/                  Included benchmark samples and encoder labels
scripts/               Benchmark and inference utilities
baselines/             External baseline integrations and result utilities
```

For source checkouts, the same CLI is available through `python -m enoki`.

## Citation

```bibtex
@misc{rykov2026enokiefficientmultilevelhallucination,
      title={Enoki: Efficient Multi-Level Hallucination Detection},
      author={Elisei Rykov and Timur Ionov and Nikolay Ivanov and Maksim Savkin and Maksim Makarenko and Alexander Panchenko and Vasily Konovalov and Julia Belikova},
      year={2026},
      eprint={2609.00581},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.00581},
}
```
