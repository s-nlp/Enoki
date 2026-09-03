# Enoki

![Enoki OpenIE](assets/enoki-openie-banner.png)

Open Information Extraction for multi-level hallucination detection. Enoki
turns text into anchored relational facts, verifies them against evidence, and
maps unsupported facts back to hallucinated spans.

Enoki ships three interchangeable extraction methods behind one interface:

| Method | Best for | Runtime requirement |
| --- | --- | --- |
| **Enoki-Encoder** | Fast, local inference | [ModernBERT model](https://huggingface.co/s-nlp/enoki-openie-encoder) |
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

Fine-tune from an existing checkpoint with `--checkpoint`. To export a trained
checkpoint as a self-contained Hugging Face repository:

```bash
python scripts/release/encoder/build_hf_encoder.py \
  --checkpoint checkpoints/best.ckpt \
  --output hf_enoki_openie_encoder
```

The training CLI reads OIE4-style label files. Run
`enoki train encoder --help` for all training and resume options.

## Validate on benchmarks

Enoki evaluates sentence-, span-, and entity-level hallucinations:

| Level | Datasets |
| --- | --- |
| Sentence | FactCheckBench, ANAH, RAGTruth |
| Span | RAGTruth, PsiloQA, MuSHROOM |
| Entity | HalluEntity |

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
tools live in `scripts/benchmarks/`. Span validation reports span coverage F1
as the primary metric and mean character-level IoU as the secondary metric.

## EnokiQA

[EnokiQA](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6TN4ZM)
is a long-form QA benchmark with claim-level verification labels aligned to
span-level localization. It contains 3,990 labeled examples across seven
generator models and 19,594 unlabeled question-answer-context triples.

## Repository map

```text
enoki/                 Public Python inference API
fact_extractor/        Encoder, LLM, and rules extraction implementations
model/                 Enoki-Encoder training model and data code
evaluation/            Sentence, span, and entity benchmark runners
nli/                   Verification backends
data/                  Included benchmark samples and encoder labels
scripts/               Benchmark, inference, and Hugging Face release tools
baselines/             External baseline integrations and result utilities
```

`python enoki_cli.py ...` remains supported for source checkouts; installing
the project adds the shorter `enoki ...` command used above.
