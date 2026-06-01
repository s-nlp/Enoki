# <img src="assets/logo.png" width="35" height="35" alt="i love enoki"> Enoki

An Open Information Extraction framework for multi-level hallucination detection. Enoki extracts text-anchored relational facts, verifies them against evidence, and projects unsupported facts back to hallucinated spans — enabling claim-level verification and span-level localization through a single shared representation, with LLM-based, encoder-based, and rule-based extraction backends.

## EnokiQA Dataset

A long-form QA benchmark for hallucination detection with dual granularity: claim-level verification labels aligned to span-level localization. Contains 3,990 labeled examples across seven generator models and 19,594 unlabeled question-answer-context triples.

[EnokiQA Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6TN4ZM)

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Enoki-LLM

Extracts facts using an OpenAI-compatible LLM, then evaluates hallucination with NLI.

### 1. Configure credentials

Create a `.env` file in the project root (or export the variables directly):

```
OPENAI_API_KEY=your-key-here
OPENAI_BASE_URL=https://your-proxy/v1   # optional; omit for api.openai.com
```

### 2. Extract triplets

```bash
python enoki_cli.py extract-triplets \
    --dataset ragtruth \
    --output data/pre_extracted/ragtruth_test.jsonl \
    --model gpt-4o \
    --workers 4
```

Supported datasets: `ragtruth`, `ragtruth-sentence`, `psiloqa`, `halluentity`, `factcheckbench`, `mushroom`, `bench`, `anah`.
Datasets loaded from local files (`mushroom`, `factcheckbench`, `bench`, `anah`, `ragtruth-sentence`) also require `--input-path`.

Key extraction options:

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `gpt-oss-120b` | Model name passed to the API |
| `--prompt` | `incremental` | Prompt variant: `incremental` (with incremental argument spans) or `original` |
| `--workers` | `1` | Parallel rows; each row is processed sentence-by-sentence |
| `--output` | — | Output JSONL path (resumes automatically if file exists) |
| `--no-resume` | off | Disable resume; re-extract all rows |
| `--limit` | — | Process only the first N rows (debugging) |

### 3. Evaluate

Pass the extracted file with `--extractor-method enoki-llm` and `--pre-extracted-facts-file`.

**Sentence-level** (FactCheckBench / ANAH / RAGTruth):
```bash
python enoki_cli.py evaluate sentence \
    --dataset factcheckbench \
    --extractor-method enoki-llm \
    --pre-extracted-facts-file data/pre_extracted/factcheckbench.jsonl
```

**Span-level** (RAGTruth / PsiloQA / MuSHROOM):
```bash
python enoki_cli.py evaluate span \
    --dataset ragtruth \
    --extractor-method enoki-llm \
    --pre-extracted-facts-file data/pre_extracted/ragtruth_test.jsonl
```

**Entity-level** (HalluEntity):
```bash
python enoki_cli.py evaluate entity \
    --extractor-method enoki-llm \
    --pre-extracted-facts-file data/pre_extracted/halluentity.jsonl
```

---

## Enoki-Encoder

Neural OIE encoder (IGL — Iterative Grid Labeling) based on ModernBERT. Facts are extracted at runtime; no pre-extraction step is needed.

### Train

```bash
python enoki_cli.py train encoder \
    --train data/enoki-encoder_train/train_labels \
    --dev   data/enoki-encoder_train/val_labels \
    --model answerdotai/ModernBERT-large \
    --epochs 15 \
    --batch-size 32 \
    --out checkpoints/
```

Fine-tune from an existing checkpoint:
```bash
python enoki_cli.py train encoder \
    --train data/train_labels \
    --dev   data/val_labels \
    --checkpoint checkpoints/pretrained.ckpt \
    --epochs 5
```

Key training options:

| Option | Default | Description |
|--------|---------|-------------|
| `--train` | `data/enoki-encoder_train/train_labels` | OIE4-format label file (train split) |
| `--dev` | — | Dev label file (auto-splits from train if omitted) |
| `--model` | `answerdotai/ModernBERT-large` | Encoder model name or HF path |
| `--epochs` | `15` | Training epochs |
| `--batch-size` | `32` | Batch size |
| `--lr` | `5e-5` | Learning rate |
| `--max-depth` | `14` | Maximum extraction depth |
| `--hungarian` | on | Optimal depth-to-triple assignment via Hungarian algorithm |
| `--checkpoint` | — | Resume from or transfer weights from a checkpoint |
| `--out` | `checkpoints/` | Output directory for saved checkpoints |

### Evaluate

**Sentence-level** (FactCheckBench / ANAH / RAGTruth):
```bash
python enoki_cli.py evaluate sentence \
    --dataset factcheckbench \
    --extractor-method enoki-encoder \
    --checkpoint checkpoints/best.ckpt
```

**Span-level** (RAGTruth / PsiloQA / MuSHROOM):
```bash
python enoki_cli.py evaluate span \
    --dataset ragtruth \
    --extractor-method enoki-encoder \
    --checkpoint checkpoints/best.ckpt
```

**Entity-level** (HalluEntity):
```bash
python enoki_cli.py evaluate entity \
    --extractor-method enoki-encoder \
    --checkpoint checkpoints/best.ckpt
```

---

## Enoki-Rules

Rule-based fact extractor. No model or pre-extraction step required.

**Sentence-level** (FactCheckBench / ANAH / RAGTruth):
```bash
python enoki_cli.py evaluate sentence \
    --dataset factcheckbench \
    --extractor-method enoki-rules
```

**Span-level** (RAGTruth / PsiloQA / MuSHROOM):
```bash
python enoki_cli.py evaluate span \
    --dataset ragtruth \
    --extractor-method enoki-rules
```

**Entity-level** (HalluEntity):
```bash
python enoki_cli.py evaluate entity \
    --extractor-method enoki-rules
```

---

## Other backends

### Stanford OpenIE

Requires `CORENLP_HOME` to point to a CoreNLP installation (downloaded automatically on first run).

```bash
python enoki_cli.py evaluate sentence \
    --dataset factcheckbench \
    --extractor-method stanford

python enoki_cli.py evaluate span \
    --dataset ragtruth \
    --extractor-method stanford

python enoki_cli.py evaluate entity \
    --extractor-method stanford
```

### MinIE (safe mode)

Requires a MinIE JAR (set `MINIE_JAR`) and a Java runtime (`JAVA_HOME`).

```bash
python enoki_cli.py evaluate sentence \
    --dataset factcheckbench \
    --extractor-method minie_safe

python enoki_cli.py evaluate span \
    --dataset ragtruth \
    --extractor-method minie_safe

python enoki_cli.py evaluate entity \
    --extractor-method minie_safe
```

---

## Common evaluation options

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | — | Dataset name (see command help for valid values) |
| `--extractor-method` | `stanford` | `stanford`, `minie_safe`, `enoki-encoder`, `enoki-rules`, `enoki-llm` |
| `--method` | `modernbert` | NLI method |
| `--checkpoint` | — | Path to `.ckpt` (required for `enoki-encoder`) |
| `--pre-extracted-facts-file` | — | Pre-extracted JSONL (required for `enoki-llm`) |
| `--data-dir` | `data` | Dataset root directory |
| `--output-dir` | `eval_results` / `predictions` | Where results are written |
| `--force-recompute` | off | Ignore cached predictions and rerun |
| `--incremental` | on/off | Build incremental NP-modifier chains |
| `--coref` | off | Enable coreference resolution |
| `--first N` | — | Limit to first N samples (quick testing) |
