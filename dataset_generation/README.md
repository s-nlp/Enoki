# EnokiQA Dataset Generation Pipeline

This directory contains all scripts used to generate the EnokiQA dataset from scratch.
The pipeline mines Wikipedia, generates questions, collects model answers, and creates balanced train/test splits.

## Prerequisites

- Python 3.10+
- An OpenAI-compatible LLM server (e.g., [vLLM](https://github.com/vllm-project/vllm)) for question generation, filtering, and answer generation
- Internet access for Wikipedia API calls

### Python Dependencies

```bash
pip install aiohttp openai pydantic orjson tqdm
```

### Starting a vLLM Server

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <model_name_or_path> \
  --port 8000 \
  --tensor-parallel-size <num_gpus>
```

## Pipeline Overview

```
Step 1: wiki_miner.py          → meta_en.raw.jsonl
Step 2: downsample_meta.py     → meta_en.sampled.jsonl
Step 3: paragraph_miner.py     → paragraphs.jsonl
Step 4: qa_gen_fast.py         → questions_long.jsonl
Step 5: qa_filter.py           → questions_long.llm_filtered.jsonl
Step 6: hallucination_answerer.py (×N models, 2 modes each)
        → hallucinations/<model>/answers_no_context.jsonl
        → hallucinations/<model>/answers_with_context.jsonl
Step 7: fetch_full_pages.py    → full_pages.jsonl
Step 8: create_balanced_split.py → enoki_{train,test}.jsonl
Step 9: export_enokiqa.py      → enokiqa_{train,test}.jsonl
```

## Step-by-Step Instructions

All outputs go to `data/wiki/en/` by default. Adjust paths as needed.

### Step 1: Mine Wikipedia Metadata

Collects page metadata, categories, pageviews, and Wikidata QIDs using the MediaWiki and Pageviews APIs.

```bash
python dataset_generation/wiki_miner.py \
  --lang en \
  --n-pages 80000 \
  --seed-mode featured \
  --pv-days 90 \
  --out data/wiki/en/meta_en.raw.jsonl \
  --errors-out data/wiki/en/meta_en.raw.errors.jsonl
```

Or use the wrapper: `bash dataset_generation/scripts/wiki_mine_en.sh`

### Step 2: Downsample Metadata

Balances the dataset by pageview distribution and topic categories.

```bash
python dataset_generation/scripts/downsample_meta.py \
  --meta data/wiki/en/meta_en.raw.jsonl \
  --out data/wiki/en/meta_en.sampled20k.jsonl \
  --count 20000
```

### Step 3: Mine Paragraphs

Extracts and splits Wikipedia article text into paragraphs.

```bash
python dataset_generation/paragraph_miner.py \
  --lang en \
  --in-meta data/wiki/en/meta_en.sampled20k.jsonl \
  --out data/wiki/en/paragraphs.sampled.jsonl \
  --errors-out data/wiki/en/paragraphs.sampled.errors.jsonl \
  --min-paragraph-chars 200 \
  --max-paragraph-chars 2000
```

Or use the wrapper: `bash dataset_generation/scripts/para_mine_en.sh`

### Step 4: Generate Questions

Uses an LLM to generate long-form factual questions grounded in paragraphs.

```bash
python dataset_generation/qa_gen_fast.py \
  --input data/wiki/en/paragraphs.sampled.jsonl \
  --out data/wiki/en/questions_long.jsonl \
  --errors-out data/wiki/en/questions_long.errors.jsonl \
  --model openai/<model_name> \
  --questions-per-context 3 \
  --workers 12
```

Or use the wrapper: `bash dataset_generation/scripts/qa_gen_en.sh`

**Environment variables:** `OPENAI_BASE_URL` (default: `http://localhost:8000/v1`), `MODEL`, `TEMP`, `MAX_TOKENS`, `CONCURRENCY`

### Step 5: Filter Questions

LLM-based quality filter that checks grounding, quality, and answerability.

```bash
python dataset_generation/qa_filter.py \
  --input data/wiki/en/questions_long.jsonl \
  --out data/wiki/en/questions_long.llm_filtered.jsonl \
  --errors-out data/wiki/en/questions_long.llm_filtered.errors.jsonl \
  --base-url http://localhost:8000/v1 \
  --model openai/<model_name> \
  --concurrency 12
```

### Step 6: Generate Answers

Generates model answers in two modes:
- **no_context**: Model answers from parametric knowledge only (hallucination-prone)
- **with_context**: Model answers with the source paragraph as context (grounded)

Run for each generator model:

```bash
# No-context answers
python dataset_generation/hallucination_answerer.py \
  --input data/wiki/en/questions_long.llm_filtered.jsonl \
  --out data/wiki/en/hallucinations/<model_slug>/answers_no_context.jsonl \
  --errors-out data/wiki/en/hallucinations/<model_slug>/answers_no_context.errors.jsonl \
  --base-url http://localhost:8000/v1 \
  --model openai/<model_name> \
  --prompt-mode no_context \
  --concurrency 12

# With-context answers
python dataset_generation/hallucination_answerer.py \
  --input data/wiki/en/questions_long.llm_filtered.jsonl \
  --out data/wiki/en/hallucinations/<model_slug>/answers_with_context.jsonl \
  --errors-out data/wiki/en/hallucinations/<model_slug>/answers_with_context.errors.jsonl \
  --base-url http://localhost:8000/v1 \
  --model openai/<model_name> \
  --prompt-mode with_context \
  --concurrency 12
```

Or use the wrapper: `MODEL=openai/<model> BASE_URL=http://localhost:8000/v1 bash dataset_generation/scripts/hallucinate_en.sh`

**Models used in EnokiQA:** Qwen2.5-{7B, 14B, 32B}-Instruct, Qwen3-{4B, 8B}, Llama-3.1-8B-Instruct, Mixtral-8x7B-Instruct

### Step 7: Fetch Full Wikipedia Pages

Fetches complete Wikipedia article text for verification context.

```bash
python dataset_generation/scripts/fetch_full_pages.py \
  --splits data/splits/enoki_train.jsonl data/splits/enoki_test.jsonl \
  --meta data/wiki/en/meta_en.sampled20k.jsonl \
  --output data/wiki/en/full_pages.jsonl
```

### Step 8: Create Balanced Splits

Creates train/test splits with the test set balanced by model, popularity tier, and answer length.

```bash
python dataset_generation/scripts/create_balanced_split.py \
  --data-dir data/wiki/en/hallucinations \
  --meta-path data/wiki/en/meta_en.sampled20k.jsonl \
  --output-dir data/splits \
  --test-size 2000
```

### Step 9: Export Final Dataset

Joins splits with all metadata (Wikipedia info, pageviews, full pages, annotations) into a single JSONL per split.

```bash
python dataset_generation/scripts/export_enokiqa.py \
  --splits data/splits/enoki_train.jsonl data/splits/enoki_test.jsonl \
  --meta data/wiki/en/meta_en.sampled20k.jsonl \
  --full-pages data/wiki/en/full_pages.jsonl \
  --output-dir data/enokiqa_export
```

## Optional Steps

### Answer Quality Filtering

Filter generated answers to remove refusals and low-quality responses:

```bash
python dataset_generation/filter_answers_llm.py \
  --input data/wiki/en/hallucinations/<model>/answers_no_context.jsonl \
  --out data/wiki/en/hallucinations/<model>/answers_no_context.filtered.jsonl \
  --base-url http://localhost:8000/v1 \
  --model openai/<model_name>
```

### NER Entity Extraction

Extract named entities for analysis:

```bash
python dataset_generation/scripts/ner_gliner_stats.py \
  --input data/wiki/en/questions_long.llm_filtered.jsonl \
  --output data/wiki/en/ner_gliner.questions.jsonl
```

### Dataset Statistics

Compute length distributions and entity statistics:

```bash
python dataset_generation/dataset_stats.py \
  --input data/wiki/en/questions_long.llm_filtered.jsonl \
  --out-json data/wiki/en/stats.json \
  --plot-prefix data/wiki/en/plots/stats
```

## Key Design Patterns

- **Resume by default**: All scripts track processed IDs and skip already-processed records. Use `--no-resume` to force reprocessing.
- **JSONL streaming**: All I/O uses line-delimited JSON for memory efficiency and crash recovery.
- **Error tracking**: Failed records are written to `--errors-out` with `_error: true` for debugging.
- **Async concurrency**: Wikipedia miners use `aiohttp` with rate limiting. LLM scripts use `AsyncOpenAI` with configurable concurrency.

## Data Flow

```
Wikipedia API
    │
    ▼
meta_en.raw.jsonl (80k pages with metadata, categories, pageviews)
    │
    ▼ downsample_meta.py
meta_en.sampled20k.jsonl (20k balanced pages)
    │
    ▼ paragraph_miner.py
paragraphs.sampled.jsonl (paragraphs with sentence splits)
    │
    ▼ qa_gen_fast.py
questions_long.jsonl (3 questions per paragraph)
    │
    ▼ qa_filter.py
questions_long.llm_filtered.jsonl (quality-filtered)
    │
    ├─▶ hallucination_answerer.py (no_context)
    │       └→ answers_no_context.jsonl (×7 models)
    │
    └─▶ hallucination_answerer.py (with_context)
            └→ answers_with_context.jsonl (×7 models)
    │
    ▼ create_balanced_split.py
enoki_train.jsonl + enoki_test.jsonl
    │
    ▼ export_enokiqa.py + fetch_full_pages.py
enokiqa_train.jsonl + enokiqa_test.jsonl (final dataset with all metadata)
```
