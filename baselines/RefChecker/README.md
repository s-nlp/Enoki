# RefChecker

RefChecker baseline runner for **FactBench**, **FELM**, and **ANAH** with offline evidence.

## Pipeline
The runner applies a simple claim-checking pipeline at the sentence level:

1. **Extraction**: RefChecker's LLM extractor turns a sentence into claims.
2. **Checking**: RefChecker verifies each extracted claim against the provided reference.
3. **Aggregation**: the final sentence label is computed with a strict rule:
   `supported` only if **every** extracted claim is labeled `Entailment`.

Very briefly, the stages work like this:

- `triplet` mode extracts `(subject, relation, object)` claims.
- `subsentence` mode extracts sentence-like claims and is usually a better fit for sentence-level benchmarks.
- the checker assigns `Entailment`, `Neutral`, or `Contradiction` to each claim.
- the runner then converts claim labels into a single strict sentence verdict.

The runner writes `metrics.json` and `segments.jsonl` to `--out_root`. Per-sentence timing is included as `extract_s`, `verify_s`, and `total_s`.

Metrics include macro-F1 with positive class `not_supported` and `roc_auc_not_supported`, where the score is the fraction of claims with a non-`Entailment` verdict.

## Undefined Predictions
`--no_claim_policy_all` is kept only as a deprecated alias. Use `--undefined_prediction_policy` instead:

- `penalize`: treat sentences without a strict prediction as `not_supported`
- `skip`: exclude such sentences from metrics

For strict sentence-level evaluation, `penalize` is usually the right choice.

## Claim Format
`--claim_format` supports:

- `triplet`
- `subsentence`

For sentence-level setups closer to Claimify / SAFE / VeriScore, prefer `subsentence`.

If the installed `refchecker` version does not support `subsentence`, the runner falls back automatically and records this in `metrics.json` via:

- `claim_format_requested`
- `claim_format_effective`
- `claim_format_note`

## Compute Estimates
If you pass `--model_params_b`, the runner writes two compute-style estimates:

- `visible_*`: a lower bound based on visible text only (`sentence`, `question`, `reference`, `claims`, `labels`)
- main fields without a suffix: an `adjusted` proxy estimate when prompt-overhead flags are provided

This is still only an **estimate**, because RefChecker does not expose true provider-side token usage or internal prompt tokens.

## Installation
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

For LLM extraction/checking, configure the provider key used by LiteLLM, for example `OPENAI_API_KEY` for OpenAI models.

For local models, use `--run_local_vllm`: the runner will start an OpenAI-compatible vLLM endpoint, wait until it is ready, run RefChecker, and stop the server afterwards.

## Usage
### FELM
```bash
python run_refchecker.py \
  --dataset felm \
  --felm_dir ../data/felm_with_ref_text \
  --subset wk \
  --split test \
  --out_root ./out/refchecker_felm_wk \
  --extractor_name gpt-4o \
  --checker_type llm \
  --checker_name gpt-4o \
  --claim_format triplet \
  --batch_size_extractor 8 \
  --batch_size_checker 8 \
  --undefined_prediction_policy penalize
```

### FELM sentence-level variant
For a comparison that is more aligned with sentence-level pipelines, try:
```bash
python run_refchecker.py \
  --dataset felm \
  --felm_dir ../data/felm_with_ref_text \
  --subset wk \
  --split test \
  --out_root ./out/refchecker_felm_wk_subsentence \
  --extractor_name gpt-4o \
  --checker_type llm \
  --checker_name gpt-4o \
  --claim_format subsentence \
  --batch_size_extractor 8 \
  --batch_size_checker 8 \
  --undefined_prediction_policy penalize
```

### FELM with prompt-overhead proxy compute
```bash
python run_refchecker.py \
  --dataset felm \
  --felm_dir ../data/felm_with_ref_text \
  --subset wk \
  --split test \
  --out_root ./out/refchecker_felm_wk_subsentence_proxy_compute \
  --extractor_name gpt-4o \
  --checker_type llm \
  --checker_name gpt-4o \
  --claim_format subsentence \
  --compute_extract_prompt_overhead_tokens 120 \
  --compute_verify_prompt_overhead_tokens 220 \
  --compute_verify_per_claim_overhead_tokens 12 \
  --undefined_prediction_policy penalize
```

### FactBench
```bash
python run_refchecker.py \
  --dataset factbench \
  --data ../data/factcheck-GPT-benchmark.jsonl \
  --out_root ./out/refchecker_factbench \
  --extractor_name gpt-4o \
  --checker_type llm \
  --checker_name gpt-4o \
  --claim_format subsentence \
  --batch_size_extractor 8 \
  --batch_size_checker 8 \
  --undefined_prediction_policy penalize
```

### ANAH from HuggingFace
```bash
python run_refchecker.py \
  --dataset anah \
  --anah_split train \
  --out_root ./out/refchecker_anah \
  --extractor_name gpt-4o \
  --checker_type llm \
  --checker_name gpt-4o \
  --claim_format subsentence \
  --batch_size_extractor 8 \
  --batch_size_checker 8 \
  --undefined_prediction_policy penalize
```

### ANAH from a pre-sampled JSONL
```bash
python run_refchecker.py \
  --dataset anah \
  --anah_sample_file ../anah_5_sample.jsonl \
  --out_root ./out/refchecker_anah_sample \
  --extractor_name gpt-4o \
  --checker_type llm \
  --checker_name gpt-4o \
  --claim_format subsentence \
  --batch_size_extractor 8 \
  --batch_size_checker 8 \
  --undefined_prediction_policy penalize
```

When `--anah_sample_file` is used, the runner evaluates each sentence against `ann_reference` when it is available, matching the main ANAH loading path.

## Local Llama 3.1
### FactBench + local Llama 3.1
The runner will start a local vLLM endpoint automatically:
```bash
python run_refchecker.py \
  --dataset factbench \
  --data ../data/factcheck-GPT-benchmark.jsonl \
  --out_root ./out/refchecker_factbench_llama31_local \
  --run_local_vllm \
  --local_vllm_model VityaVitalich/Llama3.1-8b-instruct \
  --checker_type llm \
  --claim_format subsentence \
  --batch_size_extractor 8 \
  --batch_size_checker 8 \
  --undefined_prediction_policy penalize \
  --model_params_b 8 \
  --flops_per_param 2
```

Or use the helper script:
```bash
bash run_llama31_factbench_local.sh
```

For a short test:
```bash
MAX_SAMPLES=10 bash run_llama31_factbench_local.sh
```

You can also store variables in `.env`:
```bash
cp .env.example .env
set -a; source .env; set +a
bash run_llama31_factbench_local.sh
```

### FELM + local Llama 3.1
```bash
SUBSET=wk MAX_EXAMPLES=10 bash run_llama31_felm_local.sh
```

Full run on a subset:
```bash
SUBSET=wk bash run_llama31_felm_local.sh
```

Available local variables:
`MODEL`, `FELM_DIR`, `SUBSET`, `SPLIT`, `OUT_ROOT`, `BATCH_SIZE_EXTRACTOR`, `BATCH_SIZE_CHECKER`, `EXTRACTOR_MAX_NEW_TOKENS`, `CLAIM_FORMAT`, `MAX_REFERENCE_SEGMENT_LENGTH`, `UNDEFINED_PREDICTION_POLICY`, `MODEL_PARAMS_B`, `FLOPS_PER_PARAM`, `TOKENIZER_NAME`, `LOCAL_VLLM_PORT`, `LOCAL_VLLM_GPU_MEMORY_UTILIZATION`, `LOCAL_VLLM_TENSOR_PARALLEL_SIZE`, `LOCAL_VLLM_MAX_MODEL_LEN`, `LOCAL_VLLM_STARTUP_TIMEOUT_S`.

## Existing vLLM Endpoint
```bash
python run_refchecker.py \
  --dataset felm \
  --felm_dir ../data/felm_with_ref_text \
  --subset wk \
  --split test \
  --out_root ./out/refchecker_felm_wk_local \
  --extractor_name openai/meta-llama/Meta-Llama-3-8B-Instruct \
  --checker_type llm \
  --checker_name openai/meta-llama/Meta-Llama-3-8B-Instruct \
  --claim_format subsentence \
  --extractor_api_base http://127.0.0.1:5000/v1 \
  --checker_api_base http://127.0.0.1:5000/v1
```
