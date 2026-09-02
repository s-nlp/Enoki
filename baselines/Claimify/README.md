# Claimify Baseline

Claim extraction in the style of the Claimify paper, with optional offline verification against dataset-provided evidence.

Supported datasets:
- `factbench`
- `felm`
- `anah`

## Pipeline
Claimify processes each sentence in three extraction stages, then optionally verifies the extracted claims:

1. `Selection`
Keeps the sentence only if it contains at least one specific, verifiable proposition. In `rewrite` mode, the model can rewrite the sentence to keep only verifiable content.

2. `Disambiguation`
Decontextualizes the selected sentence using the question and local context, for example by resolving pronouns or incomplete names when possible.

3. `Decomposition`
Breaks the decontextualized sentence into atomic claims.

4. `Verification` (optional)
Checks each extracted claim against offline evidence passages. Sentence-level support is then aggregated from claim-level labels.

## Evidence Sources
Used only when `--do_verify` is enabled.

- FactBench: `auto_evidence`, `human_evidence`
- FELM: `ref_text`
- ANAH: `ann_reference`

## Prompts
All Claimify prompts are stored in `settings.py`.

## Backends
- `vllm`
- `openrouter`
- `openai`

## Outputs
Each run writes:
- `metrics.json`
- `segments.jsonl`

under the directory given by `--out_root`.

## Installation
```bash
python3.11 -m venv venv
pip install -r requirements.txt
```

## Examples
### FactBench
```bash
python run_claimify.py \
  --dataset factbench \
  --data ../data/factcheck-GPT-benchmark.jsonl \
  --out_root ./out/claimify_factbench \
  --backend vllm \
  --model VityaVitalich/Llama3.1-8b-instruct \
  --do_verify \
  --evidence_scope sentence \
  --evidence_mode bm25 \
  --bm25_query claim \
  --selection_mode rewrite \
  --no_claim_policy_all penalize
```

### FELM
```bash
python run_claimify.py \
  --dataset felm \
  --felm_dir ../data/felm_with_ref_text \
  --subset wk \
  --split test \
  --out_root ./out/claimify_felm_wk \
  --backend vllm \
  --model Qwen/Qwen2.5-7B-Instruct \
  --do_verify \
  --evidence_mode bm25 \
  --bm25_query claim \
  --selection_mode rewrite \
  --no_claim_policy_all penalize
```

### ANAH From a Sample File
```bash
python run_claimify.py \
  --dataset anah \
  --anah_sample_file ../anah_250_sample.jsonl \
  --out_root ../results_anah_250/claimify \
  --backend openai \
  --model gpt-4o-mini \
  --do_verify \
  --evidence_mode bm25 \
  --bm25_query claim \
  --selection_mode rewrite \
  --no_claim_policy_all penalize
```

### ANAH Directly From Hugging Face
```bash
python run_claimify.py \
  --dataset anah \
  --anah_split train \
  --anah_max_examples 5 \
  --out_root ../results_anah_test5/claimify \
  --backend openai \
  --model gpt-4o-mini \
  --do_verify \
  --evidence_mode bm25 \
  --bm25_query claim \
  --selection_mode rewrite \
  --no_claim_policy_all penalize
```

## API Keys
For hosted backends, set one of:
- `OPENROUTER_API_KEY`
- `OPENAI_API_KEY`

and switch `--backend` accordingly.
