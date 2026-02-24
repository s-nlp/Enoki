# Claimify (Claimify baseline)

Claimify-style claim extraction with optional offline verification for **FactBench** and **FELM**.

## Pipeline
1. Selection stage (rewrite or detector).
2. Disambiguation stage.
3. Decomposition stage to produce atomic claims.
4. Optional verification against offline evidence.

**Evidence sources (verification only)**
- FactBench: `auto_evidence`, `auto_evidence_url`, `human_evidence`.
- FELM: `ref_text`.

**Prompts**
Prompts are read from `settings.py` in this folder.

## Backends
- `vllm` (default)
- `openrouter`
- `openai`

## Outputs
`--out_root` contains `metrics.json` and `segments.jsonl`.

## Installation
```bash
python3.11 -m venv venv
pip install -r requirements.txt
```

## Run
### FactBench (vLLM)
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

### FELM (vLLM)
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

### OpenRouter / OpenAI
Set `OPENROUTER_API_KEY` or `OPENAI_API_KEY` and switch `--backend` to `openrouter` or `openai`.
