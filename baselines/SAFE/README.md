# SAFE (SAFE-like Baseline)

A SAFE-like factuality evaluation pipeline for **FactBench** and **FELM** that uses only offline evidence available in the datasets.

## Evidence Sources
- FactBench: pooled `auto_evidence` and `human_evidence` from the full sample
- FELM: `ref_text` for each example

## Pipeline
1. Build the evidence context.
2. Extract atomic facts from each sentence.
3. Decontextualize each atomic fact using the full answer.
4. Check relevance: `[Foo]` vs `[Not Foo]`.
5. Verify support: `[Supported]` vs `[Not Supported]`.
6. Aggregate to the sentence level:
   - if any relevant fact is `not_supported`, the sentence is `not_supported`
   - otherwise, if at least one relevant fact is `supported`, the sentence is `supported`
   - otherwise, the sentence is `ir`

## FAIL Policy
- `empty_segment`
- `no_context`
- `no_atoms_or_abstain`
- `exception`

FAIL is always counted as an error in the `all` metrics.

`ir` is not treated as FAIL, but it is still counted as an error in the `all` metrics.

## Output Files
- FactBench: `out_root/factbench/metrics.json`, `segments_with_safe_like.jsonl`, `examples_with_safe_like.jsonl`
- FELM: `out_root/felm/<subset>/<split>/metrics.json`, `segments_with_safe_like.jsonl`, `examples_with_safe_like.jsonl`

## Installation
```bash
python3.11 -m venv venv
pip install -r requirements.txt
```

## Usage
### FactBench
```bash
python safe_run.py \
  --dataset factbench \
  --data ../data/factcheck-GPT-benchmark.jsonl \
  --model VityaVitalich/Llama3.1-8b-instruct \
  --model_params_b 8 \
  --flops_per_param 2 \
  --out_root ./cachedir_safe_fact \
  --gpu_memory_utilization 0.5
```

### FELM
```bash
python safe_run.py \
  --dataset felm \
  --felm_dir ../data/felm_with_ref_text \
  --subset wk \
  --split test \
  --model Qwen/Qwen2.5-7B-Instruct \
  --out_root ./cachedir_safe_felm \
  --gpu_memory_utilization 0.5
```
