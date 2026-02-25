# SAFE (SAFE-like baseline)

SAFE-like factuality pipeline for **FactBench** and **FELM** using only offline evidence from the dataset.

**Evidence sources**
- FactBench: `auto_evidence`, `auto_evidence_url`, `human_evidence` across the whole sample.
- FELM: `ref_text` (per example).

## Pipeline
1. Build evidence context.
2. Atomic extraction per sentence.
3. Decontextualize each atom using the full answer.
4. Relevance check: `[Foo]` vs `[Not Foo]`.
5. Verification: `[Supported]` vs `[Not Supported]`.
6. Aggregate per sentence: any `not_supported` => sentence `not_supported`; else if any `supported` => `supported`; else `ir`.

**FAIL policy**
- `empty_segment`, `no_context`, `no_atoms_or_abstain`, `exception`.
- FAIL is always counted as **wrong** in the `all` metrics.
- `ir` is **not** a FAIL; it is still counted as wrong in `all` metrics.

## Outputs
- FactBench: `out_root/factbench/metrics.json`, `segments_with_safe_like.jsonl`, `examples_with_safe_like.jsonl`.
- FELM: `out_root/felm/<subset>/<split>/metrics.json`, `segments_with_safe_like.jsonl`, `examples_with_safe_like.jsonl`.

## Installation
```bash
python3.11 -m venv venv
pip install -r requirements.txt
```

## Run
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
