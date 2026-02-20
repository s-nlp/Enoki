# Description

## FactBench/FELM
SAFE-like pipeline on FactBench (factcheck-GPT-benchmark.jsonl) with OFFLINE evidence.

Pipeline (SAFE-like):
0) Evidence (context) is built from auto_evidence/auto_evidence_url/human_evidence of the whole QA item
   and shared across all sentences in that sample.
1) Atomic extraction: extract atomic facts ONLY from the current sentence.
2) Decontextualization: rewrite each atom to be self-contained using FULL ANSWER for context.
3) Relevance check: QUESTION + FULL ANSWER + FACT -> [Foo]/[Not Foo]
4) Verification: FACT vs evidence KNOWLEDGE -> [Supported]/[Not Supported]
5) Segment label aggregation:
   - if any relevant fact is not_supported => sentence not_supported
   - else if any relevant fact is supported => sentence supported
   - else => ir (e.g., all atoms irrelevant)

FAIL policy (ALWAYS wrong in ALL-metrics):
- empty_segment / no_context / no_atoms_or_abstain / exception => FAIL

Notes:
- Gold label "NA" is skipped from metrics but logged as SKIP.
- Binary metrics treat NOT_SUPPORTED as positive class.


# How to run:

## Installation
```bash
python3.11 -m venv venv
pip install -r requirements.txt
```


## FELM
```bash
python safe_run.py \
  --dataset felm \
  --felm_dir ./data/felm_with_ref_text \
  --subset wk \
  --split test \
  --model Qwen/Qwen2.5-7B-Instruct \
  --out_root ./cachedir_safe_felm \
  --gpu_memory_utilization 0.5
```

## FactBench
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