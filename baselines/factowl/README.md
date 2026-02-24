# FactOwl (FactOwl baseline)

FactOwl-based factuality evaluation for **FactBench** and **FELM** using offline evidence only.

**Evidence sources**
- FactBench: `auto_evidence`, `auto_evidence_url`, `human_evidence` across the whole sample.
- FELM: `ref_text` (per example).

## Notes
- This runner expects the `factowl` Python package to be installed.
- It patches the FactOwl atomic extractor prompt at runtime (see `--atomic_template` and `--atomic_set_examples`).

## Installation
Typical setup (adjust to your environment):
```bash
pip install git+https://github.com/s-nlp/factowl.git
pip install jieba
python -m spacy download en_core_web_sm
```

## Run
### FactBench
```bash
python factowl_run.py factbench \
  --data ../data/factcheck-GPT-benchmark.jsonl \
  --out_root ./cachedir_factowl_factbench \
  --model Qwen/Qwen3-14B \
  --gpu_memory_utilization 0.5 \
  --verbose_patch \
  --atomic_set_examples
```

### FELM
```bash
python factowl_run.py felm \
  --felm_dir ../data/felm_with_ref_text \
  --subset wk \
  --split test \
  --out_root ./cachedir_factowl_felm \
  --model Qwen/Qwen3-14B \
  --gpu_memory_utilization 0.5 \
  --verbose_patch
```

## Outputs
- FactBench: `out_root/metrics.json`, `segments_with_factowl.jsonl`, `examples_with_factowl.jsonl`.
- FELM: `out_root/<subset>/<split>/metrics.json`, `segments_with_factowl.jsonl`, `examples_with_factowl.jsonl`.
