# FactOwl

FactOwl-based factuality evaluation for **FactBench**, **FELM**, and **ANAH** using offline evidence only.

**Evidence Sources**
- FactBench: by default, evidence is sentence-scoped
  (`auto_evidence` + `human_evidence`). The old sample-level mode is still
  available via `--evidence_scope sample`. `auto_evidence_url` is included only
  if you pass `--include_auto_evidence_url`.
- FELM: `ref_text` for each example.
- ANAH: `ann_reference` for each sentence.

## Notes
- This runner expects the Python `factowl` package to be installed.
- It patches the FactOwl atomic extraction prompt at runtime (see `--atomic_template` and `--atomic_set_examples`).

## Installation
Typical setup (adjust to your environment):
```bash
pip install git+https://github.com/s-nlp/factowl.git
pip install jieba
python -m spacy download en_core_web_sm
```

## Running
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

### ANAH
```bash
python factowl_run.py anah \
  --anah_sample_file ../anah_250_sample.jsonl \
  --out_root ./cachedir_factowl_anah \
  --model Qwen/Qwen3-14B \
  --gpu_memory_utilization 0.5 \
  --verbose_patch
```

## Output Files
- FactBench: `out_root/metrics.json`, `segments_with_factowl.jsonl`, `examples_with_factowl.jsonl`.
- FELM: `out_root/<subset>/<split>/metrics.json`, `segments_with_factowl.jsonl`, `examples_with_factowl.jsonl`.
- ANAH: `out_root/anah/<split-or-sample>/metrics.json`, `segments_with_factowl.jsonl`.
