export CUDA_VISIBLE_DEVICES=3
# FactBench
python factowl_run.py factbench \
  --data ../data/factcheck-GPT-benchmark.jsonl \
  --out_root cachedir_factbench_factowl \
  --model Qwen/Qwen3-14B \
  --gpu_memory_utilization 0.5 \
  --verbose_patch \
  --atomic_set_examples

# FELM
python factowl_run.py felm \
  --felm_dir ../data/felm_with_ref_text \
  --subset wk \
  --split test \
  --out_root cachedir_factowl_felm \
  --model Qwen/Qwen3-14B \
  --gpu_memory_utilization 0.5 \
  --verbose_patch
