export CUDA_VISIBLE_DEVICES=3
# FactBench
python safe_run.py \
  --dataset factbench \
  --data ../data/factcheck-GPT-benchmark.jsonl \
  --model VityaVitalich/Llama3.1-8b-instruct \
  --model_params_b 8 \
  --flops_per_param 2 \
  --out_root ./cachedir_safe_fact \
  --gpu_memory_utilization 0.5

# FELM
python safe_run.py \
  --dataset felm \
  --felm_dir ./data/felm_with_ref_text \
  --subset wk \
  --split test \
  --model Qwen/Qwen2.5-7B-Instruct \
  --out_root ./cachedir_safe_felm \
  --gpu_memory_utilization 0.5
