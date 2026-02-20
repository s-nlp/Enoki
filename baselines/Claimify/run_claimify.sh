export CUDA_VISIBLE_DEVICES=2
# VLLM Backend
python run_claimify.py \
  --dataset factbench \
  --data ../data/factcheck-GPT-benchmark.jsonl \
  --out_root ./out/factbench_llama31 \
  --backend vllm \
  --model VityaVitalich/Llama3.1-8b-instruct \
  --do_verify \
  --evidence_scope sentence \
  --evidence_mode bm25 \
  --bm25_query claim \
  --selection_mode rewrite \
  --no_claim_policy_all penalize
  # --max_samples 10 for testing only

# OpenRouter Backend
export OPENROUTER_API_KEY="sk-or-"
python run_claimify.py \
    --dataset factbench \
    --data ../data/factcheck-GPT-benchmark.jsonl \
    --out_root ./out/openrouter_gpt4o \
    --backend openrouter \
    --model openai/gpt-4o \
    --do_verify \
    --evidence_mode bm25 \
    --evidence_scope sentence \
    --selection_mode rewrite \
    --bm25_query question_claim \
    --no_claim_policy_all penalize
    # --max_samples 10 for testing only

# OpenAI Backend
export OPENAI_API_KEY="sk-proj-"
python run_claimify.py \
    --dataset factbench \
    --data ../data/factcheck-GPT-benchmark.jsonl \
    --out_root ./out/openai_gpt4o \
    --backend openai \
    --model gpt-4o \
    --do_verify \
    --evidence_mode bm25 \
    --evidence_scope sentence \
    --selection_mode rewrite \
    --bm25_query question_claim \
    --no_claim_policy_all penalize
    # --max_samples 10 for testing only
