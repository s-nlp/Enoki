#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

MODEL="${MODEL:-VityaVitalich/Llama3.1-8b-instruct}"
DATA="${DATA:-../data/factcheck-GPT-benchmark.jsonl}"
OUT_ROOT="${OUT_ROOT:-./out/refchecker_factbench_llama31_local}"
BATCH_SIZE_EXTRACTOR="${BATCH_SIZE_EXTRACTOR:-8}"
BATCH_SIZE_CHECKER="${BATCH_SIZE_CHECKER:-8}"
EXTRACTOR_MAX_NEW_TOKENS="${EXTRACTOR_MAX_NEW_TOKENS:-500}"
MAX_REFERENCE_SEGMENT_LENGTH="${MAX_REFERENCE_SEGMENT_LENGTH:-0}"
LOCAL_VLLM_PORT="${LOCAL_VLLM_PORT:-5000}"
LOCAL_VLLM_GPU_MEMORY_UTILIZATION="${LOCAL_VLLM_GPU_MEMORY_UTILIZATION:-0.55}"
LOCAL_VLLM_TENSOR_PARALLEL_SIZE="${LOCAL_VLLM_TENSOR_PARALLEL_SIZE:-1}"
LOCAL_VLLM_MAX_MODEL_LEN="${LOCAL_VLLM_MAX_MODEL_LEN:-0}"
LOCAL_VLLM_STARTUP_TIMEOUT_S="${LOCAL_VLLM_STARTUP_TIMEOUT_S:-900}"

MAX_SAMPLES_ARGS=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  MAX_SAMPLES_ARGS=(--max_samples "${MAX_SAMPLES}")
fi

python run_refchecker.py \
  --dataset factbench \
  --data "${DATA}" \
  --out_root "${OUT_ROOT}" \
  --run_local_vllm \
  --local_vllm_model "${MODEL}" \
  --local_vllm_served_model_name "${MODEL}" \
  --local_vllm_port "${LOCAL_VLLM_PORT}" \
  --local_vllm_gpu_memory_utilization "${LOCAL_VLLM_GPU_MEMORY_UTILIZATION}" \
  --local_vllm_tensor_parallel_size "${LOCAL_VLLM_TENSOR_PARALLEL_SIZE}" \
  --local_vllm_max_model_len "${LOCAL_VLLM_MAX_MODEL_LEN}" \
  --local_vllm_startup_timeout_s "${LOCAL_VLLM_STARTUP_TIMEOUT_S}" \
  --checker_type llm \
  --batch_size_extractor "${BATCH_SIZE_EXTRACTOR}" \
  --batch_size_checker "${BATCH_SIZE_CHECKER}" \
  --extractor_max_new_tokens "${EXTRACTOR_MAX_NEW_TOKENS}" \
  --max_reference_segment_length "${MAX_REFERENCE_SEGMENT_LENGTH}" \
  --no_claim_policy_all penalize \
  "${MAX_SAMPLES_ARGS[@]}"
