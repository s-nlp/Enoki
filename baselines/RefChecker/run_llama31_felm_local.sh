#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

MODEL="${MODEL:-VityaVitalich/Llama3.1-8b-instruct}"
FELM_DIR="${FELM_DIR:-../data/felm_with_ref_text}"
SUBSET="${SUBSET:-wk}"
SPLIT="${SPLIT:-test}"
OUT_ROOT="${OUT_ROOT:-./out/refchecker_felm_${SUBSET}_llama31_local}"
BATCH_SIZE_EXTRACTOR="${BATCH_SIZE_EXTRACTOR:-8}"
BATCH_SIZE_CHECKER="${BATCH_SIZE_CHECKER:-8}"
EXTRACTOR_MAX_NEW_TOKENS="${EXTRACTOR_MAX_NEW_TOKENS:-500}"
MAX_REFERENCE_SEGMENT_LENGTH="${MAX_REFERENCE_SEGMENT_LENGTH:-0}"
UNDEFINED_PREDICTION_POLICY="${UNDEFINED_PREDICTION_POLICY:-penalize}"
MODEL_PARAMS_B="${MODEL_PARAMS_B:-8}"
FLOPS_PER_PARAM="${FLOPS_PER_PARAM:-2}"
TOKENIZER_NAME="${TOKENIZER_NAME:-${MODEL}}"
LOCAL_VLLM_PORT="${LOCAL_VLLM_PORT:-5000}"
LOCAL_VLLM_GPU_MEMORY_UTILIZATION="${LOCAL_VLLM_GPU_MEMORY_UTILIZATION:-0.55}"
LOCAL_VLLM_TENSOR_PARALLEL_SIZE="${LOCAL_VLLM_TENSOR_PARALLEL_SIZE:-1}"
LOCAL_VLLM_MAX_MODEL_LEN="${LOCAL_VLLM_MAX_MODEL_LEN:-0}"
LOCAL_VLLM_EXTRA_ARGS="${LOCAL_VLLM_EXTRA_ARGS:---generation-config vllm}"
LOCAL_VLLM_STARTUP_TIMEOUT_S="${LOCAL_VLLM_STARTUP_TIMEOUT_S:-900}"

MAX_EXAMPLES_ARGS=()
if [[ -n "${MAX_EXAMPLES:-}" ]]; then
  MAX_EXAMPLES_ARGS=(--max_examples "${MAX_EXAMPLES}")
fi

python run_refchecker.py \
  --dataset felm \
  --felm_dir "${FELM_DIR}" \
  --subset "${SUBSET}" \
  --split "${SPLIT}" \
  --out_root "${OUT_ROOT}" \
  --run_local_vllm \
  --local_vllm_model "${MODEL}" \
  --local_vllm_served_model_name "${MODEL}" \
  --local_vllm_port "${LOCAL_VLLM_PORT}" \
  --local_vllm_gpu_memory_utilization "${LOCAL_VLLM_GPU_MEMORY_UTILIZATION}" \
  --local_vllm_tensor_parallel_size "${LOCAL_VLLM_TENSOR_PARALLEL_SIZE}" \
  --local_vllm_max_model_len "${LOCAL_VLLM_MAX_MODEL_LEN}" \
  --local_vllm_extra_args "${LOCAL_VLLM_EXTRA_ARGS}" \
  --local_vllm_startup_timeout_s "${LOCAL_VLLM_STARTUP_TIMEOUT_S}" \
  --checker_type llm \
  --batch_size_extractor "${BATCH_SIZE_EXTRACTOR}" \
  --batch_size_checker "${BATCH_SIZE_CHECKER}" \
  --extractor_max_new_tokens "${EXTRACTOR_MAX_NEW_TOKENS}" \
  --max_reference_segment_length "${MAX_REFERENCE_SEGMENT_LENGTH}" \
  --undefined_prediction_policy "${UNDEFINED_PREDICTION_POLICY}" \
  --model_params_b "${MODEL_PARAMS_B}" \
  --flops_per_param "${FLOPS_PER_PARAM}" \
  --tokenizer_name "${TOKENIZER_NAME}" \
  --tokenizer_local_files_only \
  "${MAX_EXAMPLES_ARGS[@]}"
