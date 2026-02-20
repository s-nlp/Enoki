#!/usr/bin/env bash
set -euo pipefail

# Run answer filtering (refusal/not_relevant) over all hallucination model folders.
# Uses filter_hallucination_answers.sh and the gpt-oss endpoint from env.

: "${BASE_URL:?set BASE_URL, e.g. http://99dgx-16:8128/v1/}"
API_KEY="${API_KEY:-EMPTY}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
ROOT="data/wiki/en/hallucinations"

MODELS=(
  # "Qwen_Qwen2.5-7B-Instruct"
  # "Qwen_Qwen2.5-14B-Instruct"
  # "Qwen_Qwen2.5-32B-Instruct"
  # "Qwen_Qwen3-4B-Instruct-2507"
  # "Qwen_Qwen3-8B"
  # "mistralai_Mixtral-8x7B-Instruct-v0.1"
  "unsloth_Meta-Llama-3.1-8B-Instruct"
)

for slug in "${MODELS[@]}"; do
  if [[ ! -d "${ROOT}/${slug}" ]]; then
    echo "skip: ${slug} (not found)"
    continue
  fi
  echo "=== Filtering ${slug} ==="
  MODEL_SLUG="${slug}" \
  BASE_URL="${BASE_URL}" \
  API_KEY="${API_KEY}" \
  MODEL="${MODEL}" \
  bash scripts/filter_hallucination_answers.sh
done

echo "All done."
