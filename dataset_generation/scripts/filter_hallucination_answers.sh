#!/usr/bin/env bash
set -euo pipefail

# LLM-filter hallucination answers (no_context + with_context).
# Required env: MODEL_SLUG (directory under data/wiki/en/hallucinations), BASE_URL.
# Optional: API_KEY (default EMPTY), MODEL (default openai/gpt-oss-120b), MAX_RECORDS.

: "${MODEL_SLUG:?set MODEL_SLUG to hallucination folder name (slug)}"
: "${BASE_URL:?set BASE_URL, e.g. http://127.0.0.1:8000/v1}"
API_KEY="${API_KEY:-EMPTY}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
MAX_RECORDS="${MAX_RECORDS:-0}"
TEMP="${TEMP:-0.7}"
MAX_TOKENS="${MAX_TOKENS:-256}"
CONCURRENCY="${CONCURRENCY:-32}"

ROOT="data/wiki/en/hallucinations/${MODEL_SLUG}"

run_one() {
  local infile="$1"
  local label="$2"
  local out="${infile%.jsonl}.filtered.jsonl"
  local err="${infile%.jsonl}.filtered.errors.jsonl"

  echo "[$(date -Is)] filtering ${label}: ${infile}"
  python3 filter_answers_llm.py \
    --input "${infile}" \
    --out "${out}" \
    --errors-out "${err}" \
    --base-url "${BASE_URL}" \
    --api-key "${API_KEY}" \
    --model "${MODEL}" \
    --temperature "${TEMP}" \
    --max-tokens "${MAX_TOKENS}" \
    --concurrency "${CONCURRENCY}" \
    --max-records "${MAX_RECORDS}"
}

run_one "${ROOT}/answers_no_context.jsonl" "no_context"
run_one "${ROOT}/answers_with_context.jsonl" "with_context"

echo "Done. Filtered files are alongside originals in ${ROOT}"
